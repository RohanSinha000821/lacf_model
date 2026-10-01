"""Score one completed WavLM-WA LODO run, then apply the common evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from audio_deepfake_detection.metrics import (
    calibrate_apcer_thresholds,
    compute_eer,
    evaluate_transfer,
    normalize_scores,
    read_score_csv,
    write_score_csv,
    zscore_stats,
)
from audio_deepfake_detection.protocol import FOLDS, read_dataset, utterance_id
from audio_deepfake_detection.sota.wavlm_wa import WavLMDataset, WavLMWA, make_wavlm_eval_loader


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_rows(model, loader, records, dataset_name, data_root, device):
    offset = 0
    model.eval()
    with torch.inference_mode():
        for waveforms, attention_masks, labels in loader:
            waveforms = waveforms.to(device, non_blocking=True)
            attention_masks = attention_masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(waveforms, attention_masks)
            scores = model.spoof_score(logits).float().cpu().tolist()
            batch_records = records[offset:offset + len(scores)]
            if len(batch_records) != len(scores):
                raise RuntimeError("Scoring produced more rows than input records")
            for (path, expected_label), observed_label, score in zip(batch_records, labels.tolist(), scores):
                if expected_label != observed_label:
                    raise RuntimeError("Evaluation loader changed record order")
                yield utterance_id(path, dataset_name, data_root), dataset_name, observed_label, score
            offset += len(scores)
            if offset % 5000 == 0:
                print(f"{dataset_name}: {offset:,}/{len(records):,} scored", flush=True)
    if offset != len(records):
        raise RuntimeError(f"Scored {offset} of {len(records)} records")


def score_split(model, dataset_name, split, data_root, output, device, *, num_workers, prefetch_factor):
    records = read_dataset(dataset_name, split, data_root)
    if not records:
        raise ValueError(f"No records for {dataset_name}/{split}")
    if output.exists():
        labels, _ = read_score_csv(output, dataset_name)
        if len(labels) != len(records):
            raise ValueError(f"Existing score count does not match {dataset_name}/{split}: {output}")
        print(f"Verified existing {output}: {len(labels):,} scores", flush=True)
        return
    loader = make_wavlm_eval_loader(
        WavLMDataset(records, training=False),
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )
    write_score_csv(output, score_rows(model, loader, records, dataset_name, data_root, device))
    print(f"Wrote {output}: {len(records):,} scores", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a completed WavLM LODO fold and report shared metrics")
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--source-data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--target-data-root", type=Path, default=Path("/mnt/salt/datasets/audio-deepfake"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    if args.num_workers < 0 or args.prefetch_factor < 1 or args.bootstrap_resamples < 0:
        parser.error("Workers/resamples must be non-negative and prefetch factor positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WavLM scoring")

    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    run_dir = project_root / "outputs" / "wavlm" / args.fold / str(args.seed)
    best_path = run_dir / "best.pt"
    completion_path = run_dir / "training_complete.json"
    if not best_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"Training has not completed for {run_dir}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion["fold"] != args.fold or completion["seed"] != args.seed:
        raise ValueError("Training completion record does not match requested fold/seed")

    digest = checkpoint_digest(best_path)
    manifest_path = run_dir / "score_manifest.json"
    score_files = list(run_dir.glob("source_dev_*.csv")) + list(run_dir.glob("target_scores.csv"))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["fold"], manifest["seed"], manifest["checkpoint_sha256"]) != (args.fold, args.seed, digest):
            raise ValueError("Checkpoint changed since these scores were generated")
    else:
        if score_files:
            raise ValueError("Scores exist without a checkpoint manifest; refusing to mix runs")
        manifest = {"fold": args.fold, "seed": args.seed, "checkpoint_sha256": digest}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if checkpoint["fold"] != args.fold or config["seed"] != args.seed:
        raise ValueError("Checkpoint does not match requested fold/seed")
    if checkpoint["epoch"] != completion["best_epoch"]:
        raise ValueError("Checkpoint is not the recorded best epoch")
    fold = FOLDS[args.fold]
    if tuple(config["sources"]) != fold["sources"] or config["target"] != fold["target"]:
        raise ValueError("Checkpoint LODO membership does not match the frozen protocol")
    model = WavLMWA(model_name=config["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    device = torch.device("cuda")
    model.to(device).eval()

    # First score and calibrate source development domains. The target is not
    # even opened until the selected checkpoint and source calibration are fixed.
    source_dev = {}
    for domain in fold["sources"]:
        output = run_dir / f"source_dev_{domain}.csv"
        score_split(
            model, domain, "dev", args.source_data_root, output, device,
            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        )
        source_dev[domain] = read_score_csv(output, domain)

    macro_source_eer = sum(compute_eer(labels, scores) for labels, scores in source_dev.values()) / len(source_dev)
    if abs(macro_source_eer - completion["best_macro_source_dev_eer"]) > 1e-5:
        raise ValueError("Recomputed source-dev EER differs from the selected checkpoint; target remains unopened")

    raw_labels = []
    raw_scores = []
    normalized_scores = []
    for labels, scores in source_dev.values():
        raw_labels.extend(labels)
        raw_scores.extend(scores)
        normalized_scores.extend(normalize_scores(scores, zscore_stats(scores)))
    calibrate_apcer_thresholds(raw_labels, raw_scores)
    calibrate_apcer_thresholds(raw_labels, normalized_scores)
    print("Source-only calibration complete; opening held-out target now", flush=True)

    target_output = run_dir / "target_scores.csv"
    score_split(
        model, fold["target"], fold["target_split"], args.target_data_root,
        target_output, device, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    target_labels, target_scores = read_score_csv(target_output, fold["target"])
    report = evaluate_transfer(
        source_dev, target_labels, target_scores,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    report.update({
        "fold": args.fold,
        "seed": args.seed,
        "sources": list(fold["sources"]),
        "target_domain": fold["target"],
        "target_split": fold["target_split"],
        "checkpoint_sha256": digest,
        "checkpoint_epoch": completion["best_epoch"],
    })
    temporary = run_dir / "metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "metrics.json")
    print(f"Target EER: {report['target']['eer'] * 100:.4f}%")
    print(f"Target AUROC: {report['target']['auroc']:.6f}")
    print(f"Saved metrics to {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
