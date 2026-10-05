"""Score one completed WavLM-WA LODO run, then apply the common evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from audio_deepfake_detection.metrics import (
    calibrate_source_scores,
    evaluate_transfer,
    read_score_csv,
    write_score_csv,
)
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest,
    read_dataset, save_score_manifest, utterance_id, verify_completed_run,
    verify_score_digest, verify_source_macro,
    validate_run_name,
)
from audio_deepfake_detection.sota.wavlm_wa import WavLMDataset, WavLMWA, make_wavlm_eval_loader


def verify_checkpoint(checkpoint, completion, fold_name, seed, run_name="wavlm"):
    config = checkpoint["config"]
    if config.get("run_name", "wavlm") != run_name:
        raise ValueError("Checkpoint output family does not match requested run name")
    fold = FOLDS[fold_name]
    if checkpoint["fold"] != fold_name or config["seed"] != seed or config["fold"] != fold_name:
        raise ValueError("Checkpoint does not match requested fold/seed")
    if checkpoint["epoch"] != completion["best_epoch"]:
        raise ValueError("Checkpoint is not the recorded best epoch")
    if tuple(config["sources"]) != fold["sources"] or config["target"] != fold["target"]:
        raise ValueError("Checkpoint LODO membership does not match the frozen protocol")
    verify_source_macro(checkpoint["macro_dev_eer"], completion["best_macro_source_dev_eer"])


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


def score_split(model, dataset_name, split, data_root, output, device, *, num_workers, prefetch_factor, manifest):
    records = read_dataset(dataset_name, split, data_root)
    if not records:
        raise ValueError(f"No records for {dataset_name}/{split}")
    expected = canonical_score_labels(dataset_name, split, data_root, records=records)
    if manifest["splits"].get(output.name) != {"dataset": dataset_name, "split": split}:
        raise ValueError(f"Score manifest split does not match {dataset_name}/{split}")
    if output.exists():
        verify_score_digest(output, manifest)
        labels, scores = read_score_csv(output, dataset_name, expected_labels=expected)
        print(f"Verified existing {output}: {len(labels):,} scores", flush=True)
        return labels, scores
    if output.name in manifest["score_sha256"]:
        raise ValueError(f"Previously registered score file is missing: {output}")
    loader = make_wavlm_eval_loader(
        WavLMDataset(records, training=False),
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )
    write_score_csv(output, score_rows(model, loader, records, dataset_name, data_root, device))
    result = read_score_csv(output, dataset_name, expected_labels=expected)
    manifest["score_sha256"][output.name] = checkpoint_digest(output)
    save_score_manifest(output.parent, manifest)
    print(f"Wrote {output}: {len(records):,} scores", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a completed WavLM LODO fold and report shared metrics")
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-name", type=validate_run_name, default="wavlm", help="Output family, e.g. wavlm_bs96")
    parser.add_argument("--source-data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--target-data-root", type=Path, default=Path("/mnt/salt/datasets/audio-deepfake"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    if args.num_workers < 0 or args.prefetch_factor < 1 or args.bootstrap_resamples < 0 or args.seed < 0:
        parser.error("Workers/resamples/seed must be non-negative and prefetch factor positive")

    project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    run_dir = project_root / "outputs" / args.run_name / args.fold / str(args.seed)
    best_path = run_dir / "best.pt"
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"Metrics already exist; refusing to overwrite: {run_dir}")
    completion, digest = verify_completed_run(run_dir, args.fold, args.seed)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    verify_checkpoint(checkpoint, completion, args.fold, args.seed, args.run_name)
    fold = FOLDS[args.fold]
    manifest = load_score_manifest(run_dir, args.fold, args.seed, digest, create=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WavLM scoring")
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
        source_dev[domain] = score_split(
            model, domain, "dev", args.source_data_root, output, device,
            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, manifest=manifest,
        )
    calibration = calibrate_source_scores(source_dev)
    verify_source_macro(calibration["source_dev"]["macro_eer"], completion["best_macro_source_dev_eer"])
    print("Source-only calibration complete; opening held-out target now", flush=True)

    target_output = run_dir / "target_scores.csv"
    target_labels, target_scores = score_split(
        model, fold["target"], fold["target_split"], args.target_data_root,
        target_output, device, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, manifest=manifest,
    )
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
        "model": args.run_name,
        "source_data_root": str(args.source_data_root),
        "target_data_root": str(args.target_data_root),
        "score_sha256": dict(manifest["score_sha256"]),
        "evaluator_sha256": checkpoint_digest(Path(__file__)),
    })
    temporary = run_dir / "metrics.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(run_dir / "metrics.json")
    print(f"Target EER: {report['target']['eer'] * 100:.4f}%")
    print(f"Target AUROC: {report['target']['auroc']:.6f}")
    print(f"Saved metrics to {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
