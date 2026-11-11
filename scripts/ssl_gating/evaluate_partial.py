"""Native P1 for completed top-four/BF16 SSL runs; source gates precede targets."""

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path

import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, evaluate_transfer, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, read_dataset,
    save_score_manifest, utterance_id, verify_completed_run, verify_source_macro,
)
from audio_deepfake_detection.sota.ssl_gating import FullAudioDataset, make_eval_loader
from audio_deepfake_detection.sota.ssl_gating_partial import (
    PARTIAL_MODEL_CONFIG, RUN_FAMILY, PartialSSLFeatureGating, partial_scores,
)
from scripts.ssl_gating.train import set_seed, write_json
from scripts.ssl_gating.train_partial import source_guards


def verify_checkpoint(checkpoint, completion, fold_name, seed):
    c = checkpoint["config"]
    fold = FOLDS[fold_name]
    if (checkpoint["fold"] != fold_name or c["fold"] != fold_name or c["seed"] != seed
            or c["run_name"] != RUN_FAMILY or c["model_name"] != "PartialSSLFeatureGating"
            or c["model_config"] != PARTIAL_MODEL_CONFIG
            or c["precision"] != "BF16_autocast_FP32_pooling_losses_scores"
            or c["train_max_seconds"] != 20 or c["eval_batch_size"] != 1
            or c["selection"] != "strict_equal_domain_macro_source_dev_eer_ties_earlier"
            or c["trainable_parameters"] != 53029762 or c["total_parameters"] != 318083586
            or tuple(c["sources"]) != fold["sources"] or c["target_domain"] != fold["target"]
            or c["target_access"] is not False or completion.get("target_access") is not False
            or checkpoint["epoch"] != completion["best_epoch"]
            or completion.get("selected_checkpoint_reloaded_and_source_verified") is not True):
        raise ValueError("Partial SSL checkpoint identity, recipe or completed source reload differs")
    verify_source_macro(checkpoint["macro_dev_eer"], completion["best_macro_source_dev_eer"])


def verify_run_config(config, stored_config, current_guards):
    # HF frontend dictionaries contain integer id2label keys; JSON stores them
    # as strings. Compare the serialized representation without weakening hashes.
    if json.loads(json.dumps(config)) != stored_config or config["source_sha256"] != current_guards:
        raise ValueError("Training configuration or guarded runtime changed")


def verified_sources(run, completion, config, data_root):
    """Reuse only training's hash-bound, fully reloaded canonical source exports."""
    calibration_path = run / "source_calibration.json"
    if checkpoint_digest(calibration_path) != completion["source_calibration_sha256"]:
        raise ValueError("Recorded source calibration changed")
    expected_names = {f"source_dev_{d}.csv" for d in config["sources"]}
    if set(completion["source_scores_sha256"]) != expected_names:
        raise ValueError("Completion lacks the exact source score set")
    sources, records = {}, {}
    for domain in config["sources"]:
        path = run / f"source_dev_{domain}.csv"
        if checkpoint_digest(path) != completion["source_scores_sha256"][path.name]:
            raise ValueError(f"Recorded source scores changed: {domain}")
        records[domain] = read_dataset(domain, "dev", data_root)
        if len(records[domain]) != config["dataset_counts"][domain]["dev"]:
            raise ValueError(f"Source development count changed: {domain}")
        expected = canonical_score_labels(domain, "dev", data_root, records=records[domain])
        read_score_csv(path, domain, expected_labels=expected)
        # Canonical membership is mandatory, but retain training's CSV order so
        # floating-point source statistics reproduce the frozen values exactly.
        sources[domain] = read_score_csv(path, domain)
    calibration = calibrate_source_scores(sources)
    verify_source_macro(calibration["source_dev"]["macro_eer"], completion["best_macro_source_dev_eer"])
    if calibration != json.loads(calibration_path.read_text()):
        raise ValueError("Canonical source scores do not reproduce frozen calibration")
    return sources, records


@torch.inference_mode()
def verify_source_parity(model, records, run, data_root, device):
    """First/last full source recordings must reproduce their preserved BF16 scores."""
    evidence = []
    model.eval()
    for domain, rows in records.items():
        with (run / f"source_dev_{domain}.csv").open(newline="") as handle:
            saved = {r["utterance_id"]: float(r["raw_score"]) for r in csv.DictReader(handle)}
        chosen = [rows[0], rows[-1]] if len(rows) > 1 else [rows[0]]
        dataset = FullAudioDataset(chosen)
        for i, (path, _) in enumerate(chosen):
            waveform, _ = dataset[i]
            score = float(partial_scores(model, waveform[None].to(device))[0])
            identifier = utterance_id(path, domain, data_root)
            if not math.isfinite(score) or score != saved[identifier]:
                raise ValueError(f"Fresh BF16 source score differs: {domain}/{identifier}; target unopened")
            evidence.append({"domain": domain, "id": identifier, "samples": waveform.numel(),
                             "score": score, "exact_match": True})
    return evidence


@torch.inference_mode()
def score_rows(model, loader, records, domain, data_root, device):
    model.eval()
    offset = 0
    for waves, labels in loader:
        scores = partial_scores(model, waves.to(device, non_blocking=True)).cpu().tolist()
        if len(scores) != 1 or len(labels) != 1 or offset >= len(records):
            raise RuntimeError("Native partial SSL evaluation requires complete batch-one scoring")
        path, label = records[offset]
        if labels[0].item() != label:
            raise RuntimeError("Evaluation record-label order changed")
        yield utterance_id(path, domain, data_root), domain, label, scores[0]
        offset += 1
        if offset % 5000 == 0:
            print(f"{domain}: {offset:,}/{len(records):,} full recordings scored", flush=True)
    if offset != len(records):
        raise RuntimeError("Incomplete target scores")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--source-data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--target-data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--gpu-memory-budget-gib", type=float, default=19)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    if (min(args.seed, args.num_workers) < 0 or args.prefetch_factor < 1 or args.bootstrap_resamples < 1
            or not math.isfinite(args.gpu_memory_budget_gib) or args.gpu_memory_budget_gib <= 0):
        parser.error("Invalid seed, workers, prefetch, bootstrap or memory budget")
    root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    run = root / "outputs" / RUN_FAMILY / args.fold / str(args.seed)
    with (run / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("metrics.json", "metrics.json.tmp", "target_scores.csv", "target_scores.csv.tmp",
                     "score_manifest.json", "source_validation.json"):
            if (run / name).exists():
                raise FileExistsError(f"Preserving prior evaluation artifact; no automatic retry: {run/name}")
        completion, digest = verify_completed_run(run, args.fold, args.seed)
        if digest != completion["best_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint bytes changed")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        verify_checkpoint(checkpoint, completion, args.fold, args.seed)
        config = checkpoint["config"]
        verify_run_config(config, json.loads((run / "config.json").read_text()), source_guards())
        sources, records = verified_sources(run, completion, config, args.source_data_root)
        print("Canonical source exports, hashes and frozen calibration verified", flush=True)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Native CUDA BF16 required; no precision fallback")
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_budget_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        set_seed(args.seed)
        device = torch.device("cuda")
        model = PartialSSLFeatureGating.from_frontend_config(config["frontend_config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        model.to(device).eval()
        parity = verify_source_parity(model, records, run, args.source_data_root, device)
        write_json(run / "source_validation.json", {"checkpoint_sha256": digest,
                   "source_calibration_sha256": completion["source_calibration_sha256"],
                   "source_scores_sha256": completion["source_scores_sha256"],
                   "full_source_reload_verified_during_training": True, "bounded_fresh_reload_parity": parity,
                   "target_access": False})
        fold = FOLDS[args.fold]
        manifest = {"model": RUN_FAMILY, "fold": args.fold, "seed": args.seed,
                    "checkpoint_sha256": digest, "training_completion_sha256": checkpoint_digest(run/"training_complete.json"),
                    "splits": {**{f"source_dev_{d}.csv": {"dataset": d, "split": "dev"} for d in fold["sources"]},
                               "target_scores.csv": {"dataset": fold["target"], "split": fold["target_split"]}},
                    "score_sha256": dict(completion["source_scores_sha256"])}
        save_score_manifest(run, manifest)
        manifest = load_score_manifest(run, args.fold, args.seed, digest)
        print("Fresh BF16 source parity passed; opening the authorized held-out target", flush=True)
        target_records = read_dataset(fold["target"], fold["target_split"], args.target_data_root)
        expected = canonical_score_labels(fold["target"], fold["target_split"], args.target_data_root, records=target_records)
        print(f"Target {fold['target']}/{fold['target_split']}: {len(target_records):,} full recordings", flush=True)
        loader = make_eval_loader(FullAudioDataset(target_records), num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch_factor)
        path = run / "target_scores.csv"
        write_score_csv(path, score_rows(model, loader, target_records, fold["target"], args.target_data_root, device))
        labels, scores = read_score_csv(path, fold["target"], expected_labels=expected)
        manifest["score_sha256"][path.name] = checkpoint_digest(path)
        save_score_manifest(run, manifest)
        del model, loader
        torch.cuda.empty_cache()
        print("Full target scores verified; computing shared metrics and conditional bootstrap intervals", flush=True)
        report = evaluate_transfer(sources, labels, scores, bootstrap_resamples=args.bootstrap_resamples)
        report.update({"model": RUN_FAMILY, "fold": args.fold, "seed": args.seed,
                       "sources": list(fold["sources"]), "target_domain": fold["target"], "target_split": fold["target_split"],
                       "checkpoint_sha256": digest, "checkpoint_epoch": completion["best_epoch"],
                       "score_sha256": manifest["score_sha256"], "evaluator_sha256": checkpoint_digest(Path(__file__)),
                       "source_data_root": str(args.source_data_root), "target_data_root": str(args.target_data_root),
                       "precision": config["precision"], "input_policy": PARTIAL_MODEL_CONFIG["input_policy"],
                       "train_input_policy": PARTIAL_MODEL_CONFIG["train_input_policy"], "train_max_seconds": 20,
                       "eval_batch_size": 1, "num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor,
                       "gpu_allocator_budget_gib": args.gpu_memory_budget_gib,
                       "comparison_wide_freeze_confirmed_by_operator": False,
                       "source_exports_reused_after_completed_full_reload_and_hash_verification": True})
        if source_guards() != config["source_sha256"]:
            raise RuntimeError("Guarded runtime changed; preserving scores without a final report")
        write_json(run / "metrics.json.tmp", report)
        (run / "metrics.json.tmp").replace(run / "metrics.json")
        print(f"COMPLETE target EER={100*report['target']['eer']:.4f}%; AUROC={report['target']['auroc']:.6f}", flush=True)


if __name__ == "__main__":
    main()
