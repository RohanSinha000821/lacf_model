"""Native P1 SAM-AASIST evaluation after immutable completed-source checks."""

import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, evaluate_transfer, read_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, read_dataset, save_score_manifest,
    verify_completed_run,
)
from audio_deepfake_detection.sota.aasist import AASISTDataset
from audio_deepfake_detection.sota.sam_aasist import MODEL_CONFIG, PARAMETER_COUNT, SAM_UPSTREAM_COMMIT, SAMAASIST
from scripts.aasist.evaluate import score_split
from scripts.aasist.train import write_json
from scripts.sam_aasist.train import FAMILY, ROOT, RUNTIME_FILES, check_guards, source_identity


def verify_checkpoint(checkpoint, completion, config, fold_name, seed):
    """Reject a different architecture, recipe, or selected checkpoint before data access."""
    expected = {
        "fold": fold_name, "seed": seed, "run_name": FAMILY, "model_name": "SAM-AASIST",
        "model_config": MODEL_CONFIG, "sam_upstream_commit": SAM_UPSTREAM_COMMIT,
        "sources": list(FOLDS[fold_name]["sources"]), "target": FOLDS[fold_name]["target"],
        "target_access": False, "parameter_count": PARAMETER_COUNT, "trainable_parameters": PARAMETER_COUNT,
        "precision": "float32", "batch_size": 24, "gradient_accumulation_steps": 1,
        "eval_batch_size": 1, "train_input": "random64600_after16k_resampling_repeat_short",
        "eval_input": "first64600_repeat_short", "rawboost_algorithm": 0, "freq_aug": False,
        "optimizer": "nonadaptive_SAM_Adam", "rho": 0.05, "learning_rate": 1e-4,
        "weight_decay": 1e-4, "minimum_learning_rate": 5e-6, "max_epochs": 100, "patience": 5,
        "class_weights": [0.9, 0.1], "scheduler": "per_Adam_update_cosine",
        "sampler": "equal_domain_natural_classes_replacement_N_draws",
        "selection": "strict_equal_domain_macro_source_dev_eer_ties_earlier",
        "cuda_matmul_tf32": False, "cudnn_tf32": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "label_order": ["bona_fide", "spoof"],
        "score": "spoof_logit_minus_bona_fide_logit",
    }
    # JSON round-trip accepts checkpoint tuples/lists only where JSON itself does.
    if json.loads(json.dumps(checkpoint["config"])) != config:
        raise ValueError("Checkpoint and saved configuration differ")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"SAM-AASIST configuration mismatch: {key}")
    if (checkpoint.get("fold") != fold_name or checkpoint.get("epoch") != completion["best_epoch"]
            or checkpoint.get("macro_dev_eer") != completion["best_macro_source_dev_eer"]
            or completion.get("selected_checkpoint_reloaded_and_source_verified") is not True
            or completion.get("target_access") is not False):
        raise ValueError("Selected epoch or full source-reload completion is invalid")
    check_guards(config)


def verified_sources(run, config, completion, checkpoint, data_root):
    """Reuse completed full reload exports, checking bytes, canonical rows and calibration."""
    sources, records = {}, {}
    hashes = completion["source_scores_sha256"]
    if set(hashes) != {f"source_dev_{d}.csv" for d in config["sources"]}:
        raise ValueError("Source export set differs from the fold")
    for domain in config["sources"]:
        path = run / f"source_dev_{domain}.csv"
        if checkpoint_digest(path) != hashes[path.name]:
            raise ValueError(f"Completed source CSV changed: {domain}")
        rows = read_dataset(domain, "dev", data_root)
        identity = source_identity({domain: {"dev": rows}}, data_root)[domain]["dev"]
        if identity != config["source_identity"][domain]["dev"]:
            raise ValueError(f"Canonical source identity changed: {domain}")
        read_score_csv(path, domain, expected_labels=canonical_score_labels(domain, "dev", data_root, records=rows))
        # Preserve trained CSV order for exact float64 mean/std and threshold reproduction.
        labels, scores = read_score_csv(path, domain)
        if (not np.array_equal(labels, [y for _, y in rows])
                or not np.array_equal(scores, checkpoint["source_dev_scores"][domain].numpy())):
            raise ValueError(f"Source CSV differs from selected checkpoint scores: {domain}")
        sources[domain], records[domain] = (labels, scores), rows
    calibration_path = run / "source_calibration.json"
    if checkpoint_digest(calibration_path) != completion["source_calibration_sha256"]:
        raise ValueError("Completed calibration changed")
    calibration = calibrate_source_scores(sources)
    if (calibration != json.loads(calibration_path.read_text())
            or calibration["source_dev"]["macro_eer"] != completion["best_macro_source_dev_eer"]
            or calibration["source_dev"] != {"eer_by_domain": {d: v["eer"] for d, v in checkpoint["source_dev"]["by_domain"].items()},
                                             "macro_eer": checkpoint["macro_dev_eer"]}):
        raise ValueError("Source calibration does not reproduce exactly")
    return sources, records


@torch.inference_mode()
def source_parity(model, records, source_dev, device):
    """Fresh selected-model batch-one scores must exactly match preserved source scores."""
    pairs = {}
    for domain, rows in records.items():
        indices = sorted({0, len(rows) // 2, len(rows) - 1})
        for index in indices:
            waveform, label = AASISTDataset([rows[index]], training=False)[0]
            score = model.spoof_score(model(waveform.unsqueeze(0).to(device))).float().cpu().item()
            if label != source_dev[domain][0][index] or score != source_dev[domain][1][index]:
                raise ValueError(f"Fresh source score differs: {domain}/{index}")
        pairs[domain] = indices
    return pairs


def evaluation_guards():
    files = (*RUNTIME_FILES, "scripts/sam_aasist/evaluate.py", "scripts/aasist/evaluate.py")
    return {p: checkpoint_digest(ROOT / p) for p in files}


def evaluate_run(args, run, device):
    # Never silently recover or replace an earlier evaluation attempt.
    for name in ("score_manifest.json", "source_validation.json", "target_scores.csv", "target_scores.csv.tmp",
                 "metrics.json", "metrics.json.tmp", "evaluation_complete.json"):
        if (run / name).exists():
            raise FileExistsError(f"Preserving previous evaluation artifact: {run / name}")
    guards = evaluation_guards()
    completion, digest = verify_completed_run(run, args.fold, args.seed)
    if completion.get("best_checkpoint_sha256") != digest:
        raise ValueError("Selected checkpoint SHA differs from completion")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    config = json.loads((run / "config.json").read_text())
    verify_checkpoint(checkpoint, completion, config, args.fold, args.seed)
    sources, records = verified_sources(run, config, completion, checkpoint, args.source_data_root)
    model = SAMAASIST().to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parity = source_parity(model, records, sources, device)
    if guards != evaluation_guards():
        raise RuntimeError("Evaluation runtime changed; target remains unopened")
    fold = FOLDS[args.fold]
    manifest = {
        "model": FAMILY, "fold": args.fold, "seed": args.seed, "checkpoint_sha256": digest,
        "training_completion_sha256": checkpoint_digest(run / "training_complete.json"),
        "configuration_sha256": checkpoint_digest(run / "config.json"),
        "splits": {**{f"source_dev_{d}.csv": {"dataset": d, "split": "dev"} for d in config["sources"]},
                   "target_scores.csv": {"dataset": fold["target"], "split": fold["target_split"]}},
        "score_sha256": dict(completion["source_scores_sha256"]),
    }
    validation = {
        "status": "passed", "checkpoint_sha256": digest, "selected_epoch": completion["best_epoch"],
        "full_source_reload_exports_reused": True, "canonical_source_membership_verified": True,
        "source_calibration_sha256": completion["source_calibration_sha256"],
        "calibration_reproduced_exactly": True, "fresh_source_bit_exact_indices": parity,
        "source_sha256": guards, "target_access": False,
    }
    write_json(run / "source_validation.json", validation)
    save_score_manifest(run, manifest)
    print("SAM source hashes, full-source calibration and fresh batch-one parity passed; opening target", flush=True)
    labels, scores = score_split(model, fold["target"], fold["target_split"], args.target_data_root,
                                run / "target_scores.csv", device, num_workers=args.num_workers,
                                prefetch_factor=args.prefetch_factor, manifest=manifest)
    report = evaluate_transfer(sources, labels, scores, bootstrap_resamples=args.bootstrap_resamples)
    if guards != evaluation_guards():
        raise RuntimeError("Evaluation runtime changed; no completion marker")
    report.update({
        "model": FAMILY, "fold": args.fold, "seed": args.seed, "sources": config["sources"],
        "target_domain": fold["target"], "target_split": fold["target_split"],
        "checkpoint_sha256": digest, "checkpoint_epoch": completion["best_epoch"],
        "score_sha256": dict(manifest["score_sha256"]), "source_sha256": guards,
        "source_validation_sha256": checkpoint_digest(run / "source_validation.json"),
        "source_data_root": str(args.source_data_root), "target_data_root": str(args.target_data_root),
        "num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor, "cpu_threads": args.cpu_threads,
        "eval_batch_size": 1, "precision": "float32", "input_policy": "16kHz_first_64600_repeat_if_short",
        "cuda_matmul_tf32": False, "cudnn_tf32": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "pin_memory": True, "persistent_workers": args.num_workers > 0,
        "comparison_wide_freeze_confirmed_by_operator": False, "completed_run_source_validation_required": True,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
    })
    write_json(run / "metrics.json", report)
    write_json(run / "evaluation_complete.json", {
        "model": FAMILY, "fold": args.fold, "seed": args.seed, "checkpoint_sha256": digest,
        "training_completion_sha256": manifest["training_completion_sha256"],
        "metrics_sha256": checkpoint_digest(run / "metrics.json"),
        "score_manifest_sha256": checkpoint_digest(run / "score_manifest.json"),
        "source_validation_sha256": report["source_validation_sha256"],
        "target_scores_sha256": manifest["score_sha256"]["target_scores.csv"],
        "target_recordings": len(labels),
    })
    print(f"COMPLETE {FAMILY}/{args.fold}: EER={report['target']['eer']*100:.4f}%; AUROC={report['target']['auroc']:.6f}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--source-data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--target-data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args(argv)
    if min(args.seed, args.num_workers, args.bootstrap_resamples) < 0 or min(args.prefetch_factor, args.cpu_threads) < 1:
        parser.error("Seed/workers/resamples must be nonnegative; prefetch and threads positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no precision or input fallback")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = Path(os.environ.get("PROJECT_ROOT", ROOT))
    run = root / "outputs" / FAMILY / args.fold / str(args.seed)
    with (run.parent / f".{args.seed}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate_run(args, run, torch.device("cuda"))


if __name__ == "__main__":
    main()
