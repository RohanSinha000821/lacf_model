"""Prepare source scores by default; native P1 target scoring is explicit."""

import argparse
import json
from pathlib import Path

import torch
from audio_deepfake_detection.lacf import evaluation as evaluation_code

from audio_deepfake_detection.lacf.data import LACFDataset, make_loader
from audio_deepfake_detection.lacf.evaluation import (
    cached_rows, direct_rows, load_completed_model, load_development_cache,
)
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest,
    read_dataset, save_score_manifest, verify_score_digest, verify_source_macro,
)
from audio_deepfake_detection.score_recovery import prepare_score_recovery, write_recoverable_scores
from scripts.evaluate_scores import evaluate_run
from scripts.lacf.train import development_cache_identity


def scorer_digests():
    return {"exporter": checkpoint_digest(Path(__file__)),
            "implementation": checkpoint_digest(Path(evaluation_code.__file__))}


def export_scores(output, records, domain, data_root, manifest, rows, *, inference):
    expected = canonical_score_labels(domain, manifest["splits"][output.name]["split"], data_root, records=records)
    if output.name in manifest["score_sha256"]:
        verify_score_digest(output, manifest)
    else:
        state = prepare_score_recovery(output, list(expected.items()), domain,
            {"checkpoint_sha256": manifest["checkpoint_sha256"], "inference": inference,
             "exporter_sha256": checkpoint_digest(Path(__file__))})
        if not output.exists():
            write_recoverable_scores(output, rows(state["rows"]), list(expected.items()), domain, state)
        manifest["score_sha256"][output.name] = checkpoint_digest(output)
        save_score_manifest(output.parent, manifest)
    return read_score_csv(output, domain, expected_labels=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--evaluate-target", action="store_true", help="Requires separate owner authorization")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("Workers must be nonnegative")
    if args.evaluate_target and (args.run_dir / "metrics.json").exists():
        raise FileExistsError("Metrics already exist; refusing to overwrite")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 is required; no precision fallback")
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    model, collator, config, completion = load_completed_model(args.run_dir, args.fold, device)
    digest = completion["best_checkpoint_sha256"]
    manifest = load_score_manifest(args.run_dir, args.fold, 1234, digest, create=True)
    inference = {"input_policy": config["input_policy"], "precision": config["precision"],
                 "sensitive_dtype": "float32", "batch_size": 1, "score": "scalar_spoof_logit",
                 "wavlm_revision": config["wavlm_revision"], "clap_revision": config["clap_revision"],
                 "cudnn_deterministic": True, "cudnn_benchmark": False,
                 "torch": str(torch.__version__), "cuda": torch.version.cuda}
    for key, value in {"model_family": config["run_name"], "inference": inference}.items():
        if key in manifest and manifest[key] != value:
            raise ValueError("LACF score provenance changed")
        manifest[key] = value
    save_score_manifest(args.run_dir, manifest)
    source_dev = {}
    for domain in FOLDS[args.fold]["sources"]:
        records = read_dataset(domain, "dev", args.data_root)
        cache = load_development_cache(config["development_feature_caches"][domain], records,
            {**development_cache_identity(config), "domain": domain}, device)
        source_dev[domain] = export_scores(args.run_dir / f"source_dev_{domain}.csv", records, domain,
            args.data_root, manifest,
            lambda start: cached_rows(model, cache, records, domain, args.data_root, device, start=start),
            inference={**inference, "source_feature_cache_sha256": cache.sha256})
        verify_source_macro(calibrate_source_scores({domain: source_dev[domain]})["source_dev"]["macro_eer"],
                            completion["reloaded_dev_eers"][domain])
        del cache
    calibration = calibrate_source_scores(source_dev)
    verify_source_macro(calibration["source_dev"]["macro_eer"], completion["best_macro_source_dev_eer"])
    print(f"Source-only macro EER reproduced: {100 * calibration['source_dev']['macro_eer']:.5f}%", flush=True)
    if not args.evaluate_target:
        print("Source preparation complete. Held-out target remains unopened.", flush=True)
        return
    # Separate target invocation must retain source-only checkpoint/calibration checks.
    validation_path = args.run_dir / "source_validation.json"
    validation = json.loads(validation_path.read_text())
    if (validation.get("status") != "passed" or validation.get("checkpoint_sha256") != digest
            or validation.get("inference") != inference
            or validation.get("source_sha256") != config["source_sha256"]
            or validation.get("source_cache_sha256") != {
                domain: reference["sha256"] for domain, reference in config["development_feature_caches"].items()}
            or not validation.get("direct_cache_score_bit_exact")
            or validation.get("scorer_sha256") != scorer_digests()
            or validation.get("score_sha256") != {
                f"source_dev_{domain}.csv": manifest["score_sha256"][f"source_dev_{domain}.csv"]
                for domain in FOLDS[args.fold]["sources"]}):
        raise ValueError("Fresh-process source/cache validation is required before target access")
    manifest.update(source_validation_sha256=checkpoint_digest(validation_path), scorer_sha256=scorer_digests())
    save_score_manifest(args.run_dir, manifest)
    fold = FOLDS[args.fold]
    records = read_dataset(fold["target"], fold["target_split"], args.data_root)
    def target_rows(start):
        remaining = records[start:]
        if not remaining:
            return iter(())
        loader = make_loader([LACFDataset(remaining, training=False, segment_seconds=config["segment_seconds"])],
            collator, training=False, seed=1234, batch_size=1, num_workers=args.num_workers, prefetch_factor=2)
        return direct_rows(model, loader, remaining, fold["target"], args.data_root, device, start=start)
    export_scores(args.run_dir / "target_scores.csv", records, fold["target"], args.data_root,
                  manifest, target_rows, inference=inference)
    report = evaluate_run(args.run_dir, args.fold, seed=1234, source_data_root=args.data_root,
                          target_data_root=args.data_root, bootstrap_resamples=1000)
    print(f"Target EER: {100 * report['target']['eer']:.5f}%", flush=True)


if __name__ == "__main__":
    main()
