"""Step 3 only: fresh source-only probes, three seeds and inner source splits."""

import argparse
import itertools
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from audio_deepfake_detection.lacf.probes import (
    BATCH_SIZE, HEADS, INPUTS, OUTER_FOLDS, SEEDS, FeatureBatches, Probe,
    inner_domains, load_bank, score_probe,
)
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.metrics import read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, checkpoint_digest, save_score_manifest, verify_completed_run,
    verify_score_digest, verify_source_macro,
)


ROOT = Path(__file__).resolve().parents[2]
MAX_EPOCHS, PATIENCE = 30, 5


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def verify_probe_completion(run, fold, seed, *, expected_stage=3):
    completion, digest = verify_completed_run(run, fold, seed)
    config = json.loads((run / "config.json").read_text())
    fit_domains = inner_domains(fold, config["inner_held_domain"])
    if (expected_stage not in (3, 4) or config["stage"] != expected_stage
            or completion["stage"] != expected_stage or config["fold"] != fold or config["seed"] != seed or seed not in SEEDS
            or config["fit_domains"] != list(fit_domains)
            or completion["best_checkpoint_sha256"] != digest
            or completion.get("selected_checkpoint_reloaded_and_source_verified") is not True
            or completion["config_sha256"] != checkpoint_digest(run / "config.json")
            or completion["metrics_sha256"] != checkpoint_digest(run / "metrics.json")):
        raise ValueError("Probe split/checkpoint/reload verification failed")
    verify_source_macro(completion["reloaded_inner_dev_eer"], completion["best_macro_source_dev_eer"])
    manifest = json.loads((run / "score_manifest.json").read_text())
    if (manifest["checkpoint_sha256"] != digest
            or manifest["training_completion_sha256"] != checkpoint_digest(run / "training_complete.json")
            or manifest["validation_domain"] != config["inner_held_domain"]):
        raise ValueError("Probe score provenance mismatch")
    verify_score_digest(run / "source_dev_scores.csv", manifest)
    return completion


def fit_probe(bank, run, *, fold, held, inputs, head, seed, provenance, device, logger,
              model_factory=Probe, stage=3, feature_inputs=None, last_layer_only=False):
    fit_domains = inner_domains(fold, held)
    if seed not in SEEDS or stage not in (3, 4):
        raise ValueError("Unapproved diagnostic seed")
    run.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(run / "training.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    try:
        seed_all(seed)
        model = model_factory(inputs, head).to(device)
        optimizer = make_optimizer(model)
        feature_inputs = feature_inputs or inputs
        train = FeatureBatches([bank[d, "train"] for d in fit_domains], feature_inputs, device,
                               training=True, seed=seed, last_layer_only=last_layer_only)
        dev = FeatureBatches([bank[held, "dev"]], feature_inputs, device,
                             training=False, seed=seed, last_layer_only=last_layer_only)
        model_key = "probe" if stage == 3 else "ablation"
        config = {"stage": stage, "fold": fold, "inner_held_domain": held, "fit_domains": list(fit_domains),
                  "excluded_outer_domain": FOLDS[fold]["target"], "seed": seed, "seed_plan": list(SEEDS),
                  model_key: model.configuration(), "batch_size": BATCH_SIZE, "accumulation": 1,
                  "feature_inputs": feature_inputs, "last_layer_only": last_layer_only,
                  "optimizer": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "scheduler": "constant",
                  "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "precision": "BF16 mixed; sensitive FP32",
                  "sampling": "shared inverse domain/class bucket mass; replacement; utterance count per epoch",
                  "training_views": "uniform logical choice among centre and two fixed random views per sampled utterance",
                  "validation_views": "centre only", "selection": "strictly lowest inner-held source-dev EER; ties retain earlier",
                  "train_records": train.count, "dev_records": dev.count,
                  "feature_loading": "validated CPU-resident arrays; vectorized gathers; no per-example disk/audio reads",
                  "loader_workers": 0, "model_training_resume": False, "foundation_encoders_loaded": False,
                  "provenance": provenance}
        save_json(run / "config.json", config)
        logger.info("START %s/%s %s %s seed=%d | fit=%s dev=%s | train=%d dev=%d parameters=%d",
                    fold, held, inputs, head, seed, fit_domains, held, train.count, dev.count,
                    config[model_key]["trainable_parameters"])
        best, stale, best_epoch, best_scores = float("inf"), 0, None, None
        started = time.monotonic()
        with (run / "history.jsonl").open("x") as history:
            for epoch in range(1, MAX_EPOCHS + 1):
                epoch_started = time.monotonic()
                losses, updates = train_epoch(model, train, optimizer, device, accumulation_steps=1,
                                              logger=logger, phase=f"epoch {epoch} train")
                scores, metrics = score_probe(model, dev)
                improved = metrics["eer"] < best
                if improved:
                    best, best_epoch, stale = metrics["eer"], epoch, 0
                    best_scores = scores.clone()
                    temporary = run / "best.pt.tmp"
                    if temporary.exists():
                        raise FileExistsError("Interrupted checkpoint preserved")
                    torch.save({"config": config, "epoch": epoch, "inner_dev": metrics,
                                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, temporary)
                    temporary.replace(run / "best.pt")
                else:
                    stale += 1
                row = {"epoch": epoch, "train_loss": losses["total"], "updates": updates,
                       "inner_dev": metrics, "selected": improved, "seconds": time.monotonic() - epoch_started}
                history.write(json.dumps(row) + "\n")
                history.flush()
                logger.info("epoch %d DONE | loss %.5f | %s source EER %.4f%% AUROC %.6f | best=%d stale=%d/%d | %.1fs",
                            epoch, losses["total"], held, 100 * metrics["eer"], metrics["auroc"],
                            best_epoch, stale, PATIENCE, row["seconds"])
                if stale >= PATIENCE:
                    break
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["epoch"] != best_epoch:
            raise ValueError("Selected probe checkpoint identity changed")
        reloaded = model_factory(inputs, head).to(device)
        reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
        reloaded_scores, metrics = score_probe(reloaded, dev)
        if not torch.equal(reloaded_scores, best_scores) or metrics != checkpoint["inner_dev"]:
            raise ValueError("Fresh probe checkpoint reload did not reproduce selected source scores/metrics")
        selected_hash = checkpoint_digest(run / "best.pt")
        dataset = bank[held, "dev"]
        write_score_csv(run / "source_dev_scores.csv", zip(dataset.ids, itertools.repeat(held),
                                                          dataset.labels.tolist(), reloaded_scores.tolist()))
        # Verify exported canonical IDs/labels and bytes before queue advancement.
        _, exported = read_score_csv(run / "source_dev_scores.csv", held,
                                     expected_labels=dict(zip(dataset.ids, dataset.labels.tolist())))
        order = sorted(range(len(dataset.ids)), key=dataset.ids.__getitem__)
        if not np.array_equal(exported, reloaded_scores.numpy()[order]):
            raise ValueError("Saved source scores differ from the reloaded probe")
        weights = getattr(reloaded, "layer_weights", None)
        alpha = None if weights is None else weights.detach().float().softmax(0).cpu().tolist()
        result = {"fold": fold, "inner_held_domain": held, "inputs": inputs, "head": head, "seed": seed,
                  "eer": metrics["eer"], "auroc": metrics["auroc"], "best_epoch": best_epoch, "final_epoch": epoch,
                  "seconds": time.monotonic() - started, "layer_alpha": alpha,
                  "trainable_parameters": config[model_key]["trainable_parameters"], "scope": "inner source development only"}
        save_json(run / "metrics.json", result)
        completion = {"stage": stage, "fold": fold, "seed": seed, "best_epoch": best_epoch, "final_epoch": epoch,
                      "best_macro_source_dev_eer": best, "macro_scope": "single inner-held source domain; outer macro is across three rotations",
                      "reloaded_inner_dev_eer": metrics["eer"], "best_checkpoint_sha256": selected_hash,
                      "selected_checkpoint_reloaded_and_source_verified": True,
                      "config_sha256": checkpoint_digest(run / "config.json"),
                      "metrics_sha256": checkpoint_digest(run / "metrics.json"),
                      "completed_at": datetime.now(timezone.utc).isoformat()}
        save_json(run / "training_complete.json", completion)
        save_score_manifest(run, {"stage": stage, "validation_domain": held, "checkpoint_sha256": selected_hash,
                                  "training_completion_sha256": checkpoint_digest(run / "training_complete.json"),
                                  "score_sha256": {"source_dev_scores.csv": checkpoint_digest(run / "source_dev_scores.csv")}})
        verify_probe_completion(run, fold, seed, expected_stage=stage)
        logger.info("COMPLETE %s/%s %s %s seed=%d | selected epoch %d, source EER %.4f%%; fresh reload/export verified",
                    fold, held, inputs, head, seed, best_epoch, 100 * metrics["eer"])
        return result
    finally:
        logger.removeHandler(handler)
        handler.close()


def aggregate(results, *, input_names=INPUTS, head_names=HEADS):
    seed_macros = []
    for fold, inputs, head, seed in itertools.product(OUTER_FOLDS, input_names, head_names, SEEDS):
        rows = [r for r in results if (r["fold"], r["inputs"], r["head"], r["seed"]) == (fold, inputs, head, seed)]
        if len(rows) != 3:
            continue
        if {r["inner_held_domain"] for r in rows} != set(FOLDS[fold]["sources"]):
            raise ValueError("Invalid inner rotations in summary")
        seed_macros.append({"fold": fold, "inputs": inputs, "head": head, "seed": seed,
                            **{metric: sum(r[metric] for r in rows) / 3 for metric in ("eer", "auroc")}})
    summaries = []
    for fold, inputs, head in itertools.product(OUTER_FOLDS, input_names, head_names):
        rows = [r for r in seed_macros if (r["fold"], r["inputs"], r["head"]) == (fold, inputs, head)]
        if len(rows) == len(SEEDS):
            summaries.append({"fold": fold, "inputs": inputs, "head": head,
                              **{metric + suffix: float(fn([r[metric] for r in rows])) for metric in ("eer", "auroc")
                                 for suffix, fn in (("_mean", np.mean), ("_std", lambda v: np.std(v, ddof=1)))}})
    return {"completed_fits": len(results), "seed_macros": seed_macros, "three_seed_summary": summaries,
            "scope": "equal inner-domain macro per seed, then three-seed mean/sample SD; no target metrics or bootstrap"}


def timing_check(bank, output, device, logger):
    measurements = []
    timing_directory = output / "timing"
    timing_directory.mkdir(exist_ok=False)
    datasets = [bank[d, "train"] for d in inner_domains("f2", "asv2019")]
    for inputs, head in itertools.product(INPUTS, HEADS):
        seed_all(1234)
        model = Probe(inputs, head).to(device)
        optimizer = make_optimizer(model)
        warmup = FeatureBatches(datasets, inputs, device, training=True, seed=1234, max_batches=8)
        train_epoch(model, warmup, optimizer, device, accumulation_steps=1)
        loader = FeatureBatches(datasets, inputs, device, training=True, seed=1234, max_batches=64)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        train_epoch(model, loader, optimizer, device, accumulation_steps=1)
        elapsed = time.monotonic() - started
        dev = FeatureBatches([bank["asv2019", "dev"]], inputs, device, training=False, seed=1234)
        started = time.monotonic()
        scores, metrics = score_probe(model, dev)
        dev_seconds = time.monotonic() - started
        path = timing_directory / f"{inputs}_{head}.pt"
        torch.save({"scope": "timing only; never initialize a final probe from this state",
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)
        reloaded = Probe(inputs, head).to(device)
        reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
        reloaded_scores, reloaded_metrics = score_probe(reloaded, dev)
        if not torch.equal(scores, reloaded_scores) or metrics != reloaded_metrics:
            raise ValueError("Real source timing probe checkpoint reload failed; do not start final fits")
        row = {"inputs": inputs, "head": head, "train_seconds_per_batch": elapsed / len(loader),
               "dev_seconds_per_record": dev_seconds / dev.count, "timed_batches": len(loader),
               "trainable_parameters": sum(p.numel() for p in model.parameters()),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "source_dev_reload_bit_exact": True, "timing_checkpoint_sha256": checkpoint_digest(path)}
        measurements.append(row)
        logger.info("TIMING %s %s | %.2f ms/train batch | %.2f s/%d dev records | params=%d",
                    inputs, head, row["train_seconds_per_batch"] * 1000, dev_seconds, dev.count, row["trainable_parameters"])
        del model, optimizer, reloaded
    upper_seconds = 0
    for fold, inputs, head, seed in itertools.product(OUTER_FOLDS, INPUTS, HEADS, SEEDS):
        speed = next(row for row in measurements if row["inputs"] == inputs and row["head"] == head)
        for held in FOLDS[fold]["sources"]:
            count = sum(len(bank[d, "train"].labels) for d in inner_domains(fold, held))
            dev_count = len(bank[held, "dev"].labels)
            upper_seconds += MAX_EPOCHS * (math.ceil(count / BATCH_SIZE) * speed["train_seconds_per_batch"]
                                          + dev_count * speed["dev_seconds_per_record"])
    report = {"stage": 3, "source_only": True, "measurements": measurements,
              "recipe_candidates": 0, "scope": "8 warmup+64 timing updates per fixed probe on F2 fit sources; timing models discarded",
              "maximum_epoch_estimate_hours": upper_seconds / 3600,
              "estimate_limitations": "short timing extrapolation; early stopping reduces work; excludes full export/checkpoint/I/O overhead"}
    save_json(output / "timing.json", report)
    logger.info("TIMING ESTIMATE: %.1f hours if every one of 216 fits uses all 30 epochs; early stopping may reduce it",
                upper_seconds / 3600)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs/lacf_diagnostic_features_v1")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required; no precision fallback")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("lacf.step3")
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    code_paths = [Path(__file__).resolve(), ROOT / "src/audio_deepfake_detection/lacf/probes.py",
                  ROOT / "src/audio_deepfake_detection/lacf/model.py", ROOT / "src/audio_deepfake_detection/lacf/precision.py",
                  ROOT / "src/audio_deepfake_detection/lacf/training.py", ROOT / "src/audio_deepfake_detection/lacf/data.py",
                  ROOT / "src/audio_deepfake_detection/metrics.py", ROOT / "src/audio_deepfake_detection/protocol.py",
                  ROOT / "src/audio_deepfake_detection/data.py"]
    provenance = {"cache_root": str(args.cache_root.resolve()),
                  "cache_completion_sha256": checkpoint_digest(args.cache_root / "complete.json"),
                  "code_sha256": {str(p.relative_to(ROOT)): checkpoint_digest(p) for p in code_paths},
                  "torch": str(torch.__version__), "transformers": transformers.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
    save_json(args.output / "queue_config.json", {"stage": 3, "folds": list(OUTER_FOLDS), "seeds": list(SEEDS),
              "inputs": list(INPUTS), "heads": list(HEADS), "expected_fits": 216,
              "provenance": provenance, "stop_after_stage": 3})
    logger.info("Step 3: verifying full source feature payloads once and loading CPU-resident arrays; no encoder or target access")
    bank = load_bank(args.cache_root, args.data_root, logger=logger)
    save_json(args.output / "cache_validation.json", {"cache_completion_sha256": provenance["cache_completion_sha256"],
              "all_payload_sha256_checked": True, "all_canonical_ids_labels_checked": True,
              "source_splits": {f"{d}/{s}": {"records": len(v.labels), "unique_rows": len(v.wavlm)} for (d, s), v in bank.items()}})
    timing_check(bank, args.output, "cuda", logger)
    results = []
    for inputs, head, fold, seed in itertools.product(INPUTS, HEADS, OUTER_FOLDS, SEEDS):
        for held in FOLDS[fold]["sources"]:
            for name, expected in provenance["code_sha256"].items():
                if checkpoint_digest(ROOT / name) != expected:
                    raise ValueError("Probe implementation changed after launch; queue stops")
            run = args.output / f"{inputs}_{head}" / f"inner_{held}" / fold / str(seed)
            result = fit_probe(bank, run, fold=fold, held=held, inputs=inputs, head=head, seed=seed,
                               provenance=provenance, device="cuda", logger=logger)
            results.append(result)
            summary = {**aggregate(results), "expected_fits": 216, "complete": False}
            temporary = args.output / "summary.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(args.output / "summary.json")
            logger.info("QUEUE: %d/216 fits complete and reload-verified", len(results))
    summary = {**aggregate(results), "expected_fits": 216, "complete": True}
    if len(results) != 216 or len(summary["three_seed_summary"]) != 24:
        raise ValueError("Missing trained configurations or seed/inner rotations")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_json(args.output / "complete.json", {"stage": 3, "verified_fits": len(results),
              "summary_sha256": checkpoint_digest(args.output / "summary.json"),
              "completed_at": datetime.now(timezone.utc).isoformat(), "stop_before_stage": 4})
    logger.info("Step 3 complete: 216 verified fits; stop before Step 4 and share source-only results")


if __name__ == "__main__":
    main()
