"""Owner-approved TF0–TF3 full-source F1 only; source dev, no resume/targets."""

import argparse
import gc
import itertools
import json
import logging
import math
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from audio_deepfake_detection.lacf.probes import BATCH_SIZE, SEEDS, FeatureBatches, load_split, score_probe
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, checkpoint_digest, save_score_manifest, verify_completed_run, verify_score_digest, verify_source_macro,
)
from audio_deepfake_detection.token_fusion.model import CONDITIONS, PARAMETERS, TokenFusion
from scripts.lacf.train_ablations import fixed_prototypes
from scripts.lacf.train_probes import save_json, seed_all


ROOT = Path(__file__).resolve().parents[2]
FOLD, SOURCES = "f1", tuple(FOLDS["f1"]["sources"])
MAX_EPOCHS, PATIENCE = 30, 5
SCOPE = "full F1 source train/dev; selected in-domain development, not inner-held-domain validation"
STAGE = "token_fusion_full_source"


def replace_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def assert_bank(bank):
    if set(bank) != {(d, s) for d in SOURCES for s in ("train", "dev")}:
        raise ValueError("Only complete F1 source train/dev partitions may enter this experiment")


def load_f1_bank(cache, data_root, logger):
    marker = json.loads((cache / "complete.json").read_text())
    if (marker.get("step") != 2
            or checkpoint_digest(cache / "config.json") != marker["config_sha256"]
            or checkpoint_digest(cache / "execution_accelerated.json") != marker["accelerated_execution_sha256"]):
        raise ValueError("Frozen source-cache identity changed")
    bank = {}
    for domain, split in itertools.product(SOURCES, ("train", "dev")):
        logger.info("CACHE checking/loading %s/%s; other domains remain unopened", domain, split)
        value = load_split(cache, domain, split, marker["splits"][f"{domain}/{split}"], data_root, logger=logger)
        if not torch.isfinite(value.wavlm).all() or not torch.isfinite(value.clap).all():
            raise ValueError("Nonfinite frozen feature cache")
        bank[domain, split] = value
    assert_bank(bank)
    return bank


def check_guards(provenance):
    for name, expected in provenance["code_sha256"].items():
        if checkpoint_digest(ROOT / name) != expected:
            raise ValueError(f"Execution source changed; queue stops: {name}")
    for name, expected in provenance.get("preserved_job_guards", {}).items():
        if checkpoint_digest(ROOT / name) != expected:
            raise ValueError(f"Preserved SSL source guard changed: {name}")


def score_sources(model, bank, device, logger=None, *, max_batches=None):
    scores, metrics = {}, {}
    for domain in SOURCES:
        if logger:
            logger.info("DEV %s start | %d centred cached views", domain, len(bank[domain, "dev"].labels))
        loader = FeatureBatches([bank[domain, "dev"]], "concat", device, training=False,
                               seed=1234, max_batches=max_batches)
        scores[domain], metrics[domain] = score_probe(model, loader)
        if logger:
            logger.info("DEV %s done | EER %.4f%% AUROC %.6f", domain,
                        metrics[domain]["eer"] * 100, metrics[domain]["auroc"])
    return scores, {"by_domain": metrics,
                    "macro_eer": statistics.mean(v["eer"] for v in metrics.values()),
                    "macro_auroc": statistics.mean(v["auroc"] for v in metrics.values())}


def verify_fit(run, bank):
    config = json.loads((run / "config.json").read_text())
    seed = config["seed"]
    completion, digest = verify_completed_run(run, FOLD, seed)
    condition = config["model"]["condition"]
    if (condition not in CONDITIONS or seed not in SEEDS or config["stage"] != STAGE
            or config["sources"] != list(SOURCES) or config["scope"] != SCOPE
            or config["model"]["trainable_parameters"] != PARAMETERS[condition]
            or completion.get("stage") != STAGE
            or completion.get("selected_checkpoint_reloaded_and_source_verified") is not True
            or completion["best_checkpoint_sha256"] != digest
            or completion["config_sha256"] != checkpoint_digest(run / "config.json")
            or completion["metrics_sha256"] != checkpoint_digest(run / "metrics.json")
            or completion["calibration_sha256"] != checkpoint_digest(run / "source_calibration.json")):
        raise ValueError("Full-source experiment completion/recipe binding failed")
    manifest = json.loads((run / "score_manifest.json").read_text())
    if (manifest["checkpoint_sha256"] != digest or manifest["sources"] != list(SOURCES)
            or manifest["training_completion_sha256"] != checkpoint_digest(run / "training_complete.json")
            or set(manifest["score_sha256"]) != {f"source_dev_{d}.csv" for d in SOURCES}):
        raise ValueError("Source-only score provenance failed")
    source = {}
    for domain in SOURCES:
        path = run / f"source_dev_{domain}.csv"
        verify_score_digest(path, manifest)
        dataset = bank[domain, "dev"]
        source[domain] = read_score_csv(path, domain, expected_labels=dict(zip(dataset.ids, dataset.labels.tolist())))
    calibration = calibrate_source_scores(source)
    if calibration != json.loads((run / "source_calibration.json").read_text()):
        raise ValueError("Source calibration did not reproduce from canonical score exports")
    verify_source_macro(calibration["source_dev"]["macro_eer"], completion["best_macro_source_dev_eer"])
    return completion


def fit_run(bank, run, condition, seed, prototypes, provenance, device, logger,
            *, factory=TokenFusion, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    assert_bank(bank)
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError("Unexpected experiment condition/seed")
    check_guards(provenance)
    run.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(run / "training.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    started = time.monotonic()
    try:
        seed_all(seed)
        model = factory(condition, prototypes).to(device)
        optimizer = make_optimizer(model)
        train = FeatureBatches([bank[d, "train"] for d in SOURCES], "concat", device, training=True, seed=seed)
        config = {"stage": STAGE, "scope": SCOPE, "fold": FOLD, "sources": list(SOURCES),
                  "excluded_target": FOLDS[FOLD]["target"], "target_access": False,
                  "seed": seed, "seed_plan": list(SEEDS), "model": model.configuration(),
                  "batch_size": BATCH_SIZE, "accumulation": 1, "optimizer": "AdamW", "lr": 3e-4,
                  "weight_decay": 1e-4, "scheduler": "constant", "max_epochs": max_epochs,
                  "patience": patience, "precision": "BF16 mixed; sensitive FP32",
                  "sampling": "equal domain/class mass; replacement; utterance count per epoch",
                  "training_views": "centre + two fixed random views; uniform logical view per draw",
                  "development_views": "centre only; cached unpadded encoder features",
                  "selection": "strict equal-weight macro source-dev EER; ties retain earlier",
                  "loader_workers": 0, "resume": False, "provenance": provenance,
                  "train_records": train.count,
                  "dev_records": {d: len(bank[d, "dev"].labels) for d in SOURCES}}
        save_json(run / "config.json", config)
        logger.info("START %s %s seed=%d | train=%d dev=%s | params=%d", FOLD, condition, seed,
                    train.count, config["dev_records"], config["model"]["trainable_parameters"])
        best, stale, best_epoch, selected_scores = float("inf"), 0, None, None
        with (run / "history.jsonl").open("x") as history:
            for epoch in range(1, max_epochs + 1):
                check_guards(provenance)
                tick = time.monotonic()
                losses, updates = train_epoch(model, train, optimizer, device, accumulation_steps=1,
                                              logger=logger, phase=f"{condition} seed{seed} epoch{epoch} train")
                train_seconds = time.monotonic() - tick
                scores, metrics = score_sources(model, bank, device, logger)
                improved = metrics["macro_eer"] < best
                if improved:
                    best, best_epoch, stale, selected_scores = metrics["macro_eer"], epoch, 0, scores
                    temporary = run / "best.pt.tmp"
                    if temporary.exists():
                        raise FileExistsError("Interrupted checkpoint preserved")
                    torch.save({"config": config, "epoch": epoch, "source_dev": metrics,
                                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, temporary)
                    temporary.replace(run / "best.pt")
                else:
                    stale += 1
                row = {"epoch": epoch, "losses": losses, "updates": updates, "source_dev": metrics,
                       "selected": improved, "train_seconds": train_seconds,
                       "dev_seconds": time.monotonic() - tick - train_seconds,
                       "seconds": time.monotonic() - tick}
                history.write(json.dumps(row, allow_nan=False) + "\n")
                history.flush()
                logger.info("EPOCH %d DONE | loss %.6f | macro source EER %.4f%% AUROC %.6f | best=%d stale=%d/%d | %.1fs",
                            epoch, losses["total"], metrics["macro_eer"] * 100,
                            metrics["macro_auroc"], best_epoch, stale, patience, row["seconds"])
                if stale >= patience:
                    break
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["epoch"] != best_epoch:
            raise ValueError("Selected checkpoint identity changed")
        del model, optimizer
        reloaded = factory(condition, prototypes).to(device)
        reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
        scores, metrics = score_sources(reloaded, bank, device, logger)
        if metrics != checkpoint["source_dev"] or any(not torch.equal(scores[d], selected_scores[d]) for d in SOURCES):
            raise ValueError("Selected checkpoint full source reload is not bit exact")
        hashes, calibration_inputs = {}, {}
        for domain in SOURCES:
            dataset = bank[domain, "dev"]
            path = run / f"source_dev_{domain}.csv"
            write_score_csv(path, zip(dataset.ids, itertools.repeat(domain), dataset.labels.tolist(), scores[domain].tolist()))
            labels, exported = read_score_csv(path, domain, expected_labels=dict(zip(dataset.ids, dataset.labels.tolist())))
            order = sorted(range(len(dataset.ids)), key=dataset.ids.__getitem__)
            if not np.array_equal(exported, scores[domain].numpy()[order]):
                raise ValueError("Canonical source score export differs from selected scores")
            calibration_inputs[domain] = labels, exported
            hashes[path.name] = checkpoint_digest(path)
        calibration = calibrate_source_scores(calibration_inputs)
        verify_source_macro(calibration["source_dev"]["macro_eer"], best)
        save_json(run / "source_calibration.json", calibration)
        result = {"condition": condition, "fold": FOLD, "seed": seed, "scope": SCOPE,
                  "source_dev": metrics, "best_epoch": best_epoch, "final_epoch": epoch,
                  "seconds": time.monotonic() - started, "trainable_parameters": config["model"]["trainable_parameters"],
                  "run": str(run.resolve())}
        save_json(run / "metrics.json", result)
        digest = checkpoint_digest(run / "best.pt")
        completion = {"stage": STAGE, "fold": FOLD, "seed": seed, "best_epoch": best_epoch, "final_epoch": epoch,
                      "best_macro_source_dev_eer": best, "scope": SCOPE, "sources": list(SOURCES),
                      "selected_checkpoint_reloaded_and_source_verified": True,
                      "best_checkpoint_sha256": digest, "config_sha256": checkpoint_digest(run / "config.json"),
                      "metrics_sha256": checkpoint_digest(run / "metrics.json"),
                      "calibration_sha256": checkpoint_digest(run / "source_calibration.json"),
                      "completed_at": datetime.now(timezone.utc).isoformat()}
        save_json(run / "training_complete.json", completion)
        save_score_manifest(run, {"stage": STAGE, "scope": SCOPE, "fold": FOLD, "seed": seed,
                                 "sources": list(SOURCES), "target_access": False,
                                 "checkpoint_sha256": digest,
                                 "training_completion_sha256": checkpoint_digest(run / "training_complete.json"),
                                 "score_sha256": hashes})
        verify_fit(run, bank)
        logger.info("COMPLETE %s seed%d | best epoch%d | macro source EER %.4f%% | exact reload/export/calibration",
                    condition, seed, best_epoch, best * 100)
        return result
    finally:
        logger.removeHandler(handler)
        handler.close()


def preflight(bank, output, prototypes, logger, provenance):
    rows, paired = [], None
    datasets = [bank[d, "train"] for d in SOURCES]
    for condition in CONDITIONS:
        check_guards(provenance)
        seed_all(1234)
        model = TokenFusion(condition, prototypes).cuda()
        optimizer = make_optimizer(model)
        first = next(iter(FeatureBatches(datasets, "concat", "cuda", training=True, seed=1234, max_batches=1)))
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            values = model(first)
            losses = model.objective(values, first["labels"])
        if values["features"].dtype != torch.float32 or losses["total"].dtype != torch.float32 or not torch.isfinite(losses["total"]):
            raise ValueError("GPU preflight precision/finite check failed")
        if condition == "TF2":
            paired = values["logit"].float().cpu(), torch.cuda.get_rng_state()
        if condition == "TF3" and any(not torch.equal(a, b) for a, b in zip(paired, (values["logit"].float().cpu(), torch.cuda.get_rng_state()))):
            raise ValueError("TF2/TF3 initial GPU predictions/dropout RNG differ")
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        del first, values, losses
        logger.info("PREFLIGHT %s | 8 warmup then 64 measured updates; full F1 source dev + fresh reload", condition)
        for batches in (8, 64):
            loader = FeatureBatches(datasets, "concat", "cuda", training=True, seed=1234, max_batches=batches)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.monotonic()
            losses, updates = train_epoch(model, loader, optimizer, "cuda", accumulation_steps=1)
            torch.cuda.synchronize()
            seconds = time.monotonic() - tick
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        tick = time.monotonic()
        scores, metrics = score_sources(model, bank, "cuda", logger)
        dev_seconds = time.monotonic() - tick
        path = output / f"preflight_{condition}.pt"
        torch.save({"scope": "timing only; never initialize final fits", "model_state_dict":
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)
        del model, optimizer
        fresh = TokenFusion(condition, prototypes).cuda()
        fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
        other, other_metrics = score_sources(fresh, bank, "cuda", logger)
        if metrics != other_metrics or any(not torch.equal(scores[d], other[d]) for d in SOURCES):
            raise ValueError("GPU preflight full source reload is not bit exact")
        rows.append({"condition": condition, "trainable_parameters": PARAMETERS[condition],
                     "timed_batches": len(loader), "updates": updates, "seconds_per_train_batch": seconds / len(loader),
                     "source_dev_seconds": dev_seconds, "losses": losses, "source_dev": metrics,
                     "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                     "peak_including_source_reload_bytes": torch.cuda.max_memory_allocated(),
                     "full_source_reload_bit_exact": True, "timing_checkpoint_sha256": checkpoint_digest(path)})
        replace_json(output / "preflight_progress.json", {"complete": False, "measurements": rows})
        logger.info("PREFLIGHT %s PASS | %.1f views/s | %.2f GiB allocated | %.1fs full source dev",
                    condition, BATCH_SIZE * len(loader) / seconds, peak_allocated / 1024**3, dev_seconds)
        del fresh, scores, other
        gc.collect()
        torch.cuda.empty_cache()
    train_batches = math.ceil(sum(len(bank[d, "train"].labels) for d in SOURCES) / BATCH_SIZE)
    upper = sum(3 * MAX_EPOCHS * (train_batches * r["seconds_per_train_batch"] + r["source_dev_seconds"]) for r in rows)
    report = {"complete": True, "scope": "F1 sources only; timing weights discarded", "measurements": rows,
              "tf2_tf3_initial_gpu_scores_and_dropout_rng_bit_exact": True,
              "max_30_epoch_estimate_hours": upper / 3600,
              "estimate_limitations": "short shared-GPU measurement; excludes I/O/export; early stopping may reduce time"}
    save_json(output / "preflight.json", report)
    logger.info("ALL PREFLIGHT CHECKS PASSED | max30-epoch 12-fit estimate %.1f hours, not a duration guarantee", upper / 3600)
    return report


def summary_for(results):
    summaries = []
    for condition in CONDITIONS:
        group = [r for r in results if r["condition"] == condition]
        if len(group) == len(SEEDS):
            if {r["seed"] for r in group} != set(SEEDS):
                raise ValueError("Duplicate/unexpected comparison seeds")
            summaries.append({"condition": condition,
                              **{metric + suffix: fn(r["source_dev"]["macro_" + metric] for r in group)
                                 for metric in ("eer", "auroc")
                                 for suffix, fn in (("_mean", statistics.mean), ("_std", statistics.stdev))}})
    return {"stage": STAGE, "scope": SCOPE, "completed_fits": len(results), "expected_fits": 12,
            "three_seed_summary": summaries, "fits": results,
            "target_access": False, "selected_global_winner": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs/lacf_diagnostic_features_v1")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-memory-budget-gib", type=float, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required; no precision fallback")
    total = torch.cuda.get_device_properties(0).total_memory
    if not 0 < args.gpu_memory_budget_gib * 1024**3 < total:
        raise ValueError("Invalid GPU allocator budget")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_budget_gib * 1024**3 / total)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("token_fusion")
    code = [Path(__file__).resolve(), ROOT / "scripts/lacf/train_ablations.py", ROOT / "scripts/lacf/train_probes.py",
            *[ROOT / "src/audio_deepfake_detection" / p for p in
              ("token_fusion/model.py", "lacf/probes.py", "lacf/model.py", "lacf/precision.py",
               "lacf/training.py", "lacf/data.py", "data.py", "protocol.py", "metrics.py")]]
    ssl_config = ROOT / "outputs/ssl_gating_20s/f2/1234/config.json"
    guards = json.loads(ssl_config.read_text())["source_sha256"] if ssl_config.exists() else {}
    prototypes, prototype_evidence = fixed_prototypes()
    torch.save(prototypes, args.output / "prototypes.pt")
    provenance = {"code_sha256": {str(p.relative_to(ROOT)): checkpoint_digest(p) for p in code},
                  "preserved_job_guards": guards, "cache_root": str(args.cache_root.resolve()),
                  "cache_completion_sha256": checkpoint_digest(args.cache_root / "complete.json"),
                  "prototype_evidence": prototype_evidence,
                  "prototype_file_sha256": checkpoint_digest(args.output / "prototypes.pt"),
                  "torch": str(torch.__version__), "transformers": transformers.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "gpu_allocator_budget_gib": args.gpu_memory_budget_gib,
                  "python": platform.python_version(), "cpu_threads": 2, "tf32": False,
                  "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    check_guards(provenance)
    configs = {}
    for condition in CONDITIONS:
        seed_all(1234)
        configs[condition] = TokenFusion(condition, prototypes).configuration()
    save_json(args.output / "queue_config.json", {"stage": STAGE, "fold": FOLD, "scope": SCOPE,
              "sources": list(SOURCES), "conditions": configs, "seeds": list(SEEDS), "expected_fits": 12,
              "owner_decision": "all three F1 source training domains, all four controls; not inner rotations",
              "provenance": provenance, "resume": False, "target_access": False})
    bank = load_f1_bank(args.cache_root, args.data_root, logger)
    save_json(args.output / "cache_validation.json", {"source_partitions": {f"{d}/{s}": {
              "records": len(v.labels), "unique_rows": len(v.wavlm)} for (d, s), v in bank.items()},
              "canonical_ids_labels_and_all_loaded_payload_hashes_checked": True,
              "finite_arrays": True, "excluded_domain_not_loaded": "speechfake"})
    preflight(bank, args.output, prototypes, logger, provenance)
    results = []
    for condition, seed in itertools.product(CONDITIONS, SEEDS):
        check_guards(provenance)
        run = args.output / condition / FOLD / str(seed)
        result = fit_run(bank, run, condition, seed, prototypes, provenance, "cuda", logger)
        results.append(result)
        replace_json(args.output / "summary.json", summary_for(results))
        logger.info("QUEUE %d/12 reload-verified complete; next fit starts fresh", len(results))
        gc.collect()
        torch.cuda.empty_cache()
    for result in results:
        verify_fit(Path(result["run"]), bank)
    summary = summary_for(results)
    if len(results) != 12 or len(summary["three_seed_summary"]) != 4:
        raise ValueError("Incomplete source-only comparison")
    summary["complete"] = True
    replace_json(args.output / "summary.json", summary)
    save_json(args.output / "complete.json", {"verified_fits": 12, "stage": STAGE,
              "summary_sha256": checkpoint_digest(args.output / "summary.json"),
              "completed_at": datetime.now(timezone.utc).isoformat(), "stop_after_queue": True,
              "target_access": False})
    logger.info("COMPLETE all12F1fits; stop without target evaluation or later folds")


if __name__ == "__main__":
    main()
