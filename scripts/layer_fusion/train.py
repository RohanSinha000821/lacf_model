"""Fresh C0–C4 frozen-cache comparison; source-only, sequential, no resume."""

import argparse
import hashlib
import itertools
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from audio_deepfake_detection.lacf.probes import (
    BATCH_SIZE, OUTER_FOLDS, SEEDS, FeatureBatches, inner_domains, load_bank, score_probe,
)
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.layer_fusion.diagnostics import source_diagnostics
from audio_deepfake_detection.layer_fusion.model import CONDITIONS, LayerFusionDetector
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from scripts.lacf.train_ablations import fixed_prototypes
from scripts.lacf.train_probes import aggregate, fit_probe, save_json, seed_all, verify_probe_completion


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FITS = len(CONDITIONS) * len(OUTER_FOLDS) * 3 * len(SEEDS)
STAGE = "layer_fusion"


def timing_check(bank, output, factory, logger):
    """Bounded GPU preflight before final fits; never reuse these trained weights."""
    rows, initial_checks, paired_expert_scores = [], [], None
    for condition in CONDITIONS:
        seed_all(1234)
        model = factory(condition, "relation").cuda()
        optimizer = make_optimizer(model)
        datasets = [bank[d, "train"] for d in inner_domains("f2", "asv2019")]
        initial_loader = FeatureBatches(datasets, "concat", "cuda", training=True, seed=1234, max_batches=1)
        batch = next(iter(initial_loader))
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            initial = model(batch)
            initial_loss = model.objective(initial, batch["labels"])["total"]
        if (not torch.isfinite(initial_loss) or initial_loss.dtype != torch.float32
                or initial["features"].dtype != torch.float32):
            raise ValueError("Source GPU initialization precision/finite check failed")
        scores = initial["logit"].detach().float().cpu()
        after_dropout = torch.cuda.get_rng_state()
        if condition == "C3":
            paired_expert_scores = (scores, cuda_rng, after_dropout)
        if condition == "C4" and (paired_expert_scores is None or any(
                not torch.equal(left, right) for left, right in zip(
                    paired_expert_scores, (scores, cuda_rng, after_dropout)))):
            raise ValueError("C3/C4 source GPU initial predictions/dropout RNG differ")
        initial_checks.append({"condition": condition,
                               "cuda_dropout_rng_sha256": hashlib.sha256(cuda_rng.numpy().tobytes()).hexdigest(),
                               "initial_scores_sha256": hashlib.sha256(scores.numpy().tobytes()).hexdigest(),
                               "loss_fp32_features_fp32_finite": True})
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        del batch, initial, initial_loss, initial_loader
        for batches in (8, 64):
            loader = FeatureBatches(datasets, "concat", "cuda", training=True, seed=1234, max_batches=batches)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            losses, _ = train_epoch(model, loader, optimizer, "cuda", accumulation_steps=1)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
        dev = FeatureBatches([bank["asv2019", "dev"]], "concat", "cuda", training=False, seed=1234)
        started = time.monotonic()
        scores, metrics = score_probe(model, dev)
        dev_seconds = time.monotonic() - started
        path = output / f"timing_{condition}.pt"
        torch.save({"scope": "timing only; never initialize final fits", "model_state_dict":
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)
        reloaded = factory(condition, "relation").cuda()
        reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
        other, other_metrics = score_probe(reloaded, dev)
        if not torch.equal(scores, other) or metrics != other_metrics:
            raise ValueError("Source-only layer-fusion preflight reload failed; queue stopped")
        row = {"condition": condition, "timed_batches": len(loader), "seconds_per_train_batch": seconds / len(loader),
               "seconds_per_dev_record": dev_seconds / dev.count, "losses": losses,
               "trainable_parameters": model.configuration()["trainable_parameters"],
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "source_dev_reload_bit_exact": True,
               "timing_checkpoint_sha256": checkpoint_digest(path)}
        rows.append(row)
        logger.info("TIMING %s | %.1f cached views/s | %d parameters | source reload exact", condition,
                    BATCH_SIZE / row["seconds_per_train_batch"], row["trainable_parameters"])
        del model, optimizer, reloaded
    upper = 0
    for row, fold, _seed in itertools.product(rows, OUTER_FOLDS, SEEDS):
        for held in FOLDS[fold]["sources"]:
            count = sum(len(bank[d, "train"].labels) for d in inner_domains(fold, held))
            upper += 30 * (math.ceil(count / BATCH_SIZE) * row["seconds_per_train_batch"]
                           + len(bank[held, "dev"].labels) * row["seconds_per_dev_record"])
    save_json(output / "timing.json", {"stage": STAGE, "source_only": True, "measurements": rows,
              "maximum_epoch_estimate_hours": upper / 3600, "recipe_candidates": 0,
              "initialization_checks": initial_checks, "c3_c4_initial_gpu_predictions_and_dropout_rng_bit_exact": True,
              "scope": "8 warmup + 64 measured updates per fixed condition; full ASV2019 source-dev/reload",
              "limitations": "short timing extrapolation; excludes routing reports/export/I/O; early stopping reduces work"})
    logger.info("Preflight passed; max-30-epoch training/scoring estimate %.1f hours", upper / 3600)


def fit_condition(bank, run, *, condition, fold, held, seed, provenance, device, logger, factory):
    result = fit_probe(bank, run, fold=fold, held=held, inputs=condition, head="relation", seed=seed,
                       provenance=provenance, device=device, logger=logger, model_factory=factory,
                       stage=STAGE, feature_inputs="concat")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    selected = factory(condition, "relation").to(device)
    selected.load_state_dict(checkpoint["model_state_dict"], strict=True)
    diagnostics = source_diagnostics(selected, bank, fold, seed, device)
    diagnostics.update({"checkpoint_sha256": checkpoint_digest(run / "best.pt"),
                        "training_completion_sha256": checkpoint_digest(run / "training_complete.json"),
                        "inner_held_domain": held, "seed": seed})
    path = run / "source_diagnostics.json"
    save_json(path, diagnostics)
    return {**result, "source_diagnostics_sha256": checkpoint_digest(path), "run": str(run.resolve())}


def write_summary(output, summary):
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output / "summary.json")


def run_queue(bank, output, factory, provenance, device, logger):
    """Run fixed fresh fits after preflight; factored for synthetic CPU coverage."""
    results = []
    for condition, fold, seed in itertools.product(CONDITIONS, OUTER_FOLDS, SEEDS):
        for held in FOLDS[fold]["sources"]:
            for name, expected in provenance["code_sha256"].items():
                if checkpoint_digest(ROOT / name) != expected:
                    raise ValueError("Layer-fusion implementation changed after launch; queue stopped")
            run = output / condition / f"inner_{held}" / fold / str(seed)
            results.append(fit_condition(bank, run, condition=condition, fold=fold, held=held, seed=seed,
                                         provenance=provenance, device=device, logger=logger, factory=factory))
            summary = {**aggregate(results, input_names=CONDITIONS, head_names=("relation",)),
                       "stage": STAGE, "expected_fits": EXPECTED_FITS, "complete": False,
                       "fits": results}
            write_summary(output, summary)
            logger.info("QUEUE %d/%d source fits reload-verified; diagnostics saved", len(results), EXPECTED_FITS)
    if len(results) != EXPECTED_FITS or len(summary["three_seed_summary"]) != len(CONDITIONS) * len(OUTER_FOLDS):
        raise ValueError("Missing layer-fusion inner rotations/seeds")
    for result in results:
        run = Path(result["run"])
        verify_probe_completion(run, result["fold"], result["seed"], expected_stage=STAGE)
        if checkpoint_digest(run / "source_diagnostics.json") != result["source_diagnostics_sha256"]:
            raise ValueError("Source routing report changed; completion refused")
    summary["complete"] = True
    write_summary(output, summary)
    save_json(output / "complete.json", {"stage": STAGE, "verified_fits": len(results),
              "summary_sha256": checkpoint_digest(output / "summary.json"),
              "completed_at": datetime.now(timezone.utc).isoformat(), "stop_after_queue": True})
    logger.info("C0–C4 comparison complete; report source results and stop. No fine-tuning or target evaluation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs/lacf_diagnostic_features_v1")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required; no precision fallback")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("layer_fusion")
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    prototypes, evidence = fixed_prototypes()
    torch.save(prototypes, args.output / "prototypes.pt")
    paths = [Path(__file__).resolve(), ROOT / "scripts/lacf/train_probes.py", ROOT / "scripts/lacf/train_ablations.py", *[
        ROOT / "src/audio_deepfake_detection" / name for name in (
            "layer_fusion/model.py", "layer_fusion/diagnostics.py", "lacf/ablations.py", "lacf/probes.py",
            "lacf/model.py", "lacf/precision.py", "lacf/training.py", "lacf/data.py", "data.py", "protocol.py", "metrics.py")]]
    provenance = {"cache_root": str(args.cache_root.resolve()),
                  "cache_completion_sha256": checkpoint_digest(args.cache_root / "complete.json"),
                  "prototype_evidence": evidence, "prototype_file_sha256": checkpoint_digest(args.output / "prototypes.pt"),
                  "code_sha256": {str(p.relative_to(ROOT)): checkpoint_digest(p) for p in paths},
                  "torch": str(torch.__version__), "transformers": transformers.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
    def factory(condition, head):
        if head != "relation":
            raise ValueError("C0–C4 retain the full primary relation classifier")
        return LayerFusionDetector(condition, prototypes)
    configs = {name: factory(name, "relation").configuration() for name in CONDITIONS}
    save_json(args.output / "queue_config.json", {"stage": STAGE, "conditions": configs,
              "folds": list(OUTER_FOLDS), "seeds": list(SEEDS), "expected_fits": EXPECTED_FITS,
              "provenance": provenance, "model_training_resume": False, "stop_after_queue": True})
    logger.info("Verifying existing source cache once and loading all 13 layers into CPU RAM; no encoder/target access")
    bank = load_bank(args.cache_root, args.data_root, logger=logger)
    save_json(args.output / "cache_validation.json", {"all_payload_sha256_checked": True,
              "all_canonical_ids_labels_checked": True, "cache_completion_sha256": provenance["cache_completion_sha256"],
              "source_splits": {f"{d}/{s}": {"records": len(v.labels), "unique_rows": len(v.wavlm)} for (d, s), v in bank.items()}})
    timing_check(bank, args.output, factory, logger)
    run_queue(bank, args.output, factory, provenance, "cuda", logger)


if __name__ == "__main__":
    main()
