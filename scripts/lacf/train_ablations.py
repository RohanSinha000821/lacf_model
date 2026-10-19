"""Step 4 only: A4–A9 and loss controls on source-only cached features."""

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

from audio_deepfake_detection.lacf.ablations import ALIASES, VARIANTS, RelationAblation, feature_inputs
from audio_deepfake_detection.lacf.model import CONCEPTS, TEMPLATES
from audio_deepfake_detection.lacf.probes import (
    BATCH_SIZE, OUTER_FOLDS, SEEDS, FeatureBatches, inner_domains, load_bank, score_probe,
)
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from scripts.lacf.train_probes import aggregate, fit_probe, save_json, seed_all


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FITS = len(VARIANTS) * len(OUTER_FOLDS) * 3 * len(SEEDS)


def fixed_prototypes():
    """Read only identical frozen text buffers; no learned old weights initialize fits."""
    original = json.loads((ROOT / "outputs/lacf_diagnostics_step1/static.json").read_text())
    values, evidence = [], {}
    for fold in ("f1", "f2"):
        path = ROOT / "outputs/lacf10s_optimized_20261009" / fold / "1234/best.pt"
        digest = checkpoint_digest(path)
        if digest != original["checkpoint_sha256"]["lacf_" + fold]:
            raise ValueError("Preserved prototype-source checkpoint changed")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        config = state["config"]["components"]
        if (config["concepts"] != list(CONCEPTS) or config["templates"] != list(TEMPLATES)
                or config["temperature"] != 0.07):
            raise ValueError("Primary concept/template/temperature identity changed")
        values.append(state["model_state_dict"]["prototypes"].float().clone())
        evidence[fold] = {"path": str(path), "checkpoint_sha256": digest}
        del state
    if not torch.equal(*values):
        raise ValueError("Frozen F1/F2 prototype buffers differ; resolve before fitting")
    return values[0], {"checkpoint_buffers": evidence,
                       "prototype_bytes_sha256": hashlib.sha256(values[0].numpy().tobytes()).hexdigest(),
                       "learned_checkpoint_parameters_loaded": False}


def timing_check(bank, output, factory, logger):
    rows = []
    for variant in VARIANTS:
        seed_all(1234)
        inputs = feature_inputs(variant)
        model = factory(variant, "relation").cuda()
        optimizer = make_optimizer(model)
        datasets = [bank[d, "train"] for d in inner_domains("f1", "asv2019")]
        options = dict(training=True, seed=1234, last_layer_only=inputs != "clap")
        warmup = FeatureBatches(datasets, inputs, "cuda", max_batches=8, **options)
        train_epoch(model, warmup, optimizer, "cuda", accumulation_steps=1)
        train = FeatureBatches(datasets, inputs, "cuda", max_batches=64, **options)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses, _ = train_epoch(model, train, optimizer, "cuda", accumulation_steps=1)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        dev = FeatureBatches([bank["asv2019", "dev"]], inputs, "cuda", training=False,
                             seed=1234, last_layer_only=inputs != "clap")
        started = time.monotonic()
        scores, metrics = score_probe(model, dev)
        dev_seconds = time.monotonic() - started
        path = output / f"timing_{variant}.pt"
        torch.save({"scope": "timing only; never initializes a final ablation",
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)
        reloaded = factory(variant, "relation").cuda()
        reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
        other, other_metrics = score_probe(reloaded, dev)
        if not torch.equal(scores, other) or metrics != other_metrics:
            raise ValueError("Source-only ablation timing checkpoint reload failed")
        row = {"variant": variant, "batches": len(train), "seconds_per_train_batch": seconds / len(train),
               "seconds_per_dev_record": dev_seconds / dev.count, "losses": losses,
               "trainable_parameters": model.configuration()["trainable_parameters"],
               "source_dev_reload_bit_exact": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "checkpoint_sha256": checkpoint_digest(path)}
        rows.append(row)
        logger.info("TIMING %s: %.1f audio/s; params=%d; source reload exact", variant,
                    BATCH_SIZE / row["seconds_per_train_batch"], row["trainable_parameters"])
        del model, optimizer, reloaded
    upper = 0
    for row, fold, _seed in itertools.product(rows, OUTER_FOLDS, SEEDS):
        for held in FOLDS[fold]["sources"]:
            count = sum(len(bank[d, "train"].labels) for d in inner_domains(fold, held))
            upper += 30 * (math.ceil(count / BATCH_SIZE) * row["seconds_per_train_batch"]
                           + len(bank[held, "dev"].labels) * row["seconds_per_dev_record"])
    report = {"stage": 4, "source_only": True, "measurements": rows,
              "maximum_epoch_estimate_hours": upper / 3600, "recipe_candidates": 0,
              "scope": "8 warmup+64 measured updates per fixed variant; timing models discarded",
              "limitations": "short concurrent-GPU timing; early stopping reduces work; export/I/O variability not included"}
    save_json(output / "timing.json", report)
    logger.info("All source-only timing/reload checks passed; max-30-epoch estimate %.1fh", upper / 3600)


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
    logger = logging.getLogger("lacf.step4")
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    prototypes, prototype_evidence = fixed_prototypes()
    torch.save(prototypes, args.output / "prototypes.pt")
    paths = [Path(__file__).resolve(), ROOT / "scripts/lacf/train_probes.py", *[
        ROOT / "src/audio_deepfake_detection" / name for name in (
            "lacf/ablations.py", "lacf/probes.py", "lacf/model.py", "lacf/training.py",
            "lacf/precision.py", "lacf/data.py", "data.py", "protocol.py", "metrics.py")]]
    provenance = {"cache_root": str(args.cache_root.resolve()),
                  "cache_completion_sha256": checkpoint_digest(args.cache_root / "complete.json"),
                  "prototype_evidence": prototype_evidence,
                  "prototype_file_sha256": checkpoint_digest(args.output / "prototypes.pt"),
                  "code_sha256": {str(p.relative_to(ROOT)): checkpoint_digest(p) for p in paths},
                  "torch": str(torch.__version__), "transformers": transformers.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
    configs = {name: RelationAblation(name, prototypes).configuration() for name in VARIANTS}
    save_json(args.output / "queue_config.json", {"stage": 4, "folds": list(OUTER_FOLDS), "seeds": list(SEEDS),
              "variants": configs, "aliases": ALIASES, "expected_fits": EXPECTED_FITS,
              "provenance": provenance, "stop_after_stage": 4, "model_training_resume": False})
    logger.info("Step4: verify source cache once; load compact last-WavLM/CLAP arrays; no encoder or target access")
    bank = load_bank(args.cache_root, args.data_root, logger=logger, last_layer_only=True)
    save_json(args.output / "cache_validation.json", {"all_payload_sha256_checked": True,
              "all_canonical_ids_labels_checked": True, "cache_completion_sha256": provenance["cache_completion_sha256"],
              "source_splits": {f"{d}/{s}": {"records": len(v.labels), "unique_rows": len(v.wavlm)} for (d, s), v in bank.items()},
              "compact_wavlm": "last layer only; all original thirteen-layer payload hashes/dimensions still verified"})
    def factory(variant, head):
        if head != "relation":
            raise ValueError("Step4 uses the prescribed relation classifier only")
        return RelationAblation(variant, prototypes)
    timing_check(bank, args.output, factory, logger)
    results = []
    for variant, fold, seed in itertools.product(VARIANTS, OUTER_FOLDS, SEEDS):
        for held in FOLDS[fold]["sources"]:
            for name, expected in provenance["code_sha256"].items():
                if checkpoint_digest(ROOT / name) != expected:
                    raise ValueError("Ablation implementation changed after launch; queue stops")
            inputs = feature_inputs(variant)
            run = args.output / variant / f"inner_{held}" / fold / str(seed)
            results.append(fit_probe(bank, run, fold=fold, held=held, inputs=variant, head="relation", seed=seed,
                provenance=provenance, device="cuda", logger=logger, model_factory=factory, stage=4,
                feature_inputs=inputs, last_layer_only=inputs != "clap"))
            summary = {**aggregate(results, input_names=tuple(VARIANTS), head_names=("relation",)),
                       "stage": 4, "aliases": ALIASES, "expected_fits": EXPECTED_FITS, "complete": False}
            temporary = args.output / "summary.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(args.output / "summary.json")
            logger.info("QUEUE %d/%d verified fits; A9_no_bf reuses A8", len(results), EXPECTED_FITS)
    if len(results) != EXPECTED_FITS or len(summary["three_seed_summary"]) != 21:
        raise ValueError("Missing Step4 inner rotations/seeds")
    summary["complete"] = True
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_json(args.output / "complete.json", {"stage": 4, "verified_fits": len(results), "aliases": ALIASES,
              "summary_sha256": checkpoint_digest(args.output / "summary.json"),
              "completed_at": datetime.now(timezone.utc).isoformat(), "stop_before_stage": 5})
    logger.info("Step4 complete; stop before final report/redesign/target access and share source results")


if __name__ == "__main__":
    main()
