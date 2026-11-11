"""Fresh LACF Layer-MoE on all F1 sources; no encoders or target access."""

import argparse
import gc
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.layer_fusion.diagnostics import source_diagnostics
from audio_deepfake_detection.layer_fusion.model import LayerFusionDetector, PARAMETERS
from audio_deepfake_detection.lacf.probes import BATCH_SIZE, FeatureBatches
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import checkpoint_digest, save_score_manifest, verify_source_macro
from scripts.lacf.train_ablations import fixed_prototypes
from scripts.lacf.train_probes import save_json, seed_all
from scripts.token_fusion.train import ROOT, SOURCES, assert_bank, check_guards, load_f1_bank, score_sources


NAME = "LACF Layer-MoE"
STAGE = "lacf_layer_moe_full_f1"
OUTPUT = ROOT / "outputs/experiments/layer_fusion/lacf_layer_moe/f1/1234"
SCOPE = "full F1 source train/dev; selected in-domain source development, not inner-held-domain validation"


def checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("Interrupted checkpoint preserved")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit(bank, prototypes, output, provenance, device, logger, *, max_epochs=30, patience=5):
    assert_bank(bank)
    check_guards(provenance)
    output.mkdir(parents=True, exist_ok=False)
    if provenance.get("console_log"):
        (output / "training.log").hardlink_to(provenance["console_log"])
    seed_all(1234)
    model = LayerFusionDetector("C4", prototypes).to(device)
    optimizer = make_optimizer(model)
    train = FeatureBatches([bank[d, "train"] for d in SOURCES], "concat", device, training=True, seed=1234)
    config = {"stage": STAGE, "model_name": NAME, "implementation_variant": "C4", "fold": "f1", "seed": 1234, "seed_plan": [1234], "scope": SCOPE,
              "sources": list(SOURCES), "excluded_target": "speechfake", "target_access": False,
              "model": model.configuration(), "batch_size": BATCH_SIZE, "accumulation": 1,
              "optimizer": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "scheduler": "constant",
              "max_epochs": max_epochs, "patience": patience, "precision": "BF16 mixed; sensitive FP32",
              "sampling": "equal domain/class mass; replacement; recording count per epoch",
              "training_views": "common up-to10s; centered plus two fixed random views, uniform per draw",
              "development_views": "centered up-to10s; cached unpadded encoder features",
              "selection": "strict equal-domain macro source-dev EER; ties earlier",
              "foundation_encoders_loaded": False, "feature_workers": 0, "resume": False,
              "train_records": train.count, "dev_records": {d: len(bank[d, "dev"].labels) for d in SOURCES},
              "comparison_limits": "Different fusion/readout/capacity from TF2; adapted TF2 also changes encoder adaptation and has shutdown recoveries",
              "provenance": provenance}
    if config["model"]["trainable_parameters"] != PARAMETERS["C4"]:
        raise ValueError("C4 parameter inventory changed")
    save_json(output / "config.json", config)
    digest = checkpoint_digest(output / "config.json")
    logger.info("START LACF Layer-MoE full F1 seed1234 | train=%d dev=%s | %d trainable; both encoders frozen/cached",
                train.count, config["dev_records"], PARAMETERS["C4"])
    best, best_epoch, stale, selected_scores = float("inf"), 0, 0, None
    with (output / "history.jsonl").open("x") as history:
        for epoch in range(1, max_epochs + 1):
            check_guards(provenance)
            tick = time.monotonic()
            losses, updates = train_epoch(model, train, optimizer, device, accumulation_steps=1,
                                           logger=logger, phase=f"Layer-MoE epoch{epoch}")
            scores, metrics = score_sources(model, bank, device, logger)
            if metrics["macro_eer"] < best:
                best, best_epoch, stale, selected_scores = metrics["macro_eer"], epoch, 0, scores
                checkpoint(output / "best.pt", {"model_state_dict": model.state_dict(), "epoch": epoch,
                           "source_dev": metrics, "config_sha256": digest})
            else:
                stale += 1
            checkpoint(output / "last.pt", {"model_state_dict": model.state_dict(), "epoch": epoch,
                       "optimizer_state_dict": optimizer.state_dict(), "source_dev": metrics,
                       "best_epoch": best_epoch, "patience": stale, "config_sha256": digest})
            row = {"epoch": epoch, "losses": losses, "updates": updates, "source_dev": metrics,
                   "best_epoch": best_epoch, "patience": stale, "seconds": time.monotonic() - tick}
            history.write(json.dumps(row, allow_nan=False) + "\n"); history.flush()
            logger.info("EPOCH %d | macro source EER %.4f%% AUROC %.6f | best%d patience%d/%d | %.1fs",
                        epoch, metrics["macro_eer"]*100, metrics["macro_auroc"], best_epoch, stale, patience, row["seconds"])
            if stale >= patience:
                break
    state = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if state["config_sha256"] != digest or checkpoint_digest(output / "config.json") != digest:
        raise ValueError("Selected checkpoint/config binding changed")
    del model, optimizer
    gc.collect()
    reloaded = LayerFusionDetector("C4", prototypes).to(device)
    reloaded.load_state_dict(state["model_state_dict"], strict=True)
    scores, metrics = score_sources(reloaded, bank, device, logger)
    if metrics != state["source_dev"] or any(not torch.equal(scores[d], selected_scores[d]) for d in SOURCES):
        raise ValueError("Fresh selected-checkpoint source reload differs")
    inputs, hashes = {}, {}
    for domain in SOURCES:
        dataset = bank[domain, "dev"]; path = output / f"source_dev_{domain}.csv"
        write_score_csv(path, zip(dataset.ids, itertools.repeat(domain), dataset.labels.tolist(), scores[domain].tolist()))
        labels, exported = read_score_csv(path, domain, expected_labels=dict(zip(dataset.ids, dataset.labels.tolist())))
        order = sorted(range(len(dataset.ids)), key=dataset.ids.__getitem__)
        if not np.array_equal(exported, scores[domain].numpy()[order]):
            raise ValueError("Canonical source export changed scores")
        inputs[domain] = labels, exported; hashes[path.name] = checkpoint_digest(path)
    calibration = calibrate_source_scores(inputs)
    verify_source_macro(calibration["source_dev"]["macro_eer"], best)
    save_json(output / "source_calibration.json", calibration)
    save_json(output / "routing.json", source_diagnostics(reloaded, bank, "f1", 1234, device))
    save_json(output / "metrics.json", {"model_name": NAME, "condition": "C4", "scope": SCOPE, "source_dev": metrics,
              "best_epoch": best_epoch, "final_epoch": epoch, "target_access": False})
    complete = {"stage": STAGE, "model_name": NAME, "fold": "f1", "seed": 1234, "sources": list(SOURCES),
                "best_epoch": best_epoch, "final_epoch": epoch, "best_macro_source_dev_eer": best,
                "best_checkpoint_sha256": checkpoint_digest(output / "best.pt"), "config_sha256": digest,
                "metrics_sha256": checkpoint_digest(output / "metrics.json"), "source_scores_sha256": hashes,
                "calibration_sha256": checkpoint_digest(output / "source_calibration.json"),
                "routing_sha256": checkpoint_digest(output / "routing.json"),
                "selected_checkpoint_reloaded_and_source_verified": True, "target_access": False}
    save_json(output / "training_complete.json", complete)
    save_score_manifest(output, {"stage": STAGE, "fold": "f1", "seed": 1234, "sources": list(SOURCES),
                        "target_access": False, "checkpoint_sha256": complete["best_checkpoint_sha256"],
                        "training_completion_sha256": checkpoint_digest(output / "training_complete.json"),
                        "score_sha256": hashes})
    logger.info("COMPLETE LACF Layer-MoE full F1 | selected epoch%d EER%.4f%% | source reload/export/calibration verified; no target", best_epoch, best*100)
    return complete


def preflight(bank, prototypes, provenance, device, logger):
    check_guards(provenance); seed_all(1234)
    model = LayerFusionDetector("C4", prototypes).to(device); optimizer = make_optimizer(model)
    loader = FeatureBatches([bank[d, "train"] for d in SOURCES], "concat", device, training=True, seed=1234, max_batches=32)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.monotonic()
    losses, updates = train_epoch(model, loader, optimizer, device, accumulation_steps=1)
    torch.cuda.synchronize(); train_seconds = time.monotonic() - started
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    reference, _ = score_sources(model, bank, device)
    fresh = LayerFusionDetector("C4", prototypes).to(device); fresh.load_state_dict(state, strict=True)
    tick = time.monotonic(); actual, _ = score_sources(fresh, bank, device)
    torch.cuda.synchronize(); dev_seconds = time.monotonic() - tick
    if any(not torch.equal(reference[d], actual[d]) for d in SOURCES):
        raise ValueError("C4 source GPU reload differs")
    report = {"passed": True, "target_access": False, "losses": losses, "updates": updates,
              "train_examples_per_second": loader.count/train_seconds,
              "dev_examples_per_second": sum(len(v) for v in actual.values())/dev_seconds,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "source_reload_bit_exact": True, "probe_weights_used_for_final_training": False}
    report["rough_epoch_seconds"] = (sum(len(bank[d, "train"].labels) for d in SOURCES)/report["train_examples_per_second"]
                                      + sum(len(bank[d, "dev"].labels) for d in SOURCES)/report["dev_examples_per_second"])
    report["timing_limit"] = "Short training timing and complete cached dev pass while sharing GPU; not a duration guarantee"
    logger.info("PREFLIGHT PASSED | %.1f train records/s | %.1f dev records/s | %.3fGiB allocated",
                report["train_examples_per_second"], report["dev_examples_per_second"], report["peak_allocated_bytes"]/1024**3)
    del model, optimizer, fresh; gc.collect(); torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/experiments/lacf_diagnostics/frozen_features")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT.parents[1].resolve()) or args.output.name != "1234" or args.output.parent.name != "f1":
        parser.error("Use an experiments/layer_fusion/lacf_layer_moe/ full-F1 seed1234 run")
    if args.output.exists(): raise FileExistsError("Existing run preserved")
    launch = args.output.parent.parent / "launch"
    launch.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(launch / "console.log")])
    logger = logging.getLogger("lacf_moe_full_f1")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required; no fallback")
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.nnpack.set_flags(False)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    paths = [Path(__file__).resolve(), ROOT / "scripts/token_fusion/train.py", ROOT / "scripts/lacf/train_ablations.py",
             *[(ROOT / f"src/audio_deepfake_detection/layer_fusion/{n}.py") for n in ("model", "diagnostics")],
             ROOT / "src/audio_deepfake_detection/lacf/ablations.py",
             ROOT / "scripts/experiments/layer_fusion/evaluate_lacf_layer_moe.py",
             ROOT / "scripts/experiments/token_fusion/evaluate.py",
             ROOT / "scripts/evaluate_scores.py"]
    guards = json.loads((ROOT / "outputs/experiments/token_fusion/adaptation_restart/launch/source_guards.json").read_text())
    frozen = json.loads((ROOT / "outputs/experiments/tf2_frozen_clap/f1/seed_1234/config.json").read_text())
    for path, expected in frozen["provenance"]["code_sha256"].items():
        if path in guards and guards[path] != expected:
            raise ValueError("Concurrent TF2 source identities disagree")
        guards[path] = expected
    provenance = {"code_sha256": {**guards, **{str(p.relative_to(ROOT)):checkpoint_digest(p) for p in paths}},
                  "cache": str(args.cache.resolve()), "cache_config_sha256": checkpoint_digest(args.cache / "config.json"),
                  "cache_completion_sha256": checkpoint_digest(args.cache / "complete.json"),
                  "cache_execution_sha256": checkpoint_digest(args.cache / "execution_accelerated.json"),
                  "console_log": str((launch / "console.log").resolve()),
                  "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                  "gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads(),
                                  "gpu_allocator_budget_gib": 1}}
    save_json(launch / "source_guards.json", provenance["code_sha256"])
    check_guards(provenance)
    bank = load_f1_bank(args.cache, args.data_root, logger)
    prototypes, prototype_identity = fixed_prototypes(); provenance["prototypes"] = prototype_identity
    save_json(launch / "preflight.json", preflight(bank, prototypes, provenance, "cuda", logger))
    result = fit(bank, prototypes, args.output, provenance, "cuda", logger)
    save_json(launch / "complete.json", {"source_verified": True, "run": str(args.output.resolve()),
              "completion_sha256": checkpoint_digest(args.output / "training_complete.json"), "best_epoch": result["best_epoch"]})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("LACF Layer-MoE failed; outputs preserved, no automatic retry")
        raise
