"""Evaluate the source-selected LACF Layer-MoE F1 model using existing frozen features."""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.layer_fusion.model import LayerFusionDetector
from audio_deepfake_detection.lacf.diagnostic_cache import crop_intervals
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_split, score_probe
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    canonical_score_labels, checkpoint_digest, load_score_manifest, save_score_manifest,
    verify_completed_run, verify_score_digest, verify_source_macro,
)
from scripts.evaluate_scores import evaluate_run
from scripts.experiments.layer_fusion.train_full_f1 import NAME, OUTPUT, STAGE
from scripts.experiments.token_fusion.evaluate import score_features
from scripts.token_fusion.train import ROOT, SOURCES, check_guards, save_json


EVALUATION = ROOT / "outputs/experiments/layer_fusion/evaluations/lacf_layer_moe/f1/1234"
TARGET_FEATURES = ROOT / "outputs/experiments/token_fusion/eval_f1_speechfake_20261030T131151+0100/target_features"


def read(path):
    return json.loads(path.read_text())


def verify_training(run, data_root):
    """No target reads: verify completed source selection, hashes and calibration."""
    config = read(run / "config.json")
    done, digest = verify_completed_run(run, "f1", 1234)
    if (config.get("stage") != STAGE or config.get("model_name") != NAME
            or config.get("sources") != list(SOURCES) or config.get("seed_plan") != [1234]
            or config.get("fold") != "f1" or config.get("seed") != 1234
            or config.get("target_access") is not False or config["model"]["condition"] != "C4"
            or done.get("selected_checkpoint_reloaded_and_source_verified") is not True
            or done.get("best_checkpoint_sha256") != digest):
        raise ValueError("Completed Layer-MoE source/model identity differs")
    for filename, key in (("config.json", "config_sha256"), ("metrics.json", "metrics_sha256"),
                          ("source_calibration.json", "calibration_sha256"), ("routing.json", "routing_sha256")):
        if checkpoint_digest(run / filename) != done[key]:
            raise ValueError(f"Training artifact changed: {filename}")
    check_guards(config["provenance"])
    manifest = read(run / "score_manifest.json")
    if (manifest.get("stage") != STAGE or manifest.get("fold") != "f1" or manifest.get("seed") != 1234
            or manifest.get("checkpoint_sha256") != digest
            or manifest.get("training_completion_sha256") != checkpoint_digest(run / "training_complete.json")
            or set(manifest.get("score_sha256", {})) != {f"source_dev_{d}.csv" for d in SOURCES}):
        raise ValueError("Training source-score manifest differs")
    source = {}
    for domain in SOURCES:
        path = run / f"source_dev_{domain}.csv"
        verify_score_digest(path, manifest)
        if checkpoint_digest(path) != done["source_scores_sha256"][path.name]:
            raise ValueError("Source CSV completion binding changed")
        source[domain] = read_score_csv(path, domain, expected_labels=canonical_score_labels(domain, "dev", data_root))
    calibration = calibrate_source_scores(source)
    if calibration != read(run / "source_calibration.json"):
        raise ValueError("Source calibration differs")
    verify_source_macro(calibration["source_dev"]["macro_eer"], done["best_macro_source_dev_eer"])
    return config, done


def load_detector(run, device):
    config, done = read(run / "config.json"), read(run / "training_complete.json")
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if (state["config_sha256"] != checkpoint_digest(run / "config.json") or state["epoch"] != done["best_epoch"]
            or state["source_dev"] != read(run / "metrics.json")["source_dev"]):
        raise ValueError("Selected checkpoint/config/epoch differs")
    model = LayerFusionDetector("C4", state["model_state_dict"]["backend.prototypes"])
    model.load_state_dict(state["model_state_dict"], strict=True)
    if json.loads(json.dumps(model.configuration())) != config["model"]:
        raise ValueError("Layer-MoE architecture differs")
    return model.to(device).requires_grad_(False).eval()


def source_reload(model, run, config, data_root, logger, device):
    cache = Path(config["provenance"]["cache"])
    provenance = config["provenance"]
    for name, key in (("config.json", "cache_config_sha256"), ("complete.json", "cache_completion_sha256"),
                      ("execution_accelerated.json", "cache_execution_sha256")):
        if checkpoint_digest(cache / name) != provenance[key]:
            raise ValueError("Frozen source feature-cache identity differs")
    marker = read(cache / "complete.json")
    checks = []
    for domain in SOURCES:
        dataset = load_split(cache, domain, "dev", marker["splits"][f"{domain}/dev"], data_root, logger=logger)
        scores, metrics = score_probe(model, FeatureBatches([dataset], "concat", device, training=False, seed=1234))
        order = sorted(range(len(dataset.ids)), key=dataset.ids.__getitem__)
        _, exported = read_score_csv(run / f"source_dev_{domain}.csv", domain,
                                    expected_labels=dict(zip(dataset.ids, dataset.labels.tolist())))
        if not np.array_equal(scores.numpy()[order], exported):
            raise ValueError("Fresh full-source selected-checkpoint reload differs; target held")
        checks.append({"domain": domain, "records": len(scores), "score_bit_exact": True, **metrics})
        logger.info("SOURCE RELOAD %s passed: %d scores bit-exact", domain, len(scores))
        del dataset
        gc.collect()
    return {"passed": True, "target_access": False, "checks": checks,
            "scope": "all three canonical source development sets; fresh selected checkpoint and frozen calibration"}


def target_chunks(features, source_cache, data_root):
    """Only called after source gates; validate every reused target payload and crop."""
    marker = read(features / "complete.json")
    extraction = read(features.parent / "evaluation_config.json")
    if (checkpoint_digest(features.parent / "evaluation_config.json") != marker["evaluation_config_sha256"]
            or extraction["provenance"]["cache_completion_sha256"] != checkpoint_digest(source_cache / "complete.json")
            or Path(extraction["provenance"]["cache_root"]).resolve() != source_cache.resolve()
            or extraction["batching"] != {"wavlm": 32, "clap": 64, "head": 132, "chunk": 1024, "workers": 8, "prefetch": 2}):
        raise ValueError("Target feature extraction/source-cache identity differs")
    execution = read(source_cache / "execution_accelerated.json")
    if extraction["frozen_encoder_revisions"] != execution["pretrained"]:
        raise ValueError("Target frozen encoder revisions/configurations differ")
    expected = canonical_score_labels("speechfake", "test", data_root)
    if (marker["records"] != len(expected)
            or hashlib.sha256(json.dumps(list(expected.items())).encode()).hexdigest() != marker["canonical_ids_labels_sha256"]):
        raise ValueError("Target feature membership differs")
    canonical = list(expected.items())
    offset = 0
    for index, descriptor in enumerate(marker["chunks"]):
        if descriptor["path"] != f"chunk_{index:05d}":
            raise ValueError("Nonsequential target cache")
        directory = features / descriptor["path"]
        if checkpoint_digest(directory / "complete.json") != descriptor["completion_sha256"]:
            raise ValueError("Target shard completion changed")
        chunk = read(directory / "complete.json")
        for name in ("records.json", "wavlm_layers.npy", "clap_projected.npy"):
            if checkpoint_digest(directory / name) != chunk["files_sha256"][name]:
                raise ValueError("Target cached feature payload changed")
        rows = read(directory / "records.json")
        if len(rows) != chunk["records"] or not rows or offset+len(rows) > len(canonical):
            raise ValueError("Target shard count differs")
        for local, row in enumerate(rows):
            identifier, label = canonical[offset+local]
            intervals, aliases = crop_intervals(row["frames"], row["rate"], row["id"], training=False)
            if (row["index"] != offset+local or row["id"] != f"speechfake/test/{identifier}"
                    or row["label"] != label or row["intervals"] != [list(x) for x in intervals]
                    or row["view_rows"] != aliases):
                raise ValueError("Target ordering, labels or centered physical crop differs")
        arrays = {name: np.load(directory / f"{name}.npy", allow_pickle=False)
                  for name in ("wavlm_layers", "clap_projected")}
        for name, shape in (("wavlm_layers", (len(rows), 13, 768)), ("clap_projected", (len(rows), 512))):
            if arrays[name].dtype != np.float32 or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError("Invalid target features; no inputs may be skipped")
        yield [name for name, _ in canonical[offset:offset+len(rows)]], arrays
        offset += len(rows)
    if offset != len(canonical):
        raise ValueError("Incomplete target cache")


def evaluate(training, output, features, data_root, device, logger, *, bootstrap_resamples=1000):
    if output.exists():
        raise FileExistsError("Existing evaluation preserved; no overwrite or retry")
    if output.name != "1234" or output.parent.name != "f1":
        raise ValueError("Use canonical f1/1234 evaluation path")
    config, done = verify_training(training, data_root)
    model = load_detector(training, device)
    validation = source_reload(model, training, config, data_root, logger, device)
    # No target labels, feature metadata or arrays have been read above.
    check_guards(config["provenance"])
    output.mkdir(parents=True, exist_ok=False)
    for name in ("best.pt", "config.json", "training_complete.json", "source_calibration.json"):
        shutil.copy2(training / name, output / name)
    shutil.copy2(training / "metrics.json", output / "source_metrics.json")
    save_json(output / "source_validation.json", validation)
    manifest = load_score_manifest(output, "f1", 1234, done["best_checkpoint_sha256"], create=True)
    for domain in SOURCES:
        name = f"source_dev_{domain}.csv"
        shutil.copy2(training / name, output / name)
        manifest["score_sha256"][name] = checkpoint_digest(output / name)
    inference = {"input": "centered common physical interval up to10s", "precision": config["precision"],
                 "features": "reused verified raw WavLM13-layer/CLAP projections; no encoder reruns",
                 "head_batch": 132, "chunk_records": 1024, "score": "scalar spoof logit"}
    save_json(output / "evaluation_config.json", {"model_name": NAME, "fold": "f1", "seed": 1234,
              "training_run": str(training.resolve()), "training_completion_sha256": checkpoint_digest(training / "training_complete.json"),
              "target_features": str(features.resolve()), "target": "speechfake/test", "inference": inference,
              "exploratory_after_prior_target_exposure": True, "target_tuning": False,
              "bootstrap_resamples": bootstrap_resamples, "bootstrap_seed": 2026})
    manifest.update(model_family="lacf_layer_moe", inference=inference, training_run=str(training.resolve()),
                    training_metrics_copy="source_metrics.json", exploratory_after_prior_target_exposure=True,
                    source_validation_sha256=checkpoint_digest(output / "source_validation.json"),
                    target_features_sha256=checkpoint_digest(features / "complete.json"),
                    scorer_sha256={"exporter": checkpoint_digest(Path(__file__)),
                        "head": checkpoint_digest(ROOT / "src/audio_deepfake_detection/experiments/layer_fusion/model.py"),
                        "cached_head_scoring": checkpoint_digest(ROOT / "scripts/experiments/token_fusion/evaluate.py")})
    ids, values = [], []
    for identifiers, arrays in target_chunks(features, Path(config["provenance"]["cache"]), data_root):
        check_guards(config["provenance"])
        ids.extend(identifiers)
        values.extend(score_features(model, arrays, device).tolist())
        if len(ids) % 10240 == 0:
            logger.info("LACF Layer-MoE target scoring: %d recordings", len(ids))
    expected = canonical_score_labels("speechfake", "test", data_root)
    if ids != list(expected) or len(values) != len(expected):
        raise ValueError("Incomplete or reordered target export")
    target = output / "target_scores.csv"
    write_score_csv(target, zip(ids, itertools.repeat("speechfake"), expected.values(), values))
    manifest["score_sha256"][target.name] = checkpoint_digest(target)
    save_score_manifest(output, manifest)
    del model
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    logger.info("METRICS: shared raw/transductive transfer and %d bootstrap resamples", bootstrap_resamples)
    report = evaluate_run(output, "f1", seed=1234, source_data_root=data_root,
                          target_data_root=data_root, bootstrap_resamples=bootstrap_resamples)
    check_guards(config["provenance"])
    save_json(output / "evaluation_complete.json", {"model_name": NAME, "fold": "f1", "seed": 1234,
        "checkpoint_sha256": done["best_checkpoint_sha256"], "metrics_sha256": checkpoint_digest(output / "metrics.json"),
        "source_validation_sha256": checkpoint_digest(output / "source_validation.json"),
        "score_sha256": manifest["score_sha256"], "source_selected_checkpoint_only": True,
        "exploratory_after_prior_target_exposure": True, "no_target_tuning": True, "stop_after_evaluation": True})
    logger.info("COMPLETE %s F1 | target EER %.4f%% AUROC %.6f", NAME, report["target"]["eer"]*100, report["target"]["auroc"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=EVALUATION)
    parser.add_argument("--target-features", type=Path, default=TARGET_FEATURES)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(EVALUATION.parents[1].resolve()):
        parser.error("Keep evaluation inside experiments/layer_fusion/evaluations/lacf_layer_moe/")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required")
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    evaluate(args.training_run, args.output, args.target_features, args.data_root,
             torch.device("cuda"), logging.getLogger("lacf_layer_moe_evaluation"))


if __name__ == "__main__":
    main()
