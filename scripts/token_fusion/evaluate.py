"""Fixed TF0–TF3 F1/SpeechFake comparison; one shared frozen encoder pass."""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import shutil
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor, Wav2Vec2FeatureExtractor, WavLMModel

from audio_deepfake_detection.lacf.pretrained import DualViewCollator
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_split, score_probe
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    canonical_score_labels, checkpoint_digest, load_score_manifest, read_dataset,
    save_score_manifest, utterance_id, verify_completed_run, verify_score_digest, verify_source_macro,
)
from audio_deepfake_detection.token_fusion.model import CONDITIONS, TokenFusion
from scripts.evaluate_scores import evaluate_run
from scripts.lacf.cache_diagnostics import encode_chunk, loader
from scripts.token_fusion.train import (
    ROOT, SEEDS, SOURCES, STAGE, check_guards, replace_json, save_json,
)


def read(path):
    return json.loads(path.read_text())


def verified_training_run(run, expected):
    """Canonical sources and frozen calibration must pass before any target read."""
    config, metrics = read(run / "config.json"), read(run / "metrics.json")
    completion, digest = verify_completed_run(run, "f1", expected["seed"])
    if (config["stage"] != STAGE or config["model"]["condition"] != expected["condition"]
            or config["seed"] != expected["seed"] or config["fold"] != "f1"
            or config["sources"] != list(SOURCES)
            or metrics != {k: v for k, v in expected.items() if k != "canonical_sources"}
            or not completion.get("selected_checkpoint_reloaded_and_source_verified")
            or completion["best_checkpoint_sha256"] != digest):
        raise ValueError("Training completion/model identity changed")
    for filename, key in (("config.json", "config_sha256"), ("metrics.json", "metrics_sha256"),
                          ("source_calibration.json", "calibration_sha256")):
        if checkpoint_digest(run / filename) != completion[key]:
            raise ValueError(f"Training artifact changed: {filename}")
    manifest = read(run / "score_manifest.json")
    if (manifest["checkpoint_sha256"] != digest
            or manifest["training_completion_sha256"] != checkpoint_digest(run / "training_complete.json")):
        raise ValueError("Training score manifest changed")
    source = {}
    for domain in SOURCES:
        path = run / f"source_dev_{domain}.csv"
        verify_score_digest(path, manifest)
        source[domain] = read_score_csv(path, domain, expected_labels=expected["canonical_sources"][domain])
    calibration = calibrate_source_scores(source)
    if calibration != read(run / "source_calibration.json"):
        raise ValueError("Frozen source calibration changed")
    verify_source_macro(calibration["source_dev"]["macro_eer"], completion["best_macro_source_dev_eer"])
    return config, completion


def load_detector(run, device):
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    config = read(run / "config.json")
    if checkpoint["config"] != config or checkpoint["epoch"] != read(run / "training_complete.json")["best_epoch"]:
        raise ValueError("Selected checkpoint configuration/epoch differs")
    state = checkpoint["model_state_dict"]
    model = TokenFusion(config["model"]["condition"], state.get("prototypes"))
    model.load_state_dict(state, strict=True)
    if model.configuration() != config["model"]:
        raise ValueError("Model architecture differs from training")
    return model.to(device).requires_grad_(False).eval()


def prepare_runs(training, output, data_root, device):
    complete, summary = read(training / "complete.json"), read(training / "summary.json")
    if complete["summary_sha256"] != checkpoint_digest(training / "summary.json") or complete["verified_fits"] != 12:
        raise ValueError("Training queue completion changed")
    expected_pairs = set(itertools.product(CONDITIONS, SEEDS))
    if len(summary["fits"]) != 12 or {(r["condition"], r["seed"]) for r in summary["fits"]} != expected_pairs:
        raise ValueError("All twelve predetermined models are required; no target-based seed/model selection")
    canonical = {d: canonical_score_labels(d, "dev", data_root) for d in SOURCES}
    jobs = []
    for result in summary["fits"]:
        original = training / result["condition"] / "f1" / str(result["seed"])
        # Keep the original result object unchanged when comparing its saved JSON.
        expected = {**result, "canonical_sources": canonical}
        config, completion = verified_training_run(original, expected)
        check_guards(config["provenance"])
        run = output / result["condition"] / "f1" / str(result["seed"])
        run.mkdir(parents=True, exist_ok=False)
        for name in ("best.pt", "config.json", "training_complete.json", "source_calibration.json"):
            shutil.copy2(original / name, run / name)
        shutil.copy2(original / "metrics.json", run / "source_metrics.json")
        manifest = load_score_manifest(run, "f1", result["seed"], completion["best_checkpoint_sha256"], create=True)
        for domain in SOURCES:
            name = f"source_dev_{domain}.csv"
            shutil.copy2(original / name, run / name)
            manifest["score_sha256"][name] = checkpoint_digest(run / name)
        manifest.update(model_family="token_fusion", training_run=str(original.resolve()),
                        condition=result["condition"], exploratory_after_prior_target_exposure=True,
                        training_metrics_copy="source_metrics.json",
                        completion_record_scope="original training snapshot; its metrics_sha256 binds source_metrics.json",
                        inference={"input": "centered common physical interval up to 10 seconds",
                                   "feature_policy": "same fixed diagnostic extraction; no WavLM waveform padding",
                                   "precision": config["precision"], "score": "scalar spoof logit"})
        save_score_manifest(run, manifest)
        jobs.append({"run": run, "original": original, "condition": result["condition"],
                     "seed": result["seed"], "model": load_detector(run, device), "manifest": manifest})
    return jobs


@torch.inference_mode()
def score_features(model, arrays, device):
    """Keep the trained head's batch132 arithmetic and FP32 score direction."""
    values = []
    for start in range(0, len(arrays["wavlm_layers"]), 132):
        batch = {"wavlm": torch.from_numpy(arrays["wavlm_layers"][start:start + 132]).to(device),
                 "clap": torch.from_numpy(arrays["clap_projected"][start:start + 132]).to(device)}
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            scores = model(batch)["logit"].float()
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite spoof scores; no examples may be skipped")
        values.append(scores.cpu())
    return torch.cat(values).numpy()


def verify_source_reload(jobs, cache, data_root, output, logger):
    """Full canonical source-dev scoring with each freshly loaded checkpoint."""
    root = read(cache / "complete.json")
    checks = []
    for domain in SOURCES:
        logger.info("SOURCE VERIFY %s: load centered cached development features", domain)
        dataset = load_split(cache, domain, "dev", root["splits"][f"{domain}/dev"], data_root, logger=logger)
        for job in jobs:
            scores, metrics = score_probe(job["model"], FeatureBatches(
                [dataset], "concat", "cuda", training=False, seed=job["seed"]))
            order = sorted(range(len(dataset.ids)), key=dataset.ids.__getitem__)
            _, expected = read_score_csv(job["run"] / f"source_dev_{domain}.csv", domain,
                                        expected_labels=dict(zip(dataset.ids, dataset.labels.tolist())))
            if not np.array_equal(scores.numpy()[order], expected):
                raise ValueError(f"Fresh full-source reload differs: {domain}/{job['condition']}/{job['seed']}")
            checks.append({"domain": domain, "condition": job["condition"], "seed": job["seed"],
                           "count": len(scores), "score_bit_exact": True, **metrics})
        del dataset
        gc.collect()
    save_json(output / "source_validation.json", {"passed": True, "checks": checks,
              "scope": "all three canonical cached source dev sets, all 12 fresh checkpoints; no target access"})


def load_encoders(cache):
    identity = read(cache / "execution_accelerated.json")
    pretrained = identity["pretrained"]
    wrev, crev = pretrained["wavlm_revision"], pretrained["clap_revision"]
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", revision=wrev,
                                     local_files_only=True, dtype=torch.float32).requires_grad_(False).eval()
    clap = ClapModel.from_pretrained("laion/clap-htsat-unfused", revision=crev,
                                   local_files_only=True, dtype=torch.float32).requires_grad_(False).eval()
    wprocessor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus", revision=wrev, local_files_only=True)
    cprocessor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", revision=crev, local_files_only=True)
    collator = DualViewCollator(wprocessor, cprocessor)
    if collator.configuration() != identity["processors"]:
        raise ValueError("Frozen source-cache processor policy changed")
    return SimpleNamespace(wavlm=wavlm.cuda(), clap=clap.cuda()), collator


def source_encoder_check(encoders, collator, cache, data_root, jobs, output, logger):
    """Source-only batch32 maximum-duration stress before opening test metadata."""
    directory = cache / "asv5/dev/chunk_00000"
    metadata = read(directory / "records.json")
    canonical = read_dataset("asv5", "dev", data_root)
    selected = [r for r in metadata if r["frames"] >= 10 * r["rate"]]
    if not selected:
        raise ValueError("No source 10-second input for the fixed memory check")
    selected = [selected[i % len(selected)] for i in range(32)]
    records = [canonical[r["index"]] for r in selected]
    names = [f"source-stress/{i}" for i in range(32)]
    prepared = list(loader(records, names, collator, training=False, workers=4))
    if any(row["views"][0]["wavlm"]["input_values"].shape[-1] != 160000 for row in prepared):
        raise ValueError("Maximum-duration source stress input changed")
    torch.cuda.reset_peak_memory_stats()
    tick = time.monotonic()
    arrays = encode_chunk(encoders, prepared, 32, 64)
    # Batch32 is the source-cache's approved execution policy, not a target search.
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite source encoder features")
    original = {k: np.load(directory / f"{k}.npy", mmap_mode="r")[
        [r["view_rows"][0] for r in selected]].copy() for k in arrays}
    drift = {}
    for key in arrays:
        a, b = arrays[key].reshape(32, -1).astype(np.float64), original[key].reshape(32, -1).astype(np.float64)
        cosine = (a*b).sum(1) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
        if not np.isfinite(cosine).all() or cosine.min() < 0.99:
            save_json(output / "source_encoder_failure.json", {"feature": key,
                      "minimum_cosine": float(cosine.min()), "status": "stopped before target access"})
            raise ValueError("Source feature drift exceeds cosine0.99 correctness guard; inspect before target access")
        # Existing cache mixes original batch-one and accelerated BF16 shards.
        # Record rounding, without claiming direct/cache bit-exactness.
        drift[key] = {"minimum_cosine": float(cosine.min()),
                      "max_absolute_difference": float(np.max(np.abs(a-b))),
                      "bit_exact": bool(np.array_equal(a, b))}
    for job in jobs:
        expected = score_features(job["model"], arrays, "cuda")
        fresh = load_detector(job["run"], "cuda")
        actual = score_features(fresh, arrays, "cuda")
        if not np.array_equal(expected, actual):
            raise ValueError("Direct source-feature checkpoint reload differs")
        del fresh
    report = {"passed": True, "records": 32, "seconds": time.monotonic()-tick,
              "wavlm_batch": 32, "clap_batch": 64, "source_feature_drift": drift,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "fresh_head_reload_bit_exact": True,
              "numerical_policy": "original/accelerated diagnostic cache BF16 batch rounding disclosed",
              "source_drift_guard": "minimum cosine0.99 against preserved features; correctness stop, not a recipe search"}
    save_json(output / "source_encoder_check.json", report)
    logger.info("SOURCE ENCODER CHECK PASSED | %.2fs | %.2f GiB peak", report["seconds"], report["peak_allocated_bytes"]/1024**3)


def extract_and_score(jobs, encoders, collator, data_root, output, guards, logger):
    # This is the first target read, after all source checks and model freeze.
    records = read_dataset("speechfake", "test", data_root)
    expected = canonical_score_labels("speechfake", "test", data_root, records=records)
    ids = list(expected)
    names = [f"speechfake/test/{identifier}" for identifier in ids]
    stream = iter(loader(records, names, collator, training=False, workers=8))
    features = output / "target_features"
    features.mkdir(exist_ok=False)
    chunks, done, started = [], 0, time.monotonic()
    buffers = [np.empty(len(records), dtype=np.float32) for _ in jobs]
    while prepared := list(itertools.islice(stream, 1024)):
        check_guards(guards)
        if [row["index"] for row in prepared] != list(range(done, done+len(prepared))):
            raise ValueError("Target order changed")
        arrays = encode_chunk(encoders, prepared, 32, 64)
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("Nonfinite target features")
        chunk = features / f"chunk_{len(chunks):05d}"
        chunk.mkdir(exist_ok=False)
        for key, array in arrays.items():
            with (chunk / f"{key}.npy").open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
        save_json(chunk / "records.json", [{key: value for key, value in r.items() if key != "views"} for r in prepared])
        save_json(chunk / "complete.json", {"records": len(prepared), "files_sha256": {
            p.name: checkpoint_digest(p) for p in chunk.iterdir()}})
        chunks.append({"path": chunk.name, "completion_sha256": checkpoint_digest(chunk / "complete.json")})
        for job, values in zip(jobs, buffers):
            values[done:done+len(prepared)] = score_features(job["model"], arrays, "cuda")
        done += len(prepared)
        seconds = time.monotonic()-started
        rate = done/seconds
        progress = {"phase": "frozen_target_features_and_12_heads", "scored": done, "total": len(records),
                    "examples_per_second": rate, "eta_hours": (len(records)-done)/rate/3600}
        replace_json(output / "progress.json", progress)
        logger.info("TEST FEATURES +12 HEADS %d/%d (%.1f%%) | %.1f recordings/s | ETA %.2fh",
                    done, len(records), done*100/len(records), rate, progress["eta_hours"])
    if done != len(records):
        raise ValueError("Incomplete target feature export")
    save_json(features / "complete.json", {"records": done, "chunks": chunks,
              "canonical_ids_labels_sha256": hashlib.sha256(json.dumps(list(expected.items())).encode()).hexdigest(),
              "evaluation_config_sha256": checkpoint_digest(output / "evaluation_config.json")})
    feature_digest = checkpoint_digest(features / "complete.json")
    for job, values in zip(jobs, buffers):
        path = job["run"] / "target_scores.csv"
        write_score_csv(path, zip(ids, itertools.repeat("speechfake"), expected.values(), values.tolist()))
        manifest = job["manifest"]
        manifest.update(target_features_sha256=feature_digest,
                        source_validation_sha256=checkpoint_digest(output / "source_validation.json"),
                        evaluation_config_sha256=checkpoint_digest(output / "evaluation_config.json"),
                        source_encoder_check_sha256=checkpoint_digest(output / "source_encoder_check.json"),
                        scorer_sha256={"exporter": checkpoint_digest(Path(__file__)),
                            "head": checkpoint_digest(ROOT / "src/audio_deepfake_detection/token_fusion/model.py"),
                            "encoder_features": checkpoint_digest(ROOT / "src/audio_deepfake_detection/lacf/diagnostic_cache.py")})
        manifest["score_sha256"][path.name] = checkpoint_digest(path)
        save_score_manifest(job["run"], manifest)


def target_summary(rows):
    summaries = []
    for condition in CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        if {r["seed"] for r in selected} != set(SEEDS) or len(selected) != 3:
            raise ValueError("All three seeds required; no test-selected subset")
        summaries.append({"condition": condition, **{
            metric+suffix: function(r[metric] for r in selected)
            for metric in ("eer", "auroc") for suffix, function in (("_mean", statistics.mean), ("_std", statistics.stdev))}})
    return {"fold": "f1", "target": "speechfake/test", "fits": rows, "three_seed_summary": summaries,
            "exploratory_after_prior_target_exposure": True, "selected_global_winner": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Evaluation output already exists; no overwrite or automatic restart")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required; no precision fallback")
    torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("token_fusion_evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    queue = read(args.training_root / "queue_config.json")
    provenance = queue["provenance"]
    check_guards(provenance)
    cache = Path(provenance["cache_root"])
    marker = read(cache / "complete.json")
    if (checkpoint_digest(cache / "complete.json") != provenance["cache_completion_sha256"]
            or checkpoint_digest(cache / "config.json") != marker["config_sha256"]
            or checkpoint_digest(cache / "execution_accelerated.json") != marker["accelerated_execution_sha256"]):
        raise ValueError("Frozen training feature-cache identity changed")
    jobs = prepare_runs(args.training_root, args.output, args.data_root, "cuda")
    guards = {**provenance, "code_sha256": {**provenance["code_sha256"],
        str(Path(__file__).resolve().relative_to(ROOT)): checkpoint_digest(Path(__file__)),
        "scripts/lacf/cache_diagnostics.py": checkpoint_digest(ROOT / "scripts/lacf/cache_diagnostics.py"),
        "src/audio_deepfake_detection/lacf/diagnostic_cache.py": checkpoint_digest(ROOT / "src/audio_deepfake_detection/lacf/diagnostic_cache.py")}}
    save_json(args.output / "evaluation_config.json", {
        "owner_authorization": "30 October: evaluate completed new F1 models on unseen data for comparison",
        "training_root": str(args.training_root.resolve()), "training_completion_sha256": checkpoint_digest(args.training_root / "complete.json"),
        "models": [{"condition": j["condition"], "seed": j["seed"], "checkpoint_sha256": j["manifest"]["checkpoint_sha256"]} for j in jobs],
        "provenance": guards, "frozen_encoder_revisions": read(cache / "execution_accelerated.json")["pretrained"],
        "target": "SpeechFake test only; F4 and external sets remain held",
        "input": "centered common physical interval up to 10s, independent16/48kHz; unpadded WavLM, CLAP repeatpad",
        "batching": {"wavlm": 32, "clap": 64, "head": 132, "chunk": 1024, "workers": 8, "prefetch": 2},
        "gpu_allocator_budget_gib": 10, "precision": "BF16 mixed / FP32 sensitive", "tf32": False,
        "target_copy_status": "owner reported source SHA256 verified; destination presence/sizes matched; full destination checksums waived",
        "reporting": "all12 checkpoints; raw and transductive; 1000 class-stratified bootstrap seed2026; three-seed sampleSD",
        "exploratory_after_prior_target_exposure": True, "target_tuning": False, "resume": False})
    verify_source_reload(jobs, cache, args.data_root, args.output, logger)
    encoders, collator = load_encoders(cache)
    source_encoder_check(encoders, collator, cache, args.data_root, jobs, args.output, logger)
    check_guards(guards)
    extract_and_score(jobs, encoders, collator, args.data_root, args.output, guards, logger)
    del encoders
    for job in jobs:
        del job["model"]
    gc.collect()
    torch.cuda.empty_cache()
    rows = []
    for index, job in enumerate(jobs, 1):
        check_guards(guards)
        logger.info("METRICS %d/12 %s seed%d | shared1000-bootstrap evaluator", index, job["condition"], job["seed"])
        report = evaluate_run(job["run"], "f1", seed=job["seed"], source_data_root=args.data_root,
                              target_data_root=args.data_root, bootstrap_resamples=1000)
        rows.append({"condition": job["condition"], "seed": job["seed"],
                     "eer": report["target"]["eer"], "auroc": report["target"]["auroc"],
                     "metrics_sha256": checkpoint_digest(job["run"] / "metrics.json")})
        replace_json(args.output / "progress.json", {"phase": "metrics", "completed": index, "total": 12})
    save_json(args.output / "summary.json", target_summary(rows))
    save_json(args.output / "complete.json", {"verified_evaluations": 12,
              "summary_sha256": checkpoint_digest(args.output / "summary.json"), "stop_after_evaluation": True})
    logger.info("COMPLETE twelve fixed F1 evaluations. No model selection, retraining, other folds or external evaluation.")


if __name__ == "__main__":
    main()
