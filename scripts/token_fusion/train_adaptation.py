"""Source-only TF2 adaptation: explicit full-F1 or inner-source fitting."""

import argparse
import gc
import itertools
import json
import logging
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers import ClapModel, ClapProcessor, Wav2Vec2FeatureExtractor, WavLMModel

from audio_deepfake_detection.lacf.diagnostic_cache import all_layer_features
from audio_deepfake_detection.lacf.pretrained import DualViewCollator
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_split, score_probe
from audio_deepfake_detection.lacf.training import device_batch, report_progress, make_optimizer, train_epoch as train_features
from audio_deepfake_detection.metrics import calibrate_source_scores, compute_auroc, compute_eer, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, utterance_id
from audio_deepfake_detection.token_fusion.adaptation import AdaptationConfig, AdaptedTF2
from audio_deepfake_detection.token_fusion.adaptation_data import full_f1_sources, inner_sources, make_audio_loader, verify_source_membership
from audio_deepfake_detection.token_fusion.model import TokenFusion
from scripts.lacf.train_probes import seed_all
from scripts.token_fusion.train import ROOT, SEEDS, check_guards, save_json


CACHE = ROOT / "outputs/lacf_diagnostic_features_v1"
EFFECTIVE_BATCH = 132
CONDITIONS = ("frozen", "wavlm_lora", "wavlm_lora_clap")


def load_model(config, seed, device, *, frozen=False):
    identity = json.loads((CACHE / "execution_accelerated.json").read_text())
    p = identity["pretrained"]
    revisions = {"wavlm_revision": p["wavlm_revision"], "clap_revision": p["clap_revision"]}
    if frozen:
        seed_all(seed)
        return TokenFusion("TF2").to(device), None, revisions
    # Foundation construction must not alter the shared fresh fusion/RNG state.
    with torch.random.fork_rng(devices=[]):
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", revision=p["wavlm_revision"],
                                          local_files_only=True, dtype=torch.float32)
        clap = ClapModel.from_pretrained("laion/clap-htsat-unfused", revision=p["clap_revision"],
                                         local_files_only=True, dtype=torch.float32)
        wprocessor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus", revision=p["wavlm_revision"], local_files_only=True)
        cprocessor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", revision=p["clap_revision"], local_files_only=True)
    collator = DualViewCollator(wprocessor, cprocessor)
    if collator.configuration() != identity["processors"]:
        raise ValueError("Diagnostic input/processor policy changed")
    # TF2 never uses CLAP text; release those unused frozen weights, not audio.
    del clap.text_model, clap.text_projection
    seed_all(seed)
    model = AdaptedTF2(wavlm, clap, config).to(device)
    if model.configuration()["counts"]["wavlm_lora"] != 294912 or model.configuration()["counts"]["fusion"] != 4907745:
        raise ValueError("Rank8/all12 QV or TF2 parameter inventory changed")
    expected_clap = 14846528 if config.clap_final_stage else 0
    if model.configuration()["counts"]["clap"] != expected_clap:
        raise ValueError("Approved CLAP stage/projection parameter inventory changed")
    return model, collator, revisions


def adaptation_state(model):
    return model.trainable_state() if isinstance(model, AdaptedTF2) else {
        n: p.detach().cpu().clone() for n, p in model.state_dict().items()}


def restore_adaptation(model, state):
    if isinstance(model, AdaptedTF2):
        model.load_trainable_state(state)
    else:
        model.load_state_dict(state, strict=True)


def load_weights_recovery(path, config):
    """Recover selected weights only; never pretend to restore missing optimizer/RNG state."""
    config = json.loads(json.dumps(config))
    path = path.resolve()
    if path.name != "best_adaptation.pt":
        raise ValueError("Weights recovery requires the source-selected best_adaptation.pt")
    parent_path = path.parent / "config.json"
    parent = json.loads(parent_path.read_text())
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["run_config_sha256"] != checkpoint_digest(parent_path):
        raise ValueError("Recovery checkpoint is not bound to its parent config")
    fields = ("mode", "condition", "fold", "full_f1", "held_source", "seed", "model", "pretrained",
              "training_sources", "development_sources", "selection", "source_partitions", "cache_config_sha256",
              "physical_batch", "effective_batch", "accumulation_steps", "workers", "development_workers",
              "prefetch_factor", "precision", "optimizer", "scheduler", "maximum_epochs", "patience", "target_access")
    if any(parent.get(k) != config[k] for k in fields):
        raise ValueError("Recovery recipe, source identity or seed differs from the new run")
    if not config["full_f1"] or config["mode"] != "train" or config["condition"] == "frozen" or config["target_access"]:
        raise ValueError("Weights recovery is restricted to source-only full-F1 adaptation")
    # Only this entrypoint changes for recovery; all model/data/base implementations must agree.
    entrypoint = str(Path(__file__).resolve().relative_to(ROOT))
    guards = {**parent["provenance"]["code_sha256"], **parent["provenance"]["preserved_job_guards"]}
    for name, digest in guards.items():
        if name != entrypoint and checkpoint_digest(ROOT / name) != digest:
            raise ValueError(f"Recovery parent implementation changed: {name}")
    epoch = state["epoch"]
    metrics = state["source_dev"]
    if (type(epoch) is not int or not 1 <= epoch < config["maximum_epochs"]
            or set(metrics["by_domain"]) != set(config["development_sources"])):
        raise ValueError("Invalid recovery epoch or development domains")
    for key in ("eer", "auroc"):
        values = [m[key] for m in metrics["by_domain"].values()]
        if (any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                or not math.isclose(statistics.mean(values), metrics[f"macro_{key}"], abs_tol=1e-14, rel_tol=0)):
            raise ValueError("Invalid recovery source metrics")
    return state, {"kind": "selected_weights_only_after_host_shutdown", "checkpoint": str(path),
                   "checkpoint_sha256": checkpoint_digest(path), "parent_config_sha256": checkpoint_digest(parent_path),
                   "parent_epoch": epoch, "first_epoch": epoch+1, "optimizer_restored": False,
                   "rng_or_sampler_restored": False, "patience_restarted_at": 0,
                   "exact_training_resume": False, "maximum_logical_epoch": config["maximum_epochs"]}


def save_adaptation_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("Interrupted checkpoint preserved")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_audio_epoch(model, loader, optimizer, device, *, effective_batch=EFFECTIVE_BATCH, logger=None, phase="train", max_batches=None):
    """Sample-weighted accumulation, including a short last microbatch/group."""
    physical = loader.batch_size
    if effective_batch < physical or effective_batch % physical:
        raise ValueError("Physical batch must divide the approved effective batch")
    total = len(loader.dataset) if max_batches is None else min(len(loader.dataset), max_batches*physical)
    model.train()
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    count = updates = group_seen = 0
    loss_sum = torch.zeros((), device=device)
    started = last_log = time.monotonic()
    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for index, batch in enumerate(batches):
        if count >= total:
            break
        if group_seen == 0:
            optimizer.zero_grad(set_to_none=True)
            group_count = min(effective_batch, total-count)
        batch = device_batch(batch, device)
        size = len(batch["labels"])
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            losses = model.objective(model(batch), batch["labels"])
        if losses["total"].dtype != torch.float32 or not torch.isfinite(losses["total"]):
            raise ValueError("Nonfinite loss or sensitive loss not FP32")
        (losses["total"] * (size/group_count)).backward()
        group_seen += size
        count += size
        loss_sum += losses["total"].detach()*size
        if group_seen == group_count:
            if any(p.grad is None for p in parameters) or not torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all():
                raise ValueError("Missing/nonfinite enabled gradient; no optimizer update")
            optimizer.step()
            updates += 1
            group_seen = 0
        if time.monotonic()-last_log >= 30 or count == total:
            report_progress(logger, phase, count, total, started, loss=float(loss_sum/count))
            last_log = time.monotonic()
    if count != total or group_seen:
        raise ValueError("Incomplete accumulation group or source epoch")
    return {"total": float(loss_sum/count), "samples": count, "updates": updates,
            "seconds": time.monotonic()-started}


@torch.inference_mode()
def score_audio(model, loader, device, *, logger=None, max_batches=None):
    model.eval()
    count = len(loader.dataset) if max_batches is None else min(len(loader.dataset), max_batches*loader.batch_size)
    scores = torch.empty(count, dtype=torch.float32, device=device)
    labels = torch.empty(count, dtype=torch.long)
    offset = 0
    started = last_log = time.monotonic()
    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for batch in batches:
        if offset >= count:
            break
        size = len(batch["labels"])
        labels[offset:offset+size] = batch["labels"]
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            values = model(device_batch(batch, device))["logit"].float()
        scores[offset:offset+size] = values
        offset += size
        if time.monotonic()-last_log >= 30 or offset == count:
            report_progress(logger, "source validation", offset, count, started)
            last_log = time.monotonic()
    scores = scores.cpu()
    if offset != count or not torch.isfinite(scores).all():
        raise ValueError("Incomplete/nonfinite source scores")
    return scores, labels


@torch.inference_mode()
def zero_lora_reference(model, batch):
    """Original extraction helpers and TF2 must agree at the zero-delta start."""
    model.eval()
    enabled = [p for module in (model.wavlm, model.clap) for p in module.parameters() if p.requires_grad]
    for p in enabled:
        p.requires_grad_(False)
    try:
        reference = all_layer_features(SimpleNamespace(wavlm=model.wavlm, clap=model.clap), batch, "cuda")
    finally:
        for p in enabled:
            p.requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model.fusion({"wavlm": torch.from_numpy(reference["wavlm_layers"]).cuda(),
                                 "clap": torch.from_numpy(reference["clap_projected"]).cuda()})["logit"]
        actual = model(batch)["logit"]
    if not torch.equal(expected, actual):
        raise ValueError("Zero-LoRA direct source scoring differs from the original extraction/TF2 path")


def probe(model, collator, datasets, args, logger):
    fitting, train, dev = datasets
    initial = model.trainable_state()
    # Use actual canonical long source audio, repeated only as a memory stress.
    metadata = json.loads((CACHE / "asv5/train/chunk_00000/records.json").read_text())
    domain = next(d for d in fitting if d.identifiers[0].startswith("asv5/"))
    row = next(r for r in metadata if r["frames"] >= 10*r["rate"])
    if domain.identifiers[row["index"]] != row["id"] or domain.records[row["index"]][1] != row["label"]:
        raise ValueError("Memory-stress source membership differs from the preserved cache")
    example = domain[row["index"], 0]
    batch = device_batch(collator([example]*args.physical_batch), "cuda")
    if batch["wavlm"]["input_values"].shape[-1] != 160000:
        raise ValueError("Expected actual common10s input for the memory stress")
    zero_lora_reference(model, device_batch(collator([example]), "cuda"))
    optimizer = model.optimizer()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    stress_seconds = []
    for step in range(11):
        torch.cuda.synchronize()
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.objective(model(batch), batch["labels"])["total"]
        if loss.dtype != torch.float32 or not torch.isfinite(loss):
            raise ValueError("Invalid sensitive stress loss")
        loss.backward()
        parameters = [p for p in model.parameters() if p.requires_grad]
        if any(p.grad is None for p in parameters) or not torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all():
            raise ValueError("Enabled parameters have missing/nonfinite gradients")
        # A starts with zero gradient because B starts at zero; both must learn later.
        if step > 0 and not torch.stack([p.grad.abs().sum() for p in model.wavlm.parameters() if p.requires_grad]).gt(0).all():
            raise ValueError("LoRA bypass detected: an enabled low-rank matrix has zero gradient after update")
        optimizer.step()
        torch.cuda.synchronize()
        if step >= 3:
            stress_seconds.append(time.monotonic()-tick)
    peak = torch.cuda.max_memory_allocated()
    checkpoint = model.trainable_state()
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(batch)["logit"].cpu()
    model.load_trainable_state(initial)
    model.load_trainable_state(checkpoint)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(batch)["logit"].cpu()
    if not torch.equal(expected, actual):
        raise ValueError("Sparse trainable-state reload differs")
    reload_inputs = device_batch(batch, "cpu")
    results = []
    del batch, optimizer
    for workers in (4, 8):
        model.load_trainable_state(initial)
        optimizer = model.optimizer()
        train_loader = make_audio_loader(train, collator, batch_size=args.physical_batch, workers=workers,
                                        seed=args.seed, domains=fitting)
        tick = time.monotonic()
        stats = train_audio_epoch(model, train_loader, optimizer, "cuda", effective_batch=EFFECTIVE_BATCH,
                                  logger=logger, phase=f"timing workers{workers}",
                                  max_batches=4*EFFECTIVE_BATCH//args.physical_batch)
        torch.cuda.synchronize()
        stats.update(workers=workers, measured_seconds_including_startup=time.monotonic()-tick)
        stats["examples_per_second_including_startup"] = stats["samples"]/stats["measured_seconds_including_startup"]
        results.append(stats)
        del train_loader, optimizer
        gc.collect()
    val_loader = make_audio_loader(dev, collator, batch_size=args.physical_batch, workers=4, seed=args.seed)
    tick = time.monotonic()
    scores, labels = score_audio(model, val_loader, "cuda", logger=logger,
                                 max_batches=max(12, math.ceil(EFFECTIVE_BATCH/args.physical_batch)))
    seconds = time.monotonic()-tick
    val_rate = len(scores)/seconds
    del val_loader
    gc.collect()
    model.cpu()
    torch.cuda.empty_cache()
    fresh, _, _ = load_model(model.config, args.seed, "cuda")
    fresh.load_trainable_state(checkpoint)
    fresh.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reloaded = fresh(device_batch(reload_inputs, "cuda"))["logit"].cpu()
    if not torch.equal(expected, reloaded):
        raise ValueError("Fresh pinned-base sparse reload differs on source inputs")
    del fresh
    gc.collect()
    torch.cuda.empty_cache()
    best = max(results, key=lambda r: r["examples_per_second_including_startup"])
    per_epoch = len(train)/best["examples_per_second_including_startup"] + len(dev)/val_rate
    report = {"passed": True, "purpose": "bounded source-only execution/timing; discard all pilot weights",
              "condition": args.condition, "held_source": args.held_domain, "seed": args.seed,
              "physical_batch": args.physical_batch, "effective_batch_for_final_fits": EFFECTIVE_BATCH,
              "model": model.configuration(), "zero_lora_original_source_score_bit_exact": True,
              "all_enabled_gradients_finite": True, "all_lora_a_b_gradients_nonzero_after_first_update": True,
              "sparse_reload_bit_exact": True, "fresh_pinned_base_reload_bit_exact": True,
              "loader_timing_effective_batch": EFFECTIVE_BATCH, "loader_timing_optimizer_updates": 4,
              "stress_input_seconds": 10, "stress_measured_updates": 8,
              "stress_seconds_per_update": sum(stress_seconds)/len(stress_seconds),
              "stress_peak_allocated_bytes": peak, "worker_measurements": results,
              "source_dev_records_measured": len(scores), "source_dev_seconds": seconds,
              "source_dev_examples_per_second": val_rate, "source_train_records": len(train), "source_dev_records": len(dev),
              "rough_epoch_hours": per_epoch/3600, "rough_30_epoch_hours": per_epoch*30/3600,
              "timing_limitations": "small source sample and process/worker startup; changing durations/shared SSL contention; not a promised total or recipe selection",
              "long_queue_started": False, "target_access": False}
    save_json(args.output / "probe.json", report)
    logger.info("PROBE PASSED | condition=%s batch%d | %.2fGiB peak | roughepoch %.2fh (short sample)",
                args.condition, args.physical_batch, peak/1024**3, per_epoch/3600)
    return report


def development_metrics(scores, labels):
    by_domain = {d: {"eer": compute_eer(labels[d].numpy(), values.numpy()),
                     "auroc": compute_auroc(labels[d].numpy(), values.numpy())}
                 for d, values in scores.items()}
    return {"by_domain": by_domain,
            "macro_eer": statistics.mean(m["eer"] for m in by_domain.values()),
            "macro_auroc": statistics.mean(m["auroc"] for m in by_domain.values())}


def fit(model, collator, datasets, args, provenance, logger, *, recovery=None):
    fitting, train, development = datasets
    dev = development if args.full_f1 else {args.held_domain: development}
    device = next(model.parameters()).device
    frozen = args.condition == "frozen"
    if recovery is not None:
        restore_adaptation(model, recovery["trainable_state"])
    if frozen:
        marker = json.loads((CACHE / "complete.json").read_text())
        bank = {}
        for dataset in [*fitting, *dev.values()]:
            domain, part, _ = dataset.identifiers[0].split("/", 2)
            bank[domain, part] = load_split(CACHE, domain, part, marker["splits"][f"{domain}/{part}"], args.data_root, logger=logger)
        train_loader = FeatureBatches([bank[d.identifiers[0].split("/", 1)[0], "train"] for d in fitting],
                                     "concat", device, training=True, seed=args.seed)
        val_loaders = {d: FeatureBatches([bank[d, "dev"]], "concat", device, training=False, seed=args.seed) for d in dev}
        optimizer = make_optimizer(model)
    else:
        optimizer = model.optimizer()
        train_loader = make_audio_loader(train, collator, batch_size=args.physical_batch, workers=args.workers,
                                        seed=args.seed, domains=fitting)
        val_loaders = {d: make_audio_loader(dataset, collator, batch_size=args.physical_batch, workers=4, seed=args.seed)
                       for d, dataset in dev.items()}
    def score(current):
        values, labels = {}, {}
        for domain, loader in val_loaders.items():
            logger.info("DEV %s | %d centered source recordings", domain, len(dev[domain]))
            if frozen:
                values[domain], _ = score_probe(current, loader)
                labels[domain] = bank[domain, "dev"].labels
            else:
                values[domain], labels[domain] = score_audio(current, loader, device, logger=logger)
        return values, labels, development_metrics(values, labels)
    best, stale, first_epoch, best_epoch, best_scores = float("inf"), 0, 1, 0, None
    config_digest = checkpoint_digest(args.output / "config.json")
    if recovery is not None:
        best, best_epoch = recovery["source_dev"]["macro_eer"], recovery["epoch"]
        first_epoch = best_epoch + 1
        save_adaptation_checkpoint(args.output / "best_adaptation.pt",
                                   {**recovery, "run_config_sha256": config_digest})
        logger.info("RECOVERY | epoch%d selected weights; fresh optimizer/RNG/sampler; next epoch%d; "
                    "preserving parent best EER%.4f%%, max logical epoch30, patience reset0/5; not exact resume",
                    best_epoch, first_epoch, best*100)
    with (args.output / "history.jsonl").open("x") as history:
        for epoch in range(first_epoch, 31):
            check_guards(provenance)
            if frozen:
                loss, updates = train_features(model, train_loader, optimizer, device, accumulation_steps=1, logger=logger, phase=f"epoch{epoch}")
                loss.update(samples=train_loader.count, updates=updates)
            else:
                loss = train_audio_epoch(model, train_loader, optimizer, device, logger=logger, phase=f"epoch{epoch}")
            scores, labels, metrics = score(model)
            if metrics["macro_eer"] < best:
                best, best_epoch, stale = metrics["macro_eer"], epoch, 0
                best_scores = {d: value.clone() for d, value in scores.items()}
                save_adaptation_checkpoint(args.output / "best_adaptation.pt",
                                           {"trainable_state": adaptation_state(model), "epoch": epoch,
                                            "source_dev": metrics, "run_config_sha256": config_digest})
            else:
                stale += 1
            # Retain the latest completed epoch too; best-only saving lost epochs5-7 on shutdown.
            # This does not implement exact resume (worker/RNG state is not restored).
            save_adaptation_checkpoint(args.output / "last_adaptation.pt",
                                       {"trainable_state": adaptation_state(model), "epoch": epoch,
                                        "source_dev": metrics, "run_config_sha256": config_digest,
                                        "optimizer_state_dict": optimizer.state_dict(),
                                        "best_epoch": best_epoch, "patience": stale})
            history.write(json.dumps({"epoch": epoch, "training": loss, "source_dev": metrics, "best_epoch": best_epoch, "patience": stale})+"\n")
            history.flush()
            logger.info("EPOCH %d | macro source EER%.4f%% AUROC%.6f | by_domain=%s | best%d stale%d/5",
                        epoch, metrics["macro_eer"]*100, metrics["macro_auroc"], metrics["by_domain"], best_epoch, stale)
            if stale >= 5:
                break
    state = torch.load(args.output / "best_adaptation.pt", map_location="cpu", weights_only=True)
    if state["run_config_sha256"] != checkpoint_digest(args.output / "config.json"):
        raise ValueError("Selected checkpoint recipe binding changed")
    if best_scores is None:
        # The inherited best may win every tie; verify its full scores before fresh reconstruction.
        restore_adaptation(model, state["trainable_state"])
        best_scores, _, inherited_metrics = score(model)
        if inherited_metrics != state["source_dev"]:
            raise ValueError("Inherited checkpoint source metrics differ on full reload")
    model.cpu()
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fresh, _, _ = load_model(AdaptationConfig(clap_final_stage=args.condition == "wavlm_lora_clap"), args.seed, device, frozen=frozen)
    restore_adaptation(fresh, state["trainable_state"])
    scores, labels, metrics = score(fresh)
    if metrics != state["source_dev"] or any(not torch.equal(scores[d], best_scores[d]) for d in dev):
        raise ValueError("Fresh pinned-base selected-checkpoint source reload differs")
    hashes, calibration_inputs = {}, {}
    for domain, dataset in dev.items():
        path = args.output / f"source_dev_{domain}.csv"
        ids = [utterance_id(p, domain, args.data_root) for p, _ in dataset.records]
        write_score_csv(path, zip(ids, itertools.repeat(domain), labels[domain].tolist(), scores[domain].tolist()))
        exported_labels, exported = read_score_csv(path, domain, expected_labels=dict(zip(ids, labels[domain].tolist())))
        order = sorted(range(len(ids)), key=ids.__getitem__)
        if not np.array_equal(exported, scores[domain].numpy()[order]):
            raise ValueError("Canonical source export differs from reloaded scores")
        hashes[path.name] = checkpoint_digest(path)
        calibration_inputs[domain] = exported_labels, exported
    calibration = calibrate_source_scores(calibration_inputs)
    if not math.isclose(calibration["source_dev"]["macro_eer"], best, abs_tol=1e-14, rel_tol=0):
        raise ValueError("Source calibration disagrees with checkpoint selection")
    save_json(args.output / "source_calibration.json", calibration)
    scope = "full F1 sources; equal-domain macro development selection" if args.full_f1 else "inner-source development selection"
    save_json(args.output / "metrics.json", {"source_dev": metrics, "held_source": args.held_domain,
              "sources": list(dev), "best_epoch": best_epoch, "final_epoch": epoch, "scope": scope})
    save_json(args.output / "complete.json", {"best_epoch": best_epoch, "final_epoch": epoch, "fold": "f1",
              "seed": args.seed, "held_source": args.held_domain, "condition": args.condition,
              "full_f1": args.full_f1, "sources": list(dev), "best_macro_source_dev_eer": best,
              "config_sha256": checkpoint_digest(args.output / "config.json"),
              "checkpoint_sha256": checkpoint_digest(args.output / "best_adaptation.pt"),
              "source_scores_sha256": hashes,
              "calibration_sha256": checkpoint_digest(args.output / "source_calibration.json"),
              "metrics_sha256": checkpoint_digest(args.output / "metrics.json"),
              "selected_checkpoint_reloaded_and_source_verified": True, "target_access": False})
    logger.info("COMPLETE | %s | all %d source dev sets reload-verified; no target evaluation", scope, len(dev))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--held-domain", choices=FOLDS["f1"]["sources"])
    parser.add_argument("--full-f1", action="store_true", help="Train on all three F1 sources; select by macro source-dev EER")
    parser.add_argument("--seed", choices=SEEDS, type=int, default=1234)
    parser.add_argument("--physical-batch", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-from", type=Path,
                        help="Explicit selected-weights recovery into a NEW full-F1 run; fresh optimizer, not exact resume")
    args = parser.parse_args()
    if args.full_f1 and (args.held_domain is not None or args.mode != "train"):
        parser.error("Full F1 is train mode with all three sources; do not specify a held source")
    if not args.full_f1 and args.held_domain is None:
        parser.error("Inner-source mode requires --held-domain; use --full-f1 for all three F1 sources")
    if args.physical_batch < 1 or EFFECTIVE_BATCH % args.physical_batch:
        parser.error("Physical batch must divide132")
    if args.mode == "probe" and args.held_domain == "asv5":
        parser.error("The canonical ASV5-train memory example must be a fitting source; use another inner hold for probes")
    if args.condition == "frozen" and (args.mode == "probe" or args.physical_batch != 132):
        parser.error("Frozen references use cached features/batch132 in train mode; no encoder probe needed")
    if args.recover_from is not None and (not args.full_f1 or args.mode != "train" or args.condition == "frozen"):
        parser.error("Weights recovery requires full-F1 adaptation training")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 is required; no precision fallback")
    torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(),logging.FileHandler(args.output/'run.log')])
    logger = logging.getLogger("tf2_adaptation")
    guards = json.loads((ROOT/'outputs/ssl_gating_20s/f2/1234/config.json').read_text())["source_sha256"]
    paths = [Path(__file__).resolve(), ROOT/'src/audio_deepfake_detection/token_fusion/adaptation.py',
             ROOT/'src/audio_deepfake_detection/token_fusion/adaptation_data.py', ROOT/'src/audio_deepfake_detection/token_fusion/model.py',
             ROOT/'src/audio_deepfake_detection/lacf/pretrained.py', ROOT/'src/audio_deepfake_detection/lacf/diagnostic_cache.py',
             ROOT/'src/audio_deepfake_detection/lacf/model.py', ROOT/'src/audio_deepfake_detection/lacf/precision.py',
             ROOT/'src/audio_deepfake_detection/lacf/data.py', ROOT/'src/audio_deepfake_detection/lacf/probes.py',
             ROOT/'src/audio_deepfake_detection/lacf/training.py', ROOT/'scripts/lacf/train_probes.py', ROOT/'scripts/token_fusion/train.py']
    provenance = {"code_sha256": {str(p.relative_to(ROOT)): checkpoint_digest(p) for p in paths}, "preserved_job_guards": guards}
    check_guards(provenance)
    logger.info("START %s %s | full_f1=%s held-source=%s; source-only, no target access",
                args.mode, args.condition, args.full_f1, args.held_domain)
    config = AdaptationConfig(clap_final_stage=args.condition == "wavlm_lora_clap")
    marker = json.loads((CACHE / "complete.json").read_text())
    if (checkpoint_digest(CACHE / "config.json") != marker["config_sha256"]
            or checkpoint_digest(CACHE / "execution_accelerated.json") != marker["accelerated_execution_sha256"]):
        raise ValueError("Preserved source-cache scientific identity changed")
    model, collator, revisions = load_model(config, args.seed, "cuda", frozen=args.condition == "frozen")
    data = full_f1_sources(args.data_root) if args.full_f1 else inner_sources(args.data_root, args.held_domain)
    development = data[2] if args.full_f1 else {args.held_domain: data[2]}
    membership = verify_source_membership([*data[0], *development.values()], marker)
    run_config = {"mode": args.mode, "condition": args.condition, "held_source": args.held_domain,
              "fold": "f1", "full_f1": args.full_f1,
              "scope": "full F1 source train/dev" if args.full_f1 else "inner-source train/dev",
              "training_sources": [d.identifiers[0].split("/", 1)[0] for d in data[0]],
              "development_sources": list(development),
              "selection": "strict equal-domain macro source-dev EER; ties retain earlier",
              "seed": args.seed, "model": model.configuration(), "pretrained": revisions, "provenance": provenance,
              "torch": str(torch.__version__), "transformers": transformers.__version__, "cuda": torch.version.cuda,
              "effective_batch": EFFECTIVE_BATCH, "physical_batch": args.physical_batch,
              "accumulation_steps": EFFECTIVE_BATCH//args.physical_batch, "workers": args.workers,
              "development_workers": 4, "prefetch_factor": 2, "source_partitions": membership,
              "cache_config_sha256": marker["config_sha256"],
              "precision": {"autocast": "bfloat16", "normalization_tf2_attention_loss": "float32", "tf32": False},
              "optimizer": {"name": "AdamW", "fusion_lr": 3e-4, "weight_decay": 1e-4,
                            "lora_lr": None if args.condition == "frozen" else 1e-4,
                            "clap_lr": 1e-5 if config.clap_final_stage else None},
              "initialization": "fresh TF2 on cached frozen features" if args.condition == "frozen" else
                                "fresh official base plus fresh TF2 and zero-delta LoRA; no target-selected checkpoint",
              "maximum_epochs": 30, "patience": 5, "scheduler": "constant", "resume": False, "target_access": False,
              "gpu_allocator_budget_gib": 10, "automatic_long_queue": False}
    recovery = None
    if args.recover_from is not None:
        recovery, run_config["recovery"] = load_weights_recovery(args.recover_from, run_config)
        run_config["initialization"] = "parent source-selected adaptation weights on pinned bases; fresh optimizer/RNG/sampler"
    save_json(args.output / "config.json", run_config)
    logger.info("DATA | train sources=%s records=%d | dev=%s | full_f1=%s",
                [d.identifiers[0].split("/", 1)[0] for d in data[0]], len(data[1]),
                {d: len(v) for d, v in development.items()}, args.full_f1)
    logger.info("PARAMETERS %s", model.configuration().get("counts", {"fusion": model.configuration()["trainable_parameters"]}))
    if args.mode == "probe":
        probe(model, collator, data, args, logger)
    else:
        fit(model, collator, data, args, provenance, logger, recovery=recovery)
    check_guards(provenance)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("RUN FAILED; partial outputs preserved, no automatic retry")
        raise
