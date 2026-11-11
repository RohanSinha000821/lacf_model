"""Prepare the cached-CLAP comparison; training requires matching source GPU evidence."""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

from audio_deepfake_detection.experiments.tf2_frozen_clap.model_data import (
    CachedCLAPCollator, CachedCLAPDataset, CachedCLAPTF2, cache_identity, load_clap_split, make_loader,
)
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from scripts.lacf.train_probes import seed_all
from scripts.token_fusion.train import check_guards, save_json
from scripts.token_fusion.train_adaptation import (
    adaptation_state, development_metrics, restore_adaptation, save_adaptation_checkpoint,
    score_audio, train_audio_epoch,
)

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "outputs/experiments/tf2_frozen_clap"
CACHE = ROOT / "outputs/experiments/lacf_diagnostics/frozen_features"
CHECKS = ("longest_ten_second_forward_backward", "finite_gradients", "direct_cached_clap_parity",
          "fresh_checkpoint_reload", "concurrent_memory_headroom")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def recipe(args, manifest, execution):
    files = [Path(__file__), ROOT / "src/audio_deepfake_detection/experiments/tf2_frozen_clap/model_data.py",
             *[ROOT / name for name in (
                 "src/audio_deepfake_detection/token_fusion/model.py",
                 "src/audio_deepfake_detection/token_fusion/adaptation.py",
                 "src/audio_deepfake_detection/token_fusion/adaptation_data.py",
                 "scripts/token_fusion/train_adaptation.py", "scripts/token_fusion/train.py",
                 "scripts/lacf/train_probes.py", "scripts/lacf/train_ablations.py",
                 "src/audio_deepfake_detection/lacf/probes.py", "src/audio_deepfake_detection/lacf/ablations.py",
                 "src/audio_deepfake_detection/lacf/diagnostic_cache.py",
                 "src/audio_deepfake_detection/lacf/data.py", "src/audio_deepfake_detection/lacf/model.py",
                 "src/audio_deepfake_detection/lacf/precision.py", "src/audio_deepfake_detection/lacf/training.py",
                 "src/audio_deepfake_detection/data.py", "src/audio_deepfake_detection/protocol.py",
                 "src/audio_deepfake_detection/metrics.py")]]
    return {"experiment": "tf2_frozen_clap", "fold": "f1", "seed": args.seed,
            "sources": list(FOLDS["f1"]["sources"]), "target_access": False,
            "initialization": "fresh pinned WavLM, zero-delta rank8 QV LoRA, fresh seeded TF2",
            "trainable_parameters": 5202657, "wavlm_lora_parameters": 294912, "fusion_parameters": 4907745,
            "clap_trainable_parameters": 0, "clap_encoder_loaded": False,
            "clap_input": "verified raw frozen CLAP projections; identical fixed physical view",
            "wavlm_input": "live native unpadded groups; centered plus two fixed train crops up to10s",
            "sampling": "equal source-domain and class mass; N replacement draws; uniform fixed view choice",
            "cache_payloads_used": ["records.json", "clap_projected.npy"],
            "objective": "detection BCE only", "rank": 8, "alpha": 16, "lora_dropout": 0,
            "optimizer": "AdamW", "lora_lr": 1e-4, "fusion_lr": 3e-4, "weight_decay": 1e-4,
            "scheduler": "constant", "maximum_epochs": 30, "patience": 5,
            "physical_batch": args.physical_batch, "effective_batch": 132,
            "accumulation_steps": 132//args.physical_batch, "workers": args.workers,
            "development_workers": 4, "prefetch_factor": 2,
            "precision": "BF16; FP32 normalization, fusion attention and loss", "tf32": False,
            "gpu_memory_budget_gib": float(args.gpu_memory_budget_gib),
            "selection": "strict equal-domain macro source-dev EER; ties earlier",
            "resume": False, "cache_root": str(args.cache_root.resolve()),
            "cache_completion_sha256": checkpoint_digest(args.cache_root/"complete.json"),
            "cache_config_sha256": manifest["config_sha256"],
            "cache_execution_sha256": manifest["accelerated_execution_sha256"],
            "pretrained": {key: execution["pretrained"][key] for key in ("wavlm_revision", "clap_revision")},
            "source_partitions": {f"{d}/{s}": {k: manifest["splits"][f"{d}/{s}"][k]
                for k in ("records", "canonical_ids_labels_sha256")}
                for d in FOLDS["f1"]["sources"] for s in ("train", "dev")},
            "data_root": str(args.data_root.resolve()), "torch": str(torch.__version__),
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "provenance": {"code_sha256": {str(p.relative_to(ROOT)): checkpoint_digest(p) for p in files}}}


def verify_validation(path, config):
    if path is None:
        raise ValueError("Training held: provide matching source-only GPU validation evidence")
    record = json.loads(Path(path).read_text())
    if (record.get("passed") is not True or record.get("target_access") is not False
            or record.get("recipe_sha256") != digest(config)
            or any(record.get("checks", {}).get(name) is not True for name in CHECKS)):
        raise ValueError("Source GPU validation is missing, failed or belongs to a different recipe")
    for name in ("train_examples_per_second", "source_dev_examples_per_second", "peak_allocated_bytes"):
        value = record.get("measurements", {}).get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Missing finite source memory/throughput measurements")
    if record["measurements"]["peak_allocated_bytes"] >= config["gpu_memory_budget_gib"]*1024**3:
        raise ValueError("Measured allocation leaves no configured memory headroom")
    return {"path": str(Path(path).resolve()), "sha256": checkpoint_digest(path)}


def build_model(execution, seed, device):
    revision = execution["pretrained"]["wavlm_revision"]
    with torch.random.fork_rng(devices=[]):
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", revision=revision,
                                         local_files_only=True, dtype=torch.float32)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus", revision=revision,
                                                            local_files_only=True)
    if processor.to_dict() != execution["processors"]["wavlm"]:
        raise ValueError("WavLM processor differs from cached physical-view policy")
    seed_all(seed)
    model = CachedCLAPTF2(wavlm).to(device)
    if model.configuration()["counts"] != {"wavlm_lora": 294912, "clap": 0, "fusion": 4907745}:
        raise ValueError("Cached CLAP experiment parameter inventory changed")
    return model, CachedCLAPCollator(processor)


def fit(model, fitting, development, collator, args, config, execution, logger):
    device = next(model.parameters()).device
    combined = CachedCLAPDataset([split for dataset in fitting for split in dataset.splits])
    train = make_loader(combined, collator, batch_size=args.physical_batch, workers=args.workers,
                        seed=args.seed, domains=fitting)
    loaders = {d: make_loader(dataset, collator, batch_size=args.physical_batch, workers=4, seed=args.seed)
               for d, dataset in development.items()}
    optimizer = model.optimizer()

    def score(current):
        scores, labels = {}, {}
        for domain, loader in loaders.items():
            logger.info("DEV %s | %d centered source recordings", domain, len(loader.dataset))
            scores[domain], labels[domain] = score_audio(current, loader, device, logger=logger)
        return scores, labels, development_metrics(scores, labels)

    config_hash = checkpoint_digest(args.output/"config.json")
    best, stale, best_epoch, best_scores = math.inf, 0, 0, None
    with (args.output/"history.jsonl").open("x") as history:
        for epoch in range(1, 31):
            check_guards(config["provenance"])
            losses = train_audio_epoch(model, train, optimizer, device, logger=logger, phase=f"epoch{epoch}")
            scores, labels, metrics = score(model)
            if metrics["macro_eer"] < best:
                best, stale, best_epoch = metrics["macro_eer"], 0, epoch
                best_scores = {d: value.clone() for d, value in scores.items()}
                save_adaptation_checkpoint(args.output/"best_adaptation.pt", {
                    "trainable_state": adaptation_state(model), "epoch": epoch, "source_dev": metrics,
                    "run_config_sha256": config_hash})
            else:
                stale += 1
            save_adaptation_checkpoint(args.output/"last_adaptation.pt", {
                "trainable_state": adaptation_state(model), "epoch": epoch, "source_dev": metrics,
                "optimizer_state_dict": optimizer.state_dict(), "run_config_sha256": config_hash,
                "best_epoch": best_epoch, "patience": stale})
            history.write(json.dumps({"epoch": epoch, "training": losses, "source_dev": metrics,
                                      "best_epoch": best_epoch, "patience": stale})+"\n")
            history.flush()
            logger.info("EPOCH %d | macro source EER %.4f%% | best%d patience%d/5", epoch, metrics["macro_eer"]*100,
                        best_epoch, stale)
            if stale >= 5:
                break
    checkpoint = torch.load(args.output/"best_adaptation.pt", map_location="cpu", weights_only=True)
    if checkpoint["run_config_sha256"] != config_hash or checkpoint_digest(args.output/"config.json") != config_hash:
        raise ValueError("Selected checkpoint/config identity changed")
    model.cpu()
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fresh, _ = build_model(execution, args.seed, device)
    restore_adaptation(fresh, checkpoint["trainable_state"])
    scores, labels, metrics = score(fresh)
    if metrics != checkpoint["source_dev"] or any(not torch.equal(scores[d], best_scores[d]) for d in development):
        raise ValueError("Fresh pinned-base source reload differs; no completion")
    hashes, calibration_inputs = {}, {}
    for domain, dataset in development.items():
        split = dataset.splits[0]
        path = args.output/f"source_dev_{domain}.csv"
        write_score_csv(path, zip(split.ids, itertools.repeat(domain), labels[domain].tolist(), scores[domain].tolist()))
        expected = dict(zip(split.ids, labels[domain].tolist()))
        exported_labels, exported = read_score_csv(path, domain, expected_labels=expected)
        order = sorted(range(len(split.ids)), key=split.ids.__getitem__)
        if not np.array_equal(exported, scores[domain].numpy()[order]):
            raise ValueError("Source score export differs")
        hashes[path.name] = checkpoint_digest(path)
        calibration_inputs[domain] = exported_labels, exported
    calibration = calibrate_source_scores(calibration_inputs)
    if not math.isclose(calibration["source_dev"]["macro_eer"], best, rel_tol=0, abs_tol=1e-14):
        raise ValueError("Source calibration differs from checkpoint selection")
    save_json(args.output/"source_calibration.json", calibration)
    save_json(args.output/"metrics.json", {"source_dev": metrics, "best_epoch": best_epoch, "final_epoch": epoch,
                                         "target_access": False})
    check_guards(config["provenance"])
    save_json(args.output/"complete.json", {"experiment": "tf2_frozen_clap", "fold": "f1", "seed": args.seed,
        "best_epoch": best_epoch, "final_epoch": epoch, "best_macro_source_dev_eer": best,
        "checkpoint_sha256": checkpoint_digest(args.output/"best_adaptation.pt"), "config_sha256": config_hash,
        "source_scores_sha256": hashes, "calibration_sha256": checkpoint_digest(args.output/"source_calibration.json"),
        "metrics_sha256": checkpoint_digest(args.output/"metrics.json"),
        "selected_checkpoint_reloaded_and_source_verified": True, "target_access": False})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "train"), default="prepare")
    parser.add_argument("--seed", type=int, choices=(1234,), default=1234)
    parser.add_argument("--physical-batch", type=int, default=22)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-memory-budget-gib", type=float, default=10)
    parser.add_argument("--cache-root", type=Path, default=CACHE)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--output", type=Path, default=FAMILY/"f1/seed_1234")
    parser.add_argument("--validation-record", type=Path)
    args = parser.parse_args()
    if (args.physical_batch < 1 or 132 % args.physical_batch or args.workers < 0
            or not math.isfinite(args.gpu_memory_budget_gib) or args.gpu_memory_budget_gib <= 0):
        parser.error("Invalid loading/memory settings; physical batch must divide132")
    if not args.output.resolve().is_relative_to(FAMILY.resolve()):
        parser.error("Store this experiment only under outputs/experiments/tf2_frozen_clap/")
    if args.mode == "train" and args.validation_record is None:
        parser.error("Training held until matching source-only GPU validation is supplied")
    return args


def main():
    args = parse_args()
    manifest, execution = cache_identity(args.cache_root)
    config = recipe(args, manifest, execution)
    if args.mode == "prepare":
        print(json.dumps({"recipe": config, "recipe_sha256": digest(config),
                          "training_started": False, "gpu_validation_performed": False}, indent=2))
        return
    validation = verify_validation(args.validation_record, config)
    if args.output.exists():
        raise FileExistsError("Preserve existing run; no overwrite or resume")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required")
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_budget_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
    bank = {(d, part): load_clap_split(args.cache_root, d, part, manifest["splits"][f"{d}/{part}"], args.data_root)
            for d in FOLDS["f1"]["sources"] for part in ("train", "dev")}
    fitting = [CachedCLAPDataset([bank[d, "train"]]) for d in FOLDS["f1"]["sources"]]
    development = {d: CachedCLAPDataset([bank[d, "dev"]]) for d in FOLDS["f1"]["sources"]}
    model, collator = build_model(execution, args.seed, "cuda")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output/"training.log")])
    logger = logging.getLogger("tf2_frozen_clap")
    config.update(validation_record=validation, model=model.configuration())
    save_json(args.output/"config.json", config)
    fit(model, fitting, development, collator, args, config, execution, logger)


if __name__ == "__main__":
    main()
