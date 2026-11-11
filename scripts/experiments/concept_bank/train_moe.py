"""Two fresh cached full-F1 MoE fits; fixed 4+4 banks, seed1234, source access only."""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.experiments.concept_bank.model import (
    COMMON, FINAL_CONCEPT, ConceptBankMoE, bank_definition,
)
from audio_deepfake_detection.experiments.layer_fusion.diagnostics import source_diagnostics
from audio_deepfake_detection.lacf.model import CONCEPTS, TEMPLATES, text_prototypes
from audio_deepfake_detection.lacf.probes import BATCH_SIZE, FeatureBatches
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import checkpoint_digest, save_score_manifest, verify_source_macro
from scripts.experiments.layer_fusion.train_full_f1 import checkpoint, preflight
from scripts.lacf.train_probes import save_json, seed_all
from scripts.token_fusion.train import SOURCES, assert_bank, check_guards, load_f1_bank, score_sources


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/experiments/concept_bank/moe_f1"
REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
STAGE = "lacf_layer_moe_balanced_concepts"
SCOPE = "full F1 source train/dev; source-selected exploratory comparison, not inner-held-source validation"


def tensor_digest(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def check_inputs(provenance):
    check_guards(provenance)
    for path, expected in provenance.get("input_sha256", {}).items():
        if checkpoint_digest(Path(path)) != expected:
            raise ValueError("Fixed concept input changed")


def encode_banks(directory):
    """Local-only CPU text encoder; common concepts are encoded once and shared exactly."""
    from transformers import AutoTokenizer, ClapTextModelWithProjection
    snapshot = Path.home() / ".cache/huggingface/hub/models--laion--clap-htsat-unfused/snapshots" / REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model, info = ClapTextModelWithProjection.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.float32, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError("Pinned CLAP text weights did not load completely")
    if any(not k.startswith(("audio_model.", "audio_projection.", "logit_scale")) for k in info["unexpected_keys"]):
        raise ValueError("Unexpected text checkpoint keys")
    model.requires_grad_(False).eval()
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("Text prototype encoding must remain on CPU")

    def encode(concepts):
        prompts = [t.format(concept=c) for c in concepts for t in TEMPLATES]
        features = []
        for offset in range(0, len(prompts), 6):
            tokens = tokenizer(prompts[offset:offset+6], padding=True, truncation=False, return_tensors="pt")
            if tokens["input_ids"].shape[-1] > model.config.max_position_embeddings - 2:
                raise ValueError("Text prompt would be truncated")
            with torch.inference_mode():
                features.append(model(**tokens).text_embeds.float())
        result = torch.cat(features)
        if not torch.isfinite(result).all() or (result.norm(dim=-1) == 0).any():
            raise ValueError("Nonfinite or zero text embeddings")
        return result.reshape(len(concepts), 3, 512)

    original = text_prototypes(encode(CONCEPTS).reshape(24, 512))
    saved = json.loads((ROOT / "outputs/experiments/lacf_diagnostics/step1/static.json").read_text())
    difference = float(np.abs((original @ original.T).numpy() - np.asarray(saved["prototype_cosines"]["f1"])).max())
    if difference >= 2e-5:
        raise ValueError("Pinned original text geometry did not reproduce")
    concepts = [*COMMON, *FINAL_CONCEPT.values()]
    features = encode(concepts)
    values = {}
    directory.mkdir(parents=True, exist_ok=False)
    for index, name in enumerate(FINAL_CONCEPT):
        rows = [*range(len(COMMON)), len(COMMON) + index]
        values[name] = text_prototypes(features[rows].reshape(24, 512))
        np.save(directory / f"{name}.npy", values[name].numpy(), allow_pickle=False)
        save_json(directory / f"{name}.json", bank_definition(name))
    if not torch.equal(values["vocoder"][:7], values["neural_synthesis"][:7]):
        raise ValueError("Shared seven concept prototypes differ")
    if torch.equal(values["vocoder"][7], values["neural_synthesis"][7]):
        raise ValueError("The two final-concept prototypes unexpectedly match")
    save_json(directory / "encoding.json", {"device": "cpu", "precision": "float32", "revision": REVISION,
        "checkpoint_sha256": checkpoint_digest(snapshot / "pytorch_model.bin"),
        "shared_seven_prototypes_bit_exact": True, "original_geometry_max_abs_difference": difference,
        "no_audio_encoder_loaded": True, "trained_text_parameters": 0})
    del model
    gc.collect()
    return values


def fit(bank, prototypes, name, output, provenance, device, logger, *, max_epochs=30, patience=5):
    assert_bank(bank)
    check_inputs(provenance)
    definition = bank_definition(name)
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "training.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    try:
        seed_all(1234)
        model = ConceptBankMoE(name, prototypes).to(device)
        optimizer = make_optimizer(model)
        train = FeatureBatches([bank[d, "train"] for d in SOURCES], "concat", device, training=True, seed=1234)
        config = {"stage": STAGE, "model_name": "LACF Layer-MoE", "concept_bank": definition,
            "fold": "f1", "seed": 1234, "seed_plan": [1234], "sources": list(SOURCES),
            "excluded_target": "speechfake", "target_access": False, "scope": SCOPE,
            "model": model.configuration(), "prototype_bytes_sha256": tensor_digest(prototypes),
            "batch_size": BATCH_SIZE, "accumulation": 1, "optimizer": "AdamW", "lr": 3e-4,
            "weight_decay": 1e-4, "scheduler": "constant", "max_epochs": max_epochs, "patience": patience,
            "precision": "BF16 mixed; sensitive FP32", "sampling": "equal domain/class mass; replacement",
            "training_views": "centered plus two fixed random up-to10s views, uniform per draw",
            "development_views": "centered up-to10s; native unpadded frozen feature extraction",
            "selection": "strict equal-domain macro source-dev EER; ties earlier", "resume": False,
            "foundation_encoders_loaded": False, "feature_workers": 0, "train_records": train.count,
            "dev_records": {d: len(bank[d, "dev"].labels) for d in SOURCES}, "provenance": provenance,
            "comparison": "two banks differ only in final spoof phrase; other seven prototypes and fresh trainable initialization match",
            "limitations": "after prior target exposure; no target-guided tuning; full-F1 source dev is not an inner-held-domain result"}
        if config["model"]["trainable_parameters"] != 967338:
            raise ValueError("Unchanged C4 parameter inventory required")
        save_json(output / "config.json", config)
        digest = checkpoint_digest(output / "config.json")
        logger.info("START %s | full F1 seed1234 | train%d dev%s | 967338 trainables; cached encoders; detection BCE only", name, train.count, config["dev_records"])
        best, best_epoch, stale, selected_scores = float("inf"), 0, 0, None
        with (output / "history.jsonl").open("x") as history:
            for epoch in range(1, max_epochs + 1):
                check_inputs(provenance)
                tick = time.monotonic()
                losses, updates = train_epoch(model, train, optimizer, device, accumulation_steps=1,
                    logger=logger, phase=f"{name} epoch{epoch}")
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
                logger.info("EPOCH %s %d | macro source EER %.4f%% AUROC %.6f | best%d patience%d/%d | %.1fs",
                    name, epoch, best*100 if stale == 0 else metrics["macro_eer"]*100,
                    metrics["macro_auroc"], best_epoch, stale, patience, row["seconds"])
                if stale >= patience:
                    break
        state = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        if state["config_sha256"] != digest or checkpoint_digest(output / "config.json") != digest:
            raise ValueError("Selected checkpoint/config binding changed")
        del model, optimizer
        gc.collect()
        reloaded = ConceptBankMoE(name, prototypes).to(device)
        reloaded.load_state_dict(state["model_state_dict"], strict=True)
        if not torch.equal(reloaded.backend.prototypes.cpu(), prototypes.cpu()):
            raise ValueError("Selected checkpoint concept prototypes differ")
        scores, metrics = score_sources(reloaded, bank, device, logger)
        if metrics != state["source_dev"] or any(not torch.equal(scores[d], selected_scores[d]) for d in SOURCES):
            raise ValueError("Fresh selected-checkpoint full source reload differs")
        inputs, hashes = {}, {}
        for domain in SOURCES:
            data = bank[domain, "dev"]; path = output / f"source_dev_{domain}.csv"
            write_score_csv(path, zip(data.ids, itertools.repeat(domain), data.labels.tolist(), scores[domain].tolist()))
            labels, exported = read_score_csv(path, domain, expected_labels=dict(zip(data.ids, data.labels.tolist())))
            order = sorted(range(len(data.ids)), key=data.ids.__getitem__)
            if not np.array_equal(exported, scores[domain].numpy()[order]):
                raise ValueError("Canonical source export changed scores")
            inputs[domain] = labels, exported; hashes[path.name] = checkpoint_digest(path)
        calibration = calibrate_source_scores(inputs)
        verify_source_macro(calibration["source_dev"]["macro_eer"], best)
        save_json(output / "source_calibration.json", calibration)
        save_json(output / "routing.json", source_diagnostics(reloaded, bank, "f1", 1234, device))
        save_json(output / "metrics.json", {"model_name": "LACF Layer-MoE", "concept_bank": name,
            "scope": SCOPE, "source_dev": metrics, "best_epoch": best_epoch, "final_epoch": epoch,
            "target_access": False})
        complete = {"stage": STAGE, "concept_bank": name, "fold": "f1", "seed": 1234,
            "sources": list(SOURCES), "best_epoch": best_epoch, "final_epoch": epoch,
            "best_macro_source_dev_eer": best, "best_checkpoint_sha256": checkpoint_digest(output / "best.pt"),
            "config_sha256": digest, "metrics_sha256": checkpoint_digest(output / "metrics.json"),
            "source_scores_sha256": hashes, "calibration_sha256": checkpoint_digest(output / "source_calibration.json"),
            "routing_sha256": checkpoint_digest(output / "routing.json"),
            "prototype_bytes_sha256": tensor_digest(prototypes),
            "selected_checkpoint_reloaded_and_source_verified": True, "target_access": False}
        check_inputs(provenance)
        save_json(output / "training_complete.json", complete)
        save_score_manifest(output, {"stage": STAGE, "fold": "f1", "seed": 1234, "target_access": False,
            "checkpoint_sha256": complete["best_checkpoint_sha256"],
            "training_completion_sha256": checkpoint_digest(output / "training_complete.json"), "score_sha256": hashes})
        check_inputs(provenance)
        verify_completion(output, name)
        logger.info("COMPLETE %s | selected epoch%d EER%.4f%%; full source reload/export/calibration verified; no target", name, best_epoch, best*100)
        del reloaded
        gc.collect()
        return complete
    finally:
        logger.removeHandler(handler)
        handler.close()


def verify_completion(run, name):
    config, complete = (json.loads((run / f).read_text()) for f in ("config.json", "training_complete.json"))
    if (config["stage"] != STAGE or complete["stage"] != STAGE or config["concept_bank"] != bank_definition(name)
            or config["model"]["backend"]["concepts"] != bank_definition(name)["concepts"]
            or config["model"]["backend"]["group_indices"] != bank_definition(name)["group_indices"]
            or complete["concept_bank"] != name or config["sources"] != list(SOURCES)
            or config["fold"] != "f1" or complete["fold"] != "f1" or config["seed"] != 1234 or complete["seed"] != 1234
            or complete["target_access"] or not complete["selected_checkpoint_reloaded_and_source_verified"]):
        raise ValueError("Concept-bank completion identity changed")
    for filename, expected in [("config.json", complete["config_sha256"]), ("best.pt", complete["best_checkpoint_sha256"]),
            ("metrics.json", complete["metrics_sha256"]), ("source_calibration.json", complete["calibration_sha256"]),
            ("routing.json", complete["routing_sha256"]), *complete["source_scores_sha256"].items()]:
        if checkpoint_digest(run / filename) != expected:
            raise ValueError("Completed fit artifact changed")
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if (state["config_sha256"] != complete["config_sha256"] or state["epoch"] != complete["best_epoch"]
            or tensor_digest(state["model_state_dict"]["backend.prototypes"]) != complete["prototype_bytes_sha256"]):
        raise ValueError("Completed checkpoint bank binding changed")
    manifest = json.loads((run / "score_manifest.json").read_text())
    if (manifest["checkpoint_sha256"] != complete["best_checkpoint_sha256"]
            or manifest["training_completion_sha256"] != checkpoint_digest(run / "training_complete.json")
            or manifest["score_sha256"] != complete["source_scores_sha256"]):
        raise ValueError("Completed source manifest changed")
    verify_source_macro(state["source_dev"]["macro_eer"], complete["best_macro_source_dev_eer"])
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/experiments/lacf_diagnostics/frozen_features")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any((OUTPUT / name / "f1/1234").exists() for name in FINAL_CONCEPT) or (OUTPUT / "complete.json").exists():
        raise FileExistsError("Existing concept-bank runs preserved")
    launch = OUTPUT / "launch"; launch.mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(launch / "console.log")])
    logger = logging.getLogger("concept_moe")
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required; no fallback")
    free, total = torch.cuda.mem_get_info()
    if free < 4*1024**3:
        raise RuntimeError("Need at least4GiB free before this1GiB-capped queue")
    torch.cuda.set_per_process_memory_fraction(1024**3/total)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.nnpack.set_flags(False)
    guards = json.loads((ROOT / "outputs/experiments/token_fusion/adaptation_restart/launch/source_guards.json").read_text())
    frozen = json.loads((ROOT / "outputs/experiments/tf2_frozen_clap/f1/seed_1234/config.json").read_text())
    for path, expected in frozen["provenance"]["code_sha256"].items():
        if path in guards and guards[path] != expected:
            raise ValueError("Preserved TF2 source identities disagree")
        guards[path] = expected
    paths = [Path(__file__), ROOT / "src/audio_deepfake_detection/experiments/concept_bank/model.py",
        ROOT / "scripts/experiments/layer_fusion/train_full_f1.py", ROOT / "scripts/token_fusion/train.py",
        ROOT / "src/audio_deepfake_detection/experiments/layer_fusion/model.py",
        ROOT / "src/audio_deepfake_detection/experiments/layer_fusion/diagnostics.py",
        *[ROOT / f"src/audio_deepfake_detection/lacf/{name}.py" for name in ("model", "ablations", "probes", "training", "precision")]]
    guards.update({str(p.relative_to(ROOT)): checkpoint_digest(p) for p in paths})
    provenance = {"code_sha256": guards, "cache": str(args.cache.resolve()),
        "cache_config_sha256": checkpoint_digest(args.cache / "config.json"),
        "cache_completion_sha256": checkpoint_digest(args.cache / "complete.json"),
        "cache_execution_sha256": checkpoint_digest(args.cache / "execution_accelerated.json"),
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "cpu_threads": 2, "gpu_allocator_budget_gib": 1}}
    check_inputs(provenance)
    save_json(launch / "source_guards.json", guards)
    logger.info("PREPARING two explicit4+4 text banks on CPU; final audio encoders stay unloaded")
    prototypes = encode_banks(OUTPUT / "banks")
    execution = json.loads((args.cache / "execution_accelerated.json").read_text())
    if execution["pretrained"]["clap_revision"] != REVISION:
        raise ValueError("Frozen CLAP audio/text revisions differ")
    provenance["input_sha256"] = {str(p.resolve()): checkpoint_digest(p) for p in (OUTPUT / "banks").iterdir() if p.is_file()}
    save_json(OUTPUT / "queue_config.json", {"stage": STAGE, "fold": "f1", "seed": 1234,
        "banks": [bank_definition(n) for n in FINAL_CONCEPT], "target_access": False,
        "training_resume": False, "stop_on_failure": True, "provenance": provenance})
    logger.info("CACHE validate/load once: ASV2019,ASV5,CFAD train/dev; target untouched")
    bank = load_f1_bank(args.cache, args.data_root, logger)
    counts = {f"{d}/{split}": len(bank[d, split].labels) for d in SOURCES for split in ("train", "dev")}
    if (sum(counts[f"{d}/train"] for d in SOURCES) != 246137
            or sum(counts[f"{d}/dev"] for d in SOURCES) != 180194):
        raise ValueError("Canonical full-F1 source counts changed")
    save_json(launch / "cache_validation.json", {"canonical_ids_labels_and_payload_hashes_checked": True,
        "counts": counts, "target_access": False, "cache_completion_sha256": provenance["cache_completion_sha256"]})
    results = []
    for name in FINAL_CONCEPT:
        check_inputs(provenance)
        report = preflight(bank, prototypes[name], provenance, "cuda", logger)
        report.update(concept_bank=name, probe_model="unchanged C4 with this exact revised prototype buffer; detection BCE only",
            prototype_bytes_sha256=tensor_digest(prototypes[name]))
        save_json(launch / f"preflight_{name}.json", report)
        results.append(fit(bank, prototypes[name], name, OUTPUT / name / "f1/1234", provenance, "cuda", logger))
        torch.cuda.empty_cache()
    for name in FINAL_CONCEPT:
        verify_completion(OUTPUT / name / "f1/1234", name)
    check_inputs(provenance)
    save_json(OUTPUT / "complete.json", {"stage": STAGE, "verified_fits": 2, "target_access": False,
        "fits": {name: checkpoint_digest(OUTPUT / name / "f1/1234/training_complete.json") for name in FINAL_CONCEPT},
        "source_results": [{"bank": r["concept_bank"], "best_epoch": r["best_epoch"], "macro_source_eer": r["best_macro_source_dev_eer"]} for r in results]})
    logger.info("QUEUE COMPLETE: two source-verified fits; stop before target evaluation")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Concept-bank MoE queue stopped; preserve outputs; no automatic retry")
        raise
