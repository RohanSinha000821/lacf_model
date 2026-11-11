"""Sequential native F1 evaluation of the two completed balanced-bank MoE fits."""

import gc
import itertools
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.experiments.concept_bank.model import ConceptBankMoE
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    canonical_score_labels, checkpoint_digest, load_score_manifest, save_score_manifest,
    verify_completed_run, verify_source_macro,
)
from scripts.evaluate_scores import evaluate_run
from scripts.experiments.concept_bank.train_moe import OUTPUT, ROOT, SOURCES, check_inputs, verify_completion
from scripts.experiments.layer_fusion.evaluate_lacf_layer_moe import (
    TARGET_FEATURES, read, source_reload, target_chunks,
)
from scripts.experiments.token_fusion.evaluate import score_features
from scripts.token_fusion.train import save_json


EVALUATION = OUTPUT / "evaluations"


def verify_training(run, name, data_root):
    """Verify bank-bound completion and canonical source calibration before target access."""
    done = verify_completion(run, name)
    _, digest = verify_completed_run(run, "f1", 1234)
    if digest != done["best_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint identity differs")
    config = read(run / "config.json")
    check_inputs(config["provenance"])
    source = {domain: read_score_csv(run / f"source_dev_{domain}.csv", domain,
        expected_labels=canonical_score_labels(domain, "dev", data_root)) for domain in SOURCES}
    calibration = calibrate_source_scores(source)
    if calibration != read(run / "source_calibration.json"):
        raise ValueError("Frozen source calibration differs")
    verify_source_macro(calibration["source_dev"]["macro_eer"], done["best_macro_source_dev_eer"])
    return config, done


def load_detector(run, name, config, device):
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if state["source_dev"] != read(run / "metrics.json")["source_dev"]:
        raise ValueError("Selected checkpoint/source metrics differ")
    model = ConceptBankMoE(name, state["model_state_dict"]["backend.prototypes"])
    model.load_state_dict(state["model_state_dict"], strict=True)
    if json.loads(json.dumps(model.configuration())) != config["model"]:
        raise ValueError("Concept-bank model configuration differs")
    return model.to(device).requires_grad_(False).eval()


def evaluate(training, output, name, features, data_root, device, logger, *, bootstrap_resamples=1000):
    if output.exists():
        raise FileExistsError("Existing evaluation preserved; no overwrite or retry")
    if output.name != "1234" or output.parent.name != "f1" or output.parent.parent.name != name:
        raise ValueError("Use the bank/f1/1234 evaluation path")
    config, done = verify_training(training, name, data_root)
    model = load_detector(training, name, config, device)
    validation = source_reload(model, training, config, data_root, logger, device)
    check_inputs(config["provenance"])
    # Source gates above do not read any target metadata or features.
    output.mkdir(parents=True, exist_ok=False)
    for filename in ("best.pt", "config.json", "training_complete.json", "source_calibration.json"):
        shutil.copy2(training / filename, output / filename)
    shutil.copy2(training / "metrics.json", output / "source_metrics.json")
    save_json(output / "source_validation.json", validation)
    manifest = load_score_manifest(output, "f1", 1234, done["best_checkpoint_sha256"], create=True)
    for domain in SOURCES:
        filename = f"source_dev_{domain}.csv"
        shutil.copy2(training / filename, output / filename)
        manifest["score_sha256"][filename] = checkpoint_digest(output / filename)
    inference = {"input": "centered common physical interval up to10s", "precision": config["precision"],
        "features": "verified frozen WavLM13-layer/CLAP raw projection cache; no encoder reruns",
        "head_batch": 132, "chunk_records": 1024, "score": "scalar spoof logit"}
    save_json(output / "evaluation_config.json", {"model_name": "LACF Layer-MoE", "concept_bank": config["concept_bank"],
        "fold": "f1", "seed": 1234, "training_run": str(training.resolve()),
        "training_completion_sha256": checkpoint_digest(training / "training_complete.json"),
        "target_features": str(features.resolve()), "target": "speechfake/test", "inference": inference,
        "exploratory_after_prior_target_exposure": True, "target_tuning": False,
        "bootstrap_resamples": bootstrap_resamples, "bootstrap_seed": 2026})
    manifest.update(model_family="lacf_layer_moe_balanced_concepts", concept_bank=config["concept_bank"],
        inference=inference, training_run=str(training.resolve()), training_metrics_copy="source_metrics.json",
        completion_record_scope="original training snapshot; metrics_sha256 binds source_metrics.json",
        exploratory_after_prior_target_exposure=True,
        source_validation_sha256=checkpoint_digest(output / "source_validation.json"),
        target_features_sha256=checkpoint_digest(features / "complete.json"),
        evaluation_config_sha256=checkpoint_digest(output / "evaluation_config.json"),
        scorer_sha256={str(path.relative_to(ROOT)): checkpoint_digest(path) for path in (
            Path(__file__), ROOT / "src/audio_deepfake_detection/experiments/concept_bank/model.py",
            ROOT / "scripts/experiments/layer_fusion/evaluate_lacf_layer_moe.py",
            ROOT / "scripts/experiments/token_fusion/evaluate.py")})
    ids, values = [], []
    for identifiers, arrays in target_chunks(features, Path(config["provenance"]["cache"]), data_root):
        check_inputs(config["provenance"])
        scores = score_features(model, arrays, device)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite target scores; no inputs may be skipped")
        ids.extend(identifiers); values.extend(scores.tolist())
        if len(ids) % 10240 == 0:
            logger.info("TARGET %s | %d recordings scored", name, len(ids))
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
    logger.info("METRICS %s | raw/z-score transfer; %d bootstrap resamples", name, bootstrap_resamples)
    report = evaluate_run(output, "f1", seed=1234, source_data_root=data_root,
        target_data_root=data_root, bootstrap_resamples=bootstrap_resamples)
    check_inputs(config["provenance"])
    save_json(output / "evaluation_complete.json", {"model_name": "LACF Layer-MoE", "concept_bank": name,
        "fold": "f1", "seed": 1234, "checkpoint_sha256": done["best_checkpoint_sha256"],
        "metrics_sha256": checkpoint_digest(output / "metrics.json"),
        "source_validation_sha256": checkpoint_digest(output / "source_validation.json"),
        "evaluation_config_sha256": checkpoint_digest(output / "evaluation_config.json"),
        "score_sha256": manifest["score_sha256"], "source_selected_checkpoint_only": True,
        "exploratory_after_prior_target_exposure": True, "no_target_tuning": True})
    logger.info("COMPLETE %s | target EER %.4f%% AUROC %.6f", name, report["target"]["eer"]*100, report["target"]["auroc"])
    return report


def main():
    if EVALUATION.exists():
        raise FileExistsError("Evaluation queue already exists; preserve prior evidence")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required")
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("concept_moe_evaluation")
    EVALUATION.mkdir(parents=True, exist_ok=False)
    for name in ("vocoder", "neural_synthesis"):
        evaluate(OUTPUT / name / "f1/1234", EVALUATION / name / "f1/1234", name, TARGET_FEATURES,
            Path("/mnt/drive/audio-deepfake-cache"), torch.device("cuda"), logger)
    save_json(EVALUATION / "complete.json", {"verified_evaluations": 2, "fold": "f1", "seed": 1234,
        "evaluation_completions_sha256": {name: checkpoint_digest(EVALUATION / name / "f1/1234/evaluation_complete.json")
            for name in ("vocoder", "neural_synthesis")}, "no_target_tuning": True})


if __name__ == "__main__":
    main()
