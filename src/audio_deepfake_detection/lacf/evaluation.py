"""Completed primary LACF checkpoints and batch-one native score export."""

import hashlib
import json
import platform
from pathlib import Path

import torch
import transformers

from audio_deepfake_detection.lacf.cache import FrozenDevelopmentFeatures
from audio_deepfake_detection.lacf.model import PRIMARY_TRAINABLE_PARAMETERS
from audio_deepfake_detection.lacf.pretrained import load_local_model
from audio_deepfake_detection.lacf.training import device_batch, verify_fold_completion
from audio_deepfake_detection.protocol import (
    FOLDS, checkpoint_digest, utterance_id, verify_source_macro,
)


def load_completed_model(run_dir, fold_name, device):
    """Strictly reload a selected checkpoint; never download or initialize training."""
    run_dir = Path(run_dir)
    completion = verify_fold_completion(run_dir, fold_name)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    fold = FOLDS[fold_name]
    if (checkpoint["fold"] != fold_name or config["fold"] != fold_name or config["seed"] != 1234
            or checkpoint["epoch"] != completion["best_epoch"]
            or config["sources"] != list(fold["sources"]) or config["target"] != fold["target"]
            or config["target_split"] != fold["target_split"]):
        raise ValueError("Selected LACF checkpoint identity/membership mismatch")
    verify_source_macro(checkpoint["macro_dev_eer"], completion["best_macro_source_dev_eer"])
    root = Path(__file__).resolve().parents[3]
    for name, digest in config["source_sha256"].items():
        if checkpoint_digest(root / name) != digest:
            raise ValueError("Recorded training/encoding implementation changed: " + name)
    model, collator, revisions = load_local_model(segment_seconds=config["segment_seconds"])
    if any(config[key] != revisions[key] for key in ("wavlm_revision", "clap_revision")):
        raise ValueError("Local pretrained revision differs from the trained checkpoint")
    if (json.loads(json.dumps(model.configuration())) != json.loads(json.dumps(config["components"]))
            or json.loads(json.dumps(collator.configuration())) != json.loads(json.dumps(config["input_policy"]))
            or sum(p.numel() for p in model.parameters() if p.requires_grad) != PRIMARY_TRAINABLE_PARAMETERS):
        raise ValueError("Primary LACF model/processor configuration changed")
    state = checkpoint["model_state_dict"]
    for name, value in model.state_dict().items():
        if name.startswith(("wavlm.", "clap.", "prototypes")) and not torch.equal(value, state[name]):
            raise ValueError("Frozen pretrained state or text prototypes changed: " + name)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), collator, config, completion


def load_development_cache(reference, records, identity, device):
    """Read only the recorded cache. A mismatch fails instead of rebuilding it."""
    path = Path(reference["path"])
    if checkpoint_digest(path) != reference["sha256"]:
        raise ValueError("Recorded source feature-cache checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["identity"]
    expected = {**identity, "records_sha256": hashlib.sha256(json.dumps(records).encode()).hexdigest(),
                "count": len(records), "encoder_batch_size": 1, "backend_batch_size": 1,
                "torch": str(torch.__version__), "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "precision": "bfloat16_mixed_fp32_features",
                "platform": platform.machine(),
                "gpu": torch.cuda.get_device_name() if torch.device(device).type == "cuda" else "cpu"}
    if metadata != expected:
        raise ValueError("Source cache membership, encoding identity or runtime changed")
    if path.stem != hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest():
        raise ValueError("Source cache filename does not bind its identity")
    labels = torch.tensor([label for _, label in records], dtype=torch.long)
    features = payload["features"]
    if (not torch.equal(payload["labels"], labels) or set(features) != {"z_w", "v_c"}
            or any(features[key].shape != (len(labels), dim) or features[key].dtype != torch.float32
                   or not torch.isfinite(features[key]).all() for key, dim in (("z_w", 768), ("v_c", 512)))):
        raise ValueError("Invalid canonical source cache labels/features")
    return FrozenDevelopmentFeatures(features, labels, path, reference["sha256"])


@torch.inference_mode()
def cached_scores(model, cache, device, *, start=0):
    """Preserve the training reload's batch-one backend arithmetic and BF16 logits."""
    model.eval()
    audio = {key: value.to(device) for key, value in cache.features.items()}
    for index in range(start, len(cache.labels)):
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            output = model.classify_audio({key: value[index:index+1] for key, value in audio.items()})
        yield float(model.spoof_score(output).float()[0].cpu())


def cached_rows(model, cache, records, domain, data_root, device, *, start=0):
    for index, score in enumerate(cached_scores(model, cache, device, start=start), start):
        path, label = records[index]
        yield utterance_id(path, domain, data_root), domain, label, score
        if (index + 1) % 5000 == 0 or index + 1 == len(records):
            print(f"{domain}: {index + 1:,}/{len(records):,} cached source scores", flush=True)


@torch.inference_mode()
def direct_rows(model, loader, records, domain, data_root, device, *, start=0):
    """Native centered common interval; one recording, both frozen encoders."""
    model.eval()
    offset = 0
    for batch in loader:
        if len(batch["labels"]) != 1:
            raise ValueError("Native LACF evaluation requires batch one")
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            output = model(device_batch(batch, device))
        score = float(model.spoof_score(output).float()[0].cpu())
        path, label = records[offset]
        if int(batch["labels"][0]) != label:
            raise ValueError("Native evaluation changed canonical record order")
        yield utterance_id(path, domain, data_root), domain, label, score
        offset += 1
        if (start + offset) % 1000 == 0 or offset == len(records):
            print(f"{domain}: {start + offset:,}/{start + len(records):,} direct scores", flush=True)
    if offset != len(records):
        raise ValueError("Incomplete native LACF scoring")
