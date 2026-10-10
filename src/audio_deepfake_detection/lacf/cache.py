"""Frozen source-development audio features; no random training crops or scores."""
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

from audio_deepfake_detection.protocol import checkpoint_digest
from audio_deepfake_detection.lacf.training import device_batch, report_progress


@dataclass
class FrozenDevelopmentFeatures:
    features: dict[str, torch.Tensor]
    labels: torch.Tensor
    path: Path | None = None
    sha256: str | None = None


@torch.inference_mode()
def frozen_development(model, loader, records, device, *, identity, directory, logger=None, phase="cache"):
    """Encode centered development crops once, retaining the original batch-one inputs."""
    if getattr(loader.dataset, "training", True):
        raise ValueError("Frozen feature caching is only for deterministic source development")
    if any(p.requires_grad for enc in (model.wavlm, model.clap) if enc is not None for p in enc.parameters()):
        raise ValueError("Cannot cache a trainable encoder")
    labels = torch.tensor([label for _, label in records], dtype=torch.long)
    metadata = {**identity, "records_sha256": hashlib.sha256(json.dumps(records).encode()).hexdigest(),
                "count": len(records), "encoder_batch_size": 1, "backend_batch_size": 1,
                "torch": str(torch.__version__), "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "precision": "bfloat16_mixed_fp32_features",
                "platform": platform.machine(),
                "gpu": torch.cuda.get_device_name() if torch.device(device).type == "cuda" else "cpu"}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = Path(directory) / (digest + ".pt")
    keys = {"z_w": 768} if model.wavlm is not None else {}
    if model.clap is not None:
        keys["v_c"] = 512
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["identity"] != metadata or not torch.equal(payload["labels"], labels):
            raise ValueError("Frozen development cache identity/labels mismatch")
        features = payload["features"]
        if logger is not None:
            logger.info("%s: reusing frozen audio features for %s recordings", phase, len(labels))
    else:
        model.eval()
        features = {key: torch.empty(len(labels), dim) for key, dim in keys.items()}
        started, offset = time.monotonic(), 0
        for batch in loader:
            size = len(batch["labels"])
            if size != 1 or not torch.equal(batch["labels"], labels[offset:offset+size]):
                raise ValueError("Source-development order or encoder batch size changed")
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                encoded = model.encode_audio(device_batch(batch, device))
            for key, value in encoded.items():
                features[key][offset:offset+size] = value.float().cpu()
            offset += size
            if offset % 1000 == 0 or offset == len(labels):
                report_progress(logger, phase, offset, len(labels), started)
        if offset != len(labels):
            raise ValueError("Incomplete canonical development cache")
    if set(features) != set(keys) or any(value.shape != (len(labels), keys[key])
            or value.dtype != torch.float32 or not torch.isfinite(value).all() for key, value in features.items()):
        raise ValueError("Invalid or nonfinite frozen development features")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        if temporary.exists():
            raise FileExistsError("Interrupted feature-cache build preserved: " + str(temporary))
        torch.save({"identity": metadata, "features": features, "labels": labels}, temporary)
        temporary.replace(path)
    return FrozenDevelopmentFeatures(features, labels, path, checkpoint_digest(path))
