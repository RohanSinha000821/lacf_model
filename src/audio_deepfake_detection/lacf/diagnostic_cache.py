"""Fixed source crops and frozen, pooled features for the staged diagnostics."""

import hashlib
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from audio_deepfake_detection.data import resample_audio


def crop_intervals(frames, rate, identifier, *, training):
    """Centre plus two reproducible random views; identical intervals share a row."""
    if frames <= 0 or rate <= 0:
        raise ValueError("Empty audio or invalid sample rate")
    length = min(frames, round(10 * rate))
    maximum = frames - length
    starts = [maximum // 2]
    if training:
        for view in range(2):
            digest = hashlib.sha256(f"1234\0{identifier}\0{view}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
            starts.append(int(torch.randint(maximum + 1, (1,), generator=generator)))
    unique = list(dict.fromkeys(starts))
    return [(start, length) for start in unique], [unique.index(start) for start in starts]


class FixedCropDataset(Dataset):
    def __init__(self, records, identifiers, *, training):
        if len(records) != len(identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("Expected unique canonical IDs and corresponding records")
        self.records, self.identifiers, self.training = records, identifiers, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        if label not in (0, 1):
            raise ValueError("Expected binary canonical labels")
        # Same read/mono policy as load_audio_segment; explicit offsets are needed
        # here to record fixed crops and avoid re-encoding identical short views.
        with sf.SoundFile(path) as stream:
            rate, frames = stream.samplerate, len(stream)
            intervals, aliases = crop_intervals(frames, rate, self.identifiers[index], training=self.training)
            views = []
            for start, length in intervals:
                stream.seek(start)
                audio = stream.read(length, dtype="float32", always_2d=True)
                if len(audio) != length or not np.isfinite(audio).all():
                    raise ValueError(f"Incomplete or nonfinite audio: {path}")
                common = torch.from_numpy(audio).mean(dim=1).contiguous()
                views.append((resample_audio(common, rate, 16000).float(),
                              resample_audio(common, rate, 48000).float(), label))
        return {"index": index, "id": self.identifiers[index], "label": label,
                "rate": rate, "frames": frames, "intervals": intervals,
                "view_rows": aliases, "views": views}


class FixedCropCollator:
    """Preprocess native views in workers; never add waveform padding for grouping."""
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, record):
        return {**record, "views": [self.collator([view]) for view in record["views"]]}


def grouped_views(records, batch_size, *, branches=("wavlm", "clap")):
    """Group only equal native waveform lengths, retaining unique-row indices."""
    if batch_size < 1:
        raise ValueError("Expected positive encoder batch size")
    groups = defaultdict(list)
    row = 0
    for record in records:
        for view in record["views"]:
            mask = view["wavlm"]["attention_mask"]
            if not mask.bool().all():
                raise ValueError("Diagnostic extraction must not pad WavLM waveforms")
            groups[mask.shape[-1]].append((row, view))
            row += 1
    for values in groups.values():
        for offset in range(0, len(values), batch_size):
            selected = values[offset:offset + batch_size]
            batch = {key: {name: torch.cat([item[1][key][name] for item in selected])
                           for name in selected[0][1][key]} for key in branches}
            batch["labels"] = torch.cat([item[1]["labels"] for item in selected])
            yield [item[0] for item in selected], batch


@torch.inference_mode()
def pooled_wavlm_layers(encoder, inputs, device):
    """Mask-pool projected frontend and twelve Transformer outputs in FP32."""
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Diagnostic encoders must be frozen and in eval mode")
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    captured = {}
    hook = encoder.feature_projection.register_forward_hook(
        lambda _module, _inputs, output: captured.update(frontend=output[0]))
    try:
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            output = encoder(**inputs, output_hidden_states=True, return_dict=True)
        representations = (captured.pop("frontend"), *output.hidden_states[1:])
        if len(representations) != 13:
            raise ValueError("Expected projected frontend and twelve Transformer outputs")
        mask = encoder._get_feature_vector_attention_mask(
            representations[-1].shape[1], inputs["attention_mask"])
        if not mask.any(dim=1).all():
            raise ValueError("No valid WavLM feature frames")
        with torch.autocast(torch.device(device).type, enabled=False):
            pooled = torch.stack([(value.float() * mask.unsqueeze(-1)).sum(1)
                                  / mask.sum(1, keepdim=True) for value in representations], dim=1)
        del output, representations
    finally:
        hook.remove()
    if pooled.shape[1:] != (13, 768) or not torch.isfinite(pooled).all():
        raise ValueError("Invalid or nonfinite pooled WavLM features")
    return pooled.cpu().numpy()


@torch.inference_mode()
def projected_clap_audio(encoder, inputs, device):
    """Capture the raw projection before get_audio_features normalizes it."""
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Diagnostic encoders must be frozen and in eval mode")
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    captured = {}
    hook = encoder.audio_projection.register_forward_hook(
        lambda _module, _inputs, output: captured.update(projected=output.float()))
    try:
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            encoder.get_audio_features(**inputs)
        projected = captured.pop("projected")
    finally:
        hook.remove()
    if projected.ndim != 2 or projected.shape[1] != 512 or not torch.isfinite(projected).all():
        raise ValueError("Invalid or nonfinite CLAP projection")
    return projected.cpu().numpy()


def all_layer_features(model, batch, device):
    """Native wrapper retained for direct parity checks and the original cache."""
    return {"wavlm_layers": pooled_wavlm_layers(model.wavlm, batch["wavlm"], device),
            "clap_projected": projected_clap_audio(model.clap, batch["clap"], device)}
