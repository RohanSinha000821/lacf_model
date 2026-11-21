"""Local-only official SSL initialization for the two frozen detector adaptations."""

from pathlib import Path

import torch
from torch import nn

from audio_deepfake_detection.protocol import checkpoint_digest

PRETRAINED = {
    "lhcc": ("facebook/wav2vec2-xls-r-300m", "1a640f32ac3e39899438a2931f9924c02f080a54", 24, 1024),
    "nsf_pinn": ("microsoft/wavlm-base-plus", "4c66d4806a428f2e922ccfa1a962776e232d487b", 12, 768),
}


def pretrained_identity(method):
    """Resolve and hash existing official files; never download missing weights."""
    from huggingface_hub import snapshot_download

    name, revision, _, _ = PRETRAINED[method]
    root = Path(snapshot_download(name, revision=revision, local_files_only=True))
    files = ("config.json", "pytorch_model.bin")
    return {"model": name, "revision": revision,
            "files_sha256": {name: checkpoint_digest(root / name) for name in files}}


class FrozenSpeechEncoder(nn.Module):
    def __init__(self, method, encoder=None):
        super().__init__()
        name, revision, layers, width = PRETRAINED[method]
        if encoder is None:
            from transformers import Wav2Vec2Model, WavLMModel

            cls = Wav2Vec2Model if method == "lhcc" else WavLMModel
            encoder = cls.from_pretrained(name, revision=revision, local_files_only=True,
                                          attn_implementation="sdpa" if method == "lhcc" else "eager",
                                          use_safetensors=False)
        if encoder.config.hidden_size != width or encoder.config.num_hidden_layers != layers:
            raise ValueError(f"Wrong official backbone geometry for {method}")
        self.encoder = encoder.requires_grad_(False).eval()
        self.method = method
        self.eval()

    def train(self, mode=True):
        # Frozen features must be deterministic, including dropout/SpecAugment/layerdrop.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, waves):
        if waves.ndim != 2 or waves.shape[1] < 400:
            raise ValueError("Expected fixed-length [batch,samples] 16kHz mono audio")
        self.encoder.eval()
        result = self.encoder(waves, output_hidden_states=self.method == "lhcc")
        if self.method == "lhcc":
            if len(result.hidden_states) != 25:
                raise ValueError("LHCC requires all 24 Transformer outputs, excluding convolutional state")
            return torch.stack(result.hidden_states[1:], dim=1).detach()
        return result.last_hidden_state.detach()
