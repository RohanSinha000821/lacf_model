"""Isolated TF2 encoder adaptation; original encoders and TF2 stay unchanged."""

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils import parametrize

from audio_deepfake_detection.lacf.precision import keep_normalization_fp32
from audio_deepfake_detection.token_fusion.model import TokenFusion


@dataclass(frozen=True)
class AdaptationConfig:
    clap_final_stage: bool
    rank: int = 8
    alpha: float = 16
    lora_lr: float = 1e-4
    clap_lr: float = 1e-5
    fusion_lr: float = 3e-4
    weight_decay: float = 1e-4

    def __post_init__(self):
        if self.rank < 1 or any(not math.isfinite(v) or v <= 0 for v in
            (self.alpha, self.lora_lr, self.clap_lr, self.fusion_lr)) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid adaptation configuration")


class LowRankWeight(nn.Module):
    """Parametrize effective weights: WavLM's functional attention reads .weight."""
    def __init__(self, weight, rank, alpha):
        super().__init__()
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(weight.new_empty(rank, weight.shape[1], dtype=torch.float32))
        self.lora_b = nn.Parameter(weight.new_zeros(weight.shape[0], rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, original):
        with torch.autocast(original.device.type, enabled=False):
            return original.float() + (self.lora_b.float() @ self.lora_a.float()) * self.scale


def add_wavlm_lora(wavlm, config):
    """Target exactly query/value weights in all twelve blocks; biases stay frozen."""
    if len(wavlm.encoder.layers) != 12:
        raise ValueError("Expected twelve WavLM Transformer blocks")
    paths = []
    with torch.random.fork_rng(devices=[]):
        for index, block in enumerate(wavlm.encoder.layers):
            for name in ("q_proj", "v_proj"):
                linear = getattr(block.attention, name)
                if parametrize.is_parametrized(linear):
                    raise ValueError("LoRA already installed; refusing double injection")
                parametrize.register_parametrization(linear, "weight",
                    LowRankWeight(linear.weight, config.rank, config.alpha))
                paths.append(f"encoder.layers.{index}.attention.{name}.weight")
    return paths


class AdaptedTF2(nn.Module):
    def __init__(self, wavlm, clap, config, *, fusion=None):
        super().__init__()
        self.config = config
        self.fusion = TokenFusion("TF2") if fusion is None else fusion
        self.wavlm, self.clap = wavlm, clap
        keep_normalization_fp32(self.wavlm)
        keep_normalization_fp32(self.clap)
        # Replacement normalizations create new Parameters; freeze afterwards.
        self.wavlm.requires_grad_(False)
        self.clap.requires_grad_(False)
        self.lora_targets = add_wavlm_lora(wavlm, config)
        if config.clap_final_stage:
            stages = self.clap.audio_model.audio_encoder.layers
            if len(stages) != 4:
                raise ValueError("Expected four unfused CLAP audio stages")
            stages[-1].requires_grad_(True)
            self.clap.audio_projection.requires_grad_(True)
        # Preserve the frozen-cache encoder execution policy, including BN stats.
        self.train()

    @property
    def objective(self):
        return self.fusion.objective

    def train(self, mode=True):
        super().train(mode)
        self.wavlm.eval()
        self.clap.eval()
        return self

    def wavlm_features(self, inputs):
        """Slice away collator padding before any WavLM convolution/normalization."""
        values, masks = inputs["input_values"], inputs["attention_mask"]
        lengths = masks.long().sum(-1)
        expected = torch.arange(masks.shape[1], device=masks.device)[None] < lengths[:, None]
        if not torch.equal(masks.bool(), expected) or (lengths < 400).any():
            raise ValueError("Expected right-padded valid WavLM inputs, at least400samples")
        groups = defaultdict(list)
        for index, length in enumerate(lengths.tolist()):
            groups[length].append(index)
        pooled, order = [], []
        for length, indices in groups.items():
            selected = {key: value[indices, :length] for key, value in inputs.items()}
            captured = {}
            hook = self.wavlm.feature_projection.register_forward_hook(
                lambda _m, _i, output: captured.update(frontend=output[0]))
            try:
                output = self.wavlm(**selected, output_hidden_states=True, return_dict=True)
                states = (captured.pop("frontend"), *output.hidden_states[1:])
                if len(states) != 13:
                    raise ValueError("Expected projected frontend and twelve WavLM outputs")
                mask = self.wavlm._get_feature_vector_attention_mask(states[-1].shape[1], selected["attention_mask"])
                with torch.autocast(values.device.type, enabled=False):
                    result = torch.stack([(v.float()*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
                                          for v in states], dim=1)
                pooled.append(result)
                order.extend(indices)
            finally:
                hook.remove()
        inverse = torch.argsort(torch.tensor(order, device=values.device))
        return torch.cat(pooled)[inverse]

    def clap_features(self, inputs):
        # Use the original raw projection; TF2 itself applies the two FP32 L2s.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.config.clap_final_stage):
            outputs = self.clap.audio_model(**inputs, return_dict=True)
            return self.clap.audio_projection(outputs.pooler_output).float()

    def forward(self, batch):
        return self.fusion({"wavlm": self.wavlm_features(batch["wavlm"]),
                            "clap": self.clap_features(batch["clap"])})

    def optimizer(self):
        groups = [dict(params=[p for p in self.wavlm.parameters() if p.requires_grad], lr=self.config.lora_lr, name="wavlm_lora"),
                  dict(params=list(self.fusion.parameters()), lr=self.config.fusion_lr, name="tf2")]
        clap = [p for p in self.clap.parameters() if p.requires_grad]
        if clap:
            groups.append(dict(params=clap, lr=self.config.clap_lr, name="clap_final_stage_projection"))
        selected = [id(p) for g in groups for p in g["params"]]
        if len(selected) != len(set(selected)) or set(selected) != {id(p) for p in self.parameters() if p.requires_grad}:
            raise ValueError("Optimizer must cover each enabled parameter exactly once")
        return torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)

    def trainable_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        parameters = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("Sparse adaptation state has missing/unexpected trainable parameters")
        with torch.no_grad():
            for name, p in parameters.items():
                if state[name].shape != p.shape or not torch.isfinite(state[name]).all():
                    raise ValueError("Invalid adaptation parameter")
                p.copy_(state[name])

    def configuration(self):
        counts = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                  for name, module in (("wavlm_lora", self.wavlm), ("clap", self.clap), ("fusion", self.fusion))}
        return {**asdict(self.config), "counts": counts, "trainable_parameters": sum(counts.values()),
                "lora_targets": self.lora_targets, "lora_dropout": 0,
                "lora_initialization": "A Kaiming uniform; B zero; original weights frozen",
                "clap_scope": "audio_model.audio_encoder.layers.3 and audio_projection only" if self.config.clap_final_stage else "fully frozen",
                "encoder_mode": "eval even during fitting; no new encoder dropout/masking/BN updates",
                "input_policy": "same fixed physical up-to10s views; native unpadded WavLM groups",
                "fusion": self.fusion.configuration() if hasattr(self.fusion, "configuration") else type(self.fusion).__name__,
                "objective": "TF2 detection BCE only", "checkpoint": "trainable parameters plus pinned base provenance"}
