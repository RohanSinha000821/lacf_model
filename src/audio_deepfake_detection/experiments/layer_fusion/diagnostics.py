"""Descriptive source-only routing summaries; never used for checkpoint selection."""

import torch

from audio_deepfake_detection.lacf.probes import FeatureBatches, OUTER_FOLDS
from audio_deepfake_detection.protocol import FOLDS


@torch.inference_mode()
def source_diagnostics(model, bank, fold, seed, device):
    if fold not in OUTER_FOLDS:
        raise ValueError("Layer-fusion diagnostics are restricted to F1–F3 sources")
    result = {"scope": "source development only; descriptive, not a selection metric",
              "condition": model.condition, "layer_alpha": None, "domains": {}}
    if model.layer_weights is not None:
        result["layer_alpha"] = model.layer_weights.detach().float().softmax(0).cpu().tolist()
    if model.condition not in ("C3", "C4"):
        return result
    model.eval()
    for domain in FOLDS[fold]["sources"]:
        loader = FeatureBatches([bank[domain, "dev"]], "weighted", device, training=False, seed=seed)
        # Per group: count, sum weights, sum squared weights, sum entropy, argmax counts.
        totals = torch.zeros(3, 11, dtype=torch.float64, device=device)
        for batch in loader:
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                weights = model.layer_fusion.routing_weights(batch["wavlm"])
            with torch.autocast(torch.device(device).type, enabled=False):
                entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1, keepdim=True)
                top = torch.nn.functional.one_hot(weights.argmax(-1), 3)
                weights64 = weights.double()
                values = torch.cat((torch.ones_like(entropy).double(), weights64, weights64.square(),
                                    entropy.double(), top.double()), -1)
                masks = torch.stack((torch.ones_like(batch["labels"]), batch["labels"] == 0,
                                     batch["labels"] == 1)).double()
                totals += masks @ values
        groups = {}
        for name, row in zip(("all", "bonafide", "spoof"), totals.cpu()):
            count = int(row[0])
            if count == 0:
                groups[name] = {"count": 0}
                continue
            mean = row[1:4] / count
            groups[name] = {"count": count, "mean_weights": mean.tolist(),
                            "std_weights": (row[4:7] / count - mean.square()).clamp_min(0).sqrt().tolist(),
                            "mean_entropy_nats": float(row[7] / count),
                            "top_expert_counts": row[8:11].long().tolist()}
        result["domains"][domain] = groups
    result["interpretation"] = ("Compare soft weights, entropy and class/domain distributions; argmax ties select index 0. "
                                "Routing differences do not establish that the router learned a dataset shortcut.")
    return result
