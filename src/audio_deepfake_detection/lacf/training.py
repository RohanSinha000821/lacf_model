"""The frozen-stage loop: configured model/objective, accumulation, source EER only."""

import json

import torch

from audio_deepfake_detection.metrics import compute_eer
from audio_deepfake_detection.lacf.model import PRIMARY_TRAINABLE_PARAMETERS
from audio_deepfake_detection.protocol import FOLDS, verify_completed_run, verify_source_macro


def device_batch(batch, device):
    return {key: ({name: value.to(device, non_blocking=True) for name, value in values.items()}
                  if isinstance(values, dict) else values.to(device, non_blocking=True))
            for key, values in batch.items()}


def make_optimizer(model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No enabled trainable components")
    return torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-4)


def train_epoch(model, loader, optimizer, device, *, accumulation_steps=8):
    if accumulation_steps < 1 or len(loader) < 1:
        raise ValueError("Expected positive accumulation and a nonempty loader")
    model.train()
    loss_sums = dict.fromkeys(("total", "detection", "semantic", "bonafide"), 0.0)
    count, updates = 0, 0
    for index, batch in enumerate(loader):
        offset = index % accumulation_steps
        if offset == 0:
            optimizer.zero_grad(set_to_none=True)
            # Last incomplete group divides by its actual microbatch count.
            group_size = min(accumulation_steps, len(loader) - index)
        batch = device_batch(batch, device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            losses = model.objective(model(batch), batch["labels"])
        if not all(torch.isfinite(loss).all() for loss in losses.values()):
            raise RuntimeError("Nonfinite LACF loss; training stops")
        (losses["total"] / group_size).backward()
        if offset + 1 == group_size:
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise RuntimeError("Nonfinite LACF gradients; optimizer update refused")
            optimizer.step()
            updates += 1
        size = len(batch["labels"])
        count += size
        for name, loss in losses.items():
            loss_sums[name] += float(loss.detach()) * size
    if count == 0:
        raise ValueError("No training examples")
    return {name: value / count for name, value in loss_sums.items()}, updates


@torch.inference_mode()
def source_eer(model, loader, device):
    model.eval()
    labels, scores = [], []
    for batch in loader:
        labels.extend(batch["labels"].tolist())
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            output = model(device_batch(batch, device))
        scores.extend(model.spoof_score(output).float().cpu().tolist())
    return compute_eer(labels, scores)


def verify_fold_completion(run_dir, fold_name, seed=1234):
    """Queue gate: bound checkpoint, primary recipe and recorded source reload."""
    completion, digest = verify_completed_run(run_dir, fold_name, seed)
    config = json.loads((run_dir / "config.json").read_text())
    if (seed != 1234 or config.get("run_name") != "lacf" or config.get("fold") != fold_name
            or config.get("seed") != seed or config.get("final_seed_plan") != [1234]
            or config.get("sources") != list(FOLDS[fold_name]["sources"])
            or config.get("precision") != "bfloat16_mixed" or config.get("sensitive_dtype") != "float32"
            or config.get("trainable_parameters") != PRIMARY_TRAINABLE_PARAMETERS
            or config.get("selected_epoch") != completion["best_epoch"]
            or config.get("selected_checkpoint_sha256") != digest
            or completion.get("best_checkpoint_sha256") != digest
            or completion.get("selected_checkpoint_reloaded_and_source_verified") is not True):
        raise ValueError("LACF fold completion/recipe/checkpoint verification failed")
    reloaded = completion.get("reloaded_dev_eers", {})
    if set(reloaded) != set(FOLDS[fold_name]["sources"]):
        raise ValueError("Missing or unexpected source-domain reload EERs")
    for eer in reloaded.values():
        verify_source_macro(eer, eer)
    macro = sum(reloaded.values()) / len(reloaded)
    verify_source_macro(macro, completion["best_macro_source_dev_eer"])
    verify_source_macro(macro, config.get("selected_macro_source_dev_eer"))
    return completion
