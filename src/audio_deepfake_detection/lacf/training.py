"""The frozen-stage loop: configured model/objective, accumulation, source EER only."""

import json
import time

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


def report_progress(logger, phase, done, total, started, *, loss=None, examples=None):
    if logger is None:
        return
    elapsed = max(time.monotonic() - started, 1e-6)
    progress_rate = done / elapsed
    rate = (examples if examples is not None else done) / elapsed
    remaining = (total - done) / max(progress_rate, 1e-6)
    logger.info("%s | %s/%s %s (%.1f%%)%s | %.1f audio/s | elapsed %.1f min | ETA %.1f min",
                phase, done, total, "batches" if examples is not None else "recordings", 100 * done / total,
                " | loss %.5f" % loss if loss is not None else "", rate, elapsed / 60, remaining / 60)


def train_epoch(model, loader, optimizer, device, *, accumulation_steps=8, logger=None, phase="train"):
    if accumulation_steps < 1 or len(loader) < 1:
        raise ValueError("Expected positive accumulation and a nonempty loader")
    model.train()
    names = ("total", "detection", "semantic", "bonafide")
    loss_sums = torch.zeros(len(names), device=device)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    started = time.monotonic()
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
        values = torch.stack([losses[name].detach() for name in names])
        if not torch.isfinite(values).all():
            raise RuntimeError("Nonfinite LACF loss; training stops")
        (losses["total"] / group_size).backward()
        if offset + 1 == group_size:
            finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters if p.grad is not None])
            if not finite.all():
                raise RuntimeError("Nonfinite LACF gradients; optimizer update refused")
            optimizer.step()
            updates += 1
        size = len(batch["labels"])
        count += size
        loss_sums += values * size
        if (index + 1) % 100 == 0 or index + 1 == len(loader):
            report_progress(logger, phase, index + 1, len(loader), started,
                            loss=float(loss_sums[0] / count) if logger is not None else None, examples=count)
    if count == 0:
        raise ValueError("No training examples")
    return dict(zip(names, (loss_sums / count).cpu().tolist())), updates


@torch.inference_mode()
def source_eer(model, loader, device, *, logger=None, phase="dev"):
    from audio_deepfake_detection.lacf.cache import FrozenDevelopmentFeatures
    model.eval()
    started = time.monotonic()
    if isinstance(loader, FrozenDevelopmentFeatures):
        # Keep the exact batch-one backend arithmetic; transfer each domain once.
        audio = {key: value.to(device) for key, value in loader.features.items()}
        scores = torch.empty(len(loader.labels), device=device)
        for index in range(len(loader.labels)):
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                output = model.classify_audio({key: value[index:index+1] for key, value in audio.items()})
            scores[index] = model.spoof_score(output).float()[0]
            if (index + 1) % 5000 == 0 or index + 1 == len(loader.labels):
                report_progress(logger, phase, index + 1, len(loader.labels), started)
        scores = scores.cpu()
        if not torch.isfinite(scores).all():
            raise RuntimeError("Nonfinite source development scores")
        return compute_eer(loader.labels.tolist(), scores.tolist())
    labels, scores = [], []
    for index, batch in enumerate(loader):
        labels.extend(batch["labels"].tolist())
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            output = model(device_batch(batch, device))
        scores.extend(model.spoof_score(output).float().cpu().tolist())
        if (index + 1) % 1000 == 0 or index + 1 == len(loader):
            report_progress(logger, phase, index + 1, len(loader), started)
    return compute_eer(labels, scores)


def verify_fold_completion(run_dir, fold_name, seed=1234):
    """Queue gate: bound checkpoint, primary recipe and recorded source reload."""
    completion, digest = verify_completed_run(run_dir, fold_name, seed)
    config = json.loads((run_dir / "config.json").read_text())
    duration = config.get("segment_seconds", 10)
    family = "lacf4s" if duration == 4 else "lacf"
    if (duration not in (4, 10) or seed != 1234 or config.get("run_name") != family or config.get("fold") != fold_name
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
