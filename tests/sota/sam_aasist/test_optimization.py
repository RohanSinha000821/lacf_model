"""Synthetic CPU checks of SAM mathematics, BatchNorm and fail-closed checkpoints."""
import copy
import math
import numpy as np
import pytest
import torch
from torch import nn
from audio_deepfake_detection.sota.sam_aasist import SAMAASIST
from audio_deepfake_detection.sota.sam_optimization import NonFiniteSAMStep, SharpnessAwareStep
from scripts.aasist.train import CLASS_WEIGHTS, MAX_EPOCHS, MIN_LEARNING_RATE, LEARNING_RATE, make_optimizer

class Tiny(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8), nn.SELU(), nn.Dropout(dropout), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def spoof_score(logits):
        return logits[:, 1] - logits[:, 0]

def batch(seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(12, 6, generator=generator, dtype=dtype), torch.tensor([0, 1] * 6)

def constant_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

def weighted_ce(dtype=torch.float32):
    return nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, dtype=dtype))

def parameters(model):
    return [p.detach().clone() for p in model.parameters()]

def test_label_weight_and_score_direction_match_upstream_convention():
    upstream_logits = torch.randn(16, 2, requires_grad=True)  # [spoof, bona fide]
    public_labels = torch.randint(0, 2, (16,))  # 0 bona fide, 1 spoof
    upstream_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))(upstream_logits, 1 - public_labels)
    public_logits = upstream_logits.flip(1)
    candidate_loss = weighted_ce()(public_logits, public_labels)
    torch.testing.assert_close(candidate_loss, upstream_loss)
    score = SAMAASIST.spoof_score(public_logits)
    torch.testing.assert_close(score, upstream_logits[:, 0] - upstream_logits[:, 1])

def reference_sam(model, x, labels, rho, lr, weight_decay):
    """Independent autograd reference: no candidate or upstream SAM code."""
    model = copy.deepcopy(model).train()
    params = list(model.parameters())
    start = [p.detach().clone() for p in params]
    first = nn.functional.cross_entropy(model(x), labels, weight=torch.tensor(CLASS_WEIGHTS, dtype=x.dtype))
    g1 = torch.autograd.grad(first, params)
    norm = torch.sqrt(sum((g * g).sum() for g in g1))
    with torch.no_grad():
        for p, g in zip(params, g1):
            p.add_(rho * g / (norm + 1e-12))
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            momenta[module], module.momentum = module.momentum, 0.0
    second = nn.functional.cross_entropy(model(x), labels, weight=torch.tensor(CLASS_WEIGHTS, dtype=x.dtype))
    g2 = torch.autograd.grad(second, params)
    for module, momentum in momenta.items():
        module.momentum = momentum
    with torch.no_grad():
        for p, w in zip(params, start):
            p.copy_(w)
    adam = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    for p, g in zip(params, g2):
        p.grad = g.clone()
    adam.step()
    return model, g1, g2, start

def candidate_step(model, x, labels, rho=0.05, dtype=torch.float32):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = constant_scheduler(optimizer)
    sam = SharpnessAwareStep(model, optimizer, scheduler, weighted_ce(dtype), rho=rho)
    return sam, sam.step(x, labels)

def test_sam_update_matches_independent_reference_and_uses_second_pass_gradient():
    torch.manual_seed(0)
    model = Tiny().double().train()
    x, labels = batch(dtype=torch.float64)
    expected, g1, g2, start = reference_sam(model, x, labels, rho=0.05, lr=LEARNING_RATE, weight_decay=1e-4)
    sam, (loss, perturbed_loss) = candidate_step(model, x, labels, dtype=torch.float64)
    for actual, reference in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-14)
    for name in ("running_mean", "running_var", "num_batches_tracked"):
        torch.testing.assert_close(getattr(model.net[1], name), getattr(expected.net[1], name), rtol=0, atol=1e-14)
    # Adam's first moment equals (1-beta1)*(g2 + wd*w): the perturbed-point gradient at restored w.
    beta1 = 0.9
    for p, a, b, w in zip(model.parameters(), g1, g2, start):
        moment = sam.optimizer.state[p]["exp_avg"]
        torch.testing.assert_close(moment, (1 - beta1) * (b + 1e-4 * w), rtol=0, atol=1e-15)
    assert any(not torch.allclose(a, b) for a, b in zip(g1, g2))
    assert perturbed_loss > loss

def test_same_unmodified_batch_feeds_both_passes():
    model = Tiny(dropout=0.2).train()
    x, labels = batch()
    original = x.clone()
    seen = []
    model.register_forward_pre_hook(lambda _module, inputs: seen.append((inputs[0].data_ptr(), inputs[0].clone())))
    candidate_step(model, x, labels)
    assert len(seen) == 2 and seen[0][0] == seen[1][0] == x.data_ptr()
    assert torch.equal(seen[0][1], original) and torch.equal(seen[1][1], original) and torch.equal(x, original)

def test_exactly_one_adam_and_scheduler_update_per_batch_with_production_cosine():
    model = Tiny().train()
    optimizer, scheduler = make_optimizer(model, steps_per_epoch=10)
    sam = SharpnessAwareStep(model, optimizer, scheduler, weighted_ce(), rho=0.05)
    for step in range(1, 4):
        sam.step(*batch(step))
        assert all(int(state["step"]) == step for state in optimizer.state.values())
        assert scheduler.last_epoch == step
        expected = MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * 0.5 * (
            1 + math.cos(math.pi * step / (MAX_EPOCHS * 10)))
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-12)
        assert all(p.grad is None for p in model.parameters())

def test_production_cosine_matches_upstream_closed_form_over_whole_schedule():
    model = Tiny()
    optimizer, scheduler = make_optimizer(model, steps_per_epoch=7)
    total = MAX_EPOCHS * 7
    worst = 0.0
    for step in range(1, total + 1):
        optimizer.step()
        scheduler.step()
        # Upstream optimizers/scheduler.py CosineAnnealingScheduler (LambdaLR) closed form.
        expected = LEARNING_RATE * (MIN_LEARNING_RATE / LEARNING_RATE + (1 - MIN_LEARNING_RATE / LEARNING_RATE)
                                    * 0.5 * (1 + np.cos(step / total * np.pi)))
        worst = max(worst, abs(optimizer.param_groups[0]["lr"] - expected) / expected)
    assert worst < 1e-9

def test_batchnorm_running_stats_follow_first_pass_only():
    torch.manual_seed(2)
    model = Tiny().train()
    x, labels = batch()
    first_pass_only = copy.deepcopy(model)
    first_pass_only(x)
    candidate_step(model, x, labels)
    bn, expected = model.net[1], first_pass_only.net[1]
    torch.testing.assert_close(bn.running_mean, expected.running_mean, rtol=0, atol=0)
    torch.testing.assert_close(bn.running_var, expected.running_var, rtol=0, atol=0)
    assert int(bn.num_batches_tracked) == 2  # Same counter behavior as the upstream momentum-zero policy.
    assert bn.momentum == 0.1

class FailingSecondForward(Tiny):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("synthetic perturbed-pass failure")
        return super().forward(x)

@pytest.mark.parametrize("failure", ["nonfinite_second_loss", "exception_in_second_forward"])
def test_failure_restores_unperturbed_weights_batchnorm_and_refuses_update(failure):
    torch.manual_seed(3)
    model = (FailingSecondForward() if failure == "exception_in_second_forward" else Tiny()).train()
    before = parameters(model)
    stats = model.net[1].running_mean.clone()
    calls = {"n": 0}

    def criterion(logits, labels):
        calls["n"] += 1
        loss = weighted_ce()(logits, labels)
        return loss * float("nan") if failure == "nonfinite_second_loss" and calls["n"] == 2 else loss

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = constant_scheduler(optimizer)
    sam = SharpnessAwareStep(model, optimizer, scheduler, criterion, rho=0.05)
    with pytest.raises(NonFiniteSAMStep if failure == "nonfinite_second_loss" else RuntimeError):
        sam.step(*batch())
    assert all(torch.equal(a, b) for a, b in zip(parameters(model), before))
    assert not sam.perturbed and model.net[1].momentum == 0.1
    assert not optimizer.state and scheduler.last_epoch == 0
    assert not torch.equal(model.net[1].running_mean, stats)  # Only the legitimate first pass updated them.

def test_checkpoint_round_trip_includes_adam_moments_and_continues_identically(tmp_path):
    torch.manual_seed(4)
    model = Tiny(dropout=0.2).train()
    optimizer, scheduler = make_optimizer(model, steps_per_epoch=5)
    sam = SharpnessAwareStep(model, optimizer, scheduler, weighted_ce(), rho=0.05)
    for step in range(2):
        sam.step(*batch(step))
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "sam_state": sam.state_dict()}, path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert {"exp_avg", "exp_avg_sq", "step"} <= set(saved["optimizer_state_dict"]["state"][0])
    restored = Tiny(dropout=0.2).train()
    restored.load_state_dict(saved["model_state_dict"])
    optimizer2, scheduler2 = make_optimizer(restored, steps_per_epoch=5)
    optimizer2.load_state_dict(saved["optimizer_state_dict"])
    scheduler2.load_state_dict(saved["scheduler_state_dict"])
    sam2 = SharpnessAwareStep(restored, optimizer2, scheduler2, weighted_ce(), rho=0.05)
    sam2.load_state_dict(saved["sam_state"])
    for candidate in (sam, sam2):
        torch.manual_seed(99)
        candidate.step(*batch(9))
    assert all(torch.equal(a, b) for a, b in zip(parameters(model), parameters(restored)))
    with pytest.raises(ValueError):
        sam2.load_state_dict({**saved["sam_state"], "rho": 0.01})
