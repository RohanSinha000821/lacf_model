"""One non-adaptive SAM update on an already prepared mini-batch.

Mathematics and BatchNorm policy follow the pinned SAM-AntiSpoofing code
(optimizers/sam.py and optimizers/bypass_bn.py, commit 59c69de):

    g1 = grad L(w)                      first pass, BatchNorm running stats update
    e  = rho * g1 / (||g1||_2 + 1e-12)  global L2 norm over every gradient
    g2 = grad L(w + e)                  second pass, running-stat momentum 0
    w  <- base_optimizer(w, g2)         restored unperturbed w, one Adam update
    scheduler.step()                    once per completed Adam update

Unlike the upstream wrapper, this is not a torch.optim.Optimizer subclass: the
base Adam, its scheduler and the SAM hyperparameters are serialized separately,
so a checkpoint always contains the Adam moments. The perturbation is transient
and is never present in parameters returned to the caller.
"""

import math

import torch
from torch.nn.modules.batchnorm import _BatchNorm

SAM_EPSILON = 1e-12


class NonFiniteSAMStep(RuntimeError):
    """A loss or gradient norm was nonfinite; the update was refused."""


class SharpnessAwareStep:
    def __init__(self, model, base_optimizer, scheduler, criterion, *, rho):
        if not math.isfinite(rho) or rho < 0:
            raise ValueError("SAM radius rho must be finite and nonnegative")
        self.model, self.optimizer, self.scheduler, self.criterion = model, base_optimizer, scheduler, criterion
        self.rho = float(rho)
        self.params = [p for group in base_optimizer.param_groups for p in group["params"]]
        # Collect normalization layers once; upstream traverses the model twice per update.
        self.batch_norms = [module for module in model.modules() if isinstance(module, _BatchNorm)]
        self.momenta = [module.momentum for module in self.batch_norms]
        if any(momentum is None for momentum in self.momenta):
            raise ValueError("Cumulative BatchNorm averaging (momentum=None) is not supported by this BN policy")
        # Reused perturbation backups avoid a fresh allocation for every update.
        self._backup = [torch.empty_like(p, memory_format=torch.preserve_format) for p in self.params]
        self.perturbed = False

    def state_dict(self):
        return {"rho": self.rho, "adaptive": False, "epsilon": SAM_EPSILON,
                "perturbation": "rho*g/(global_l2_norm(g)+epsilon), applied to every parameter with a gradient",
                "batchnorm_policy": "first pass updates running statistics; perturbed pass uses momentum 0; "
                                    "original momenta restored after every update and on failure",
                "updates_per_batch": {"forward_backward_passes": 2, "base_optimizer_steps": 1, "scheduler_steps": 1}}

    def load_state_dict(self, state):
        # SAM itself has no persistent tensors; refuse a different recipe instead of silently adopting it.
        if state != self.state_dict():
            raise ValueError("Saved SAM recipe differs from this implementation")

    def _restore_running_stats(self):
        for module, momentum in zip(self.batch_norms, self.momenta):
            module.momentum = momentum

    def _freeze_running_stats(self):
        for module in self.batch_norms:
            module.momentum = 0.0

    def _forward_backward(self, waveforms, labels):
        loss = self.criterion(self.model(waveforms), labels)
        loss.backward()
        return loss.detach()

    def step(self, waveforms, labels):
        """Return detached (unperturbed, perturbed) losses; one host sync for the finite check."""
        if not self.model.training:
            raise RuntimeError("SAM training step requires train mode")
        self.optimizer.zero_grad(set_to_none=True)
        try:
            self._restore_running_stats()
            loss = self._forward_backward(waveforms, labels)
            active = [index for index, p in enumerate(self.params) if p.grad is not None]
            params = [self.params[index] for index in active]
            backup = [self._backup[index] for index in active]
            with torch.no_grad():
                grad_norm = torch.nn.utils.get_total_norm([p.grad for p in params], 2.0, foreach=True)
                torch._foreach_copy_(backup, params)
                self.perturbed = True
                torch._foreach_add_(params, torch._foreach_mul([p.grad for p in params],
                                                               self.rho / (grad_norm + SAM_EPSILON)))
            self.optimizer.zero_grad(set_to_none=True)
            self._freeze_running_stats()
            perturbed_loss = self._forward_backward(waveforms, labels)
            with torch.no_grad():
                torch._foreach_copy_(params, backup)
                self.perturbed = False
                perturbed_norm = torch.nn.utils.get_total_norm(
                    [p.grad for p in self.params if p.grad is not None], 2.0, foreach=True)
                finite = torch.isfinite(torch.stack((loss, perturbed_loss, grad_norm, perturbed_norm))).all()
            if not bool(finite):  # The single deliberate host synchronization per update.
                raise NonFiniteSAMStep("Nonfinite SAM loss or gradient; update refused and run must stop")
            self.optimizer.step()
            self.scheduler.step()
        finally:
            if self.perturbed:
                with torch.no_grad():
                    torch._foreach_copy_(params, backup)
                self.perturbed = False
            self._restore_running_stats()
        self.optimizer.zero_grad(set_to_none=True)
        return loss, perturbed_loss
