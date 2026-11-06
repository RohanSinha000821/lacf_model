"""CPU-only checks for the isolated top-four/BF16 SSL adaptation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from audio_deepfake_detection.sota.ssl_gating import SSLFeatureGating
from audio_deepfake_detection.sota.ssl_gating_partial import PartialSSLFeatureGating, detection_losses, partial_scores
from scripts.ssl_gating.train_partial import save_checkpoint, source_metrics


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))
        self._requires_grad = True

    def _freeze_parameters(self):
        self.requires_grad_(False)
        self._requires_grad = False


class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1024, num_hidden_layers=24)
        self.feature_extractor = Extractor()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(8,8) for _ in range(24)])
        self.grad_flags = []

    def forward(self, waves, *, output_hidden_states):
        x = waves[:,::100,None] * self.feature_extractor.weight
        states = [x.repeat(1,1,128)]
        for layer in self.encoder.layers:
            x = x + .1*torch.tanh(layer(x))
            states.append(x.repeat(1,1,128))
        self.grad_flags = [v.requires_grad for v in states]
        return SimpleNamespace(hidden_states=states)


def test_partial_freezes_exact_prefix_without_changing_initial_weights_or_fp32_math():
    torch.manual_seed(1234)
    original = SSLFeatureGating(Frontend()).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(1234)
    partial = PartialSSLFeatureGating(Frontend()).eval()
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in original.state_dict().items():
        assert torch.equal(value,partial.state_dict()[key]),key
    waves = torch.randn(2,800)
    with torch.inference_mode():
        torch.testing.assert_close(original(waves),partial(waves),rtol=0,atol=0)


def test_bf16_has_fp32_pooling_losses_and_only_last_four_and_backend_gradients():
    torch.manual_seed(1234)
    model = PartialSSLFeatureGating(Frontend()).train()
    assert not model.feature_extractor.feature_extractor._requires_grad
    # Preserve original stochastic train policy rather than silently disabling encoder regularization.
    assert all(layer.training for layer in model.feature_extractor.encoder.layers)
    enabled = {n for n,p in model.named_parameters() if p.requires_grad}
    encoder = {n for n in enabled if n.startswith('feature_extractor.')}
    assert encoder == {f'feature_extractor.encoder.layers.{i}.{p}' for i in range(20,24) for p in ('weight','bias')}
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    pooled = []
    hook = model.pooling.register_forward_hook(lambda _,inputs,output:pooled.append(output.dtype))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-6,weight_decay=1e-4)
    loss,ce,cka = detection_losses(model,torch.randn(5,800),torch.tensor([0,1,0,1,1]))
    assert loss.dtype == ce.dtype == cka.dtype == torch.float32 and pooled==[torch.float32]
    assert model.feature_extractor.grad_flags == [False]*21+[True]*4
    loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if n in frozen)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if n in enabled)
    assert all(model.feature_extractor.encoder.layers[i].weight.grad.norm()>0 for i in range(20,24))
    optimizer.step()
    assert all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen)
    hook.remove()
    model.eval()
    scores = partial_scores(model,torch.randn(2,800))
    assert scores.dtype == torch.float32 and torch.isfinite(scores).all()


def test_partial_checkpoint_reload_retains_freeze_policy_and_exact_scores(tmp_path):
    torch.manual_seed(3)
    model = PartialSSLFeatureGating(Frontend()).eval()
    wave = torch.randn(2,800)
    expected = partial_scores(model,wave)
    path = tmp_path/'recovery.pt'
    save_checkpoint(path,{'model_state_dict':model.state_dict(),'kind':'intermediate_training_snapshot_not_completion'})
    restored = PartialSSLFeatureGating(Frontend()).eval()
    restored.load_state_dict(torch.load(path,weights_only=True)['model_state_dict'],strict=True)
    torch.testing.assert_close(partial_scores(restored,wave),expected,rtol=0,atol=0)
    assert not restored.feature_extractor.encoder.layers[19].weight.requires_grad
    assert restored.feature_extractor.encoder.layers[20].weight.requires_grad
    assert not (tmp_path/'training_complete.json').exists()
    before = path.read_bytes()
    path.with_suffix('.pt.tmp').write_text('interrupted')
    with pytest.raises(FileExistsError,match='Preserving'):
        save_checkpoint(path,{})
    assert path.read_bytes()==before


def test_source_macro_gives_equal_mass_to_domains_and_includes_auroc():
    scored={'small':(np.array([0,1]),np.array([-1.,1.])),
            'large':(np.tile([0,1],50),np.tile([1.,-1.],50))}
    metrics=source_metrics(scored)
    assert metrics['macro_eer']==.5
    assert metrics['by_domain']['small']['auroc']==1.
    assert metrics['by_domain']['large']['auroc']==0.
