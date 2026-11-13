"""Minimal CPU checks for the last-block-only SSL adaptation."""
import torch

from audio_deepfake_detection.sota.ssl_gating_last import LastSSLFeatureGating, LAST_MODEL_CONFIG
from audio_deepfake_detection.sota.ssl_gating_partial import PartialSSLFeatureGating, detection_losses, partial_scores
from test_ssl_gating_partial import Frontend


def test_same_initial_weights_rng_and_scores_as_existing_partial():
    torch.set_num_threads(2);torch.manual_seed(1234)
    reference=PartialSSLFeatureGating(Frontend()).eval();state=torch.get_rng_state()
    torch.manual_seed(1234);model=LastSSLFeatureGating(Frontend()).eval()
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in reference.state_dict().items())
    waves=torch.randn(2,800)
    assert torch.equal(partial_scores(reference,waves),partial_scores(model,waves))
    assert LAST_MODEL_CONFIG['trainable_encoder_blocks']==[23]


def test_only_last_block_and_backend_gradients_frozen_weights_unchanged():
    torch.manual_seed(1234);model=LastSSLFeatureGating(Frontend()).train()
    enabled={n for n,p in model.named_parameters() if p.requires_grad}
    assert {n for n in enabled if n.startswith('feature_extractor.')}=={
        'feature_extractor.encoder.layers.23.weight','feature_extractor.encoder.layers.23.bias'}
    assert not model.feature_extractor.feature_extractor._requires_grad
    assert all(layer.training for layer in model.feature_extractor.encoder.layers)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-6,weight_decay=1e-4)
    loss,ce,cka=detection_losses(model,torch.randn(5,800),torch.tensor([0,1,0,1,1]))
    assert loss.dtype==ce.dtype==cka.dtype==torch.float32
    assert model.feature_extractor.grad_flags==[False]*24+[True]
    loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    optimizer.step()
    assert all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen)


def test_fresh_reload_retains_last_block_freeze_and_exact_scores(tmp_path):
    model=LastSSLFeatureGating(Frontend()).eval();waves=torch.randn(2,800)
    expected=partial_scores(model,waves)
    torch.save(model.state_dict(),tmp_path/'state.pt')
    fresh=LastSSLFeatureGating(Frontend()).eval()
    fresh.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True),strict=True)
    assert torch.equal(expected,partial_scores(fresh,waves))
    assert not fresh.feature_extractor.encoder.layers[22].weight.requires_grad
    assert fresh.feature_extractor.encoder.layers[23].weight.requires_grad
