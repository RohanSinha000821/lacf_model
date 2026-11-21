"""FP32 execution, source-only fitting and artifact gates for cached LACF-MoE."""
import json
import logging

import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches, FeatureSplit
from audio_deepfake_detection.lacf_moe.model import LACFMoE, TRAINABLE_PARAMETERS
from scripts.lacf_moe import train_fp32 as runner


def prototypes():
    return unit_vectors(torch.randn(8, 512))


def bank():
    return {(d, s): FeatureSplit([(str(i), i % 2) for i in range(8)],
        [f'{d}_{s}_{i}' for i in range(8)], torch.tensor([0, 1] * 4),
        torch.arange(8)[:, None].repeat(1, 3 if s == 'train' else 1),
        torch.randn(8, 13, 768), torch.randn(8, 512))
        for d in runner.FOLDS['f1']['sources'] for s in ('train', 'dev')}


def provenance():
    return {'code_sha256': {}, 'target_access': False,
            'environment': {'cuda_matmul_tf32': False, 'cudnn_tf32': False}}


def observe_linears(model):
    seen = []
    handles = [m.register_forward_hook(lambda m, args, out: seen.append(out.dtype))
               for m in model.modules() if isinstance(m, torch.nn.Linear)]
    return seen, handles


def test_training_is_fp32_even_inside_bf16_autocast():
    model = LACFMoE(prototypes())
    seen, handles = observe_linears(model)
    b = bank()
    loader = FeatureBatches([b[d, 'train'] for d in runner.FOLDS['f1']['sources']],
                            'concat', 'cpu', training=True, seed=1234, batch_size=4, max_batches=2)
    before = model.backend.classifier[-1].weight.detach().clone()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        losses, updates = runner.train_epoch(model, loader, runner.make_optimizer(model), 'cpu')
    for h in handles: h.remove()
    assert seen and set(seen) == {torch.float32}
    assert updates == 2 and losses['semantic'] == losses['bonafide'] == 0
    assert not torch.equal(before, model.backend.classifier[-1].weight)
    for component in [model.layer_fusion.experts, model.layer_fusion.router, model.backend.adapter, model.backend.classifier]:
        assert all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
                   for p in component.parameters())
    assert sum(p.numel() for p in model.parameters()) == TRAINABLE_PARAMETERS == 967338


def test_scoring_is_fp32_not_posthoc_cast_and_reload_exact():
    model = LACFMoE(prototypes()).eval()
    seen, handles = observe_linears(model)
    data = bank()['asv2019', 'dev']
    def loader():
        return FeatureBatches([data], 'concat', 'cpu', training=False, seed=1234, batch_size=4)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        scores, metrics = runner.score_probe(model, loader())
    for h in handles: h.remove()
    assert set(seen) == {torch.float32} and scores.dtype == torch.float32
    assert not torch.equal(scores, scores.bfloat16().float())
    fresh = LACFMoE(model.backend.prototypes)
    fresh.load_state_dict(model.state_dict())
    other, other_metrics = runner.score_probe(fresh, loader())
    assert torch.equal(scores, other) and metrics == other_metrics


def test_rejects_non_fp32_model():
    with pytest.raises(TypeError, match='FP32'):
        runner.assert_fp32(LACFMoE(prototypes()).double())


def test_rejects_non_fp32_cache_inputs():
    model = LACFMoE(prototypes())
    class BadLoader:
        def __len__(self): return 1
        def __iter__(self):
            yield {'wavlm': torch.randn(2, 13, 768).bfloat16(),
                   'clap': torch.randn(2, 512), 'labels': torch.tensor([0, 1])}
    with pytest.raises(TypeError, match='inputs must be FP32'):
        runner.train_epoch(model, BadLoader(), runner.make_optimizer(model), 'cpu')


def test_source_only_fit_exports_reload_verified_scores_and_refuses_overwrite(tmp_path):
    b = bank()
    output = tmp_path / 'f1' / '1234'
    done = runner.fit(b, prototypes(), output, 'f1', provenance(), 'cpu',
                      logging.getLogger('test_fp32'), max_epochs=2)
    assert done['selected_checkpoint_reloaded_and_source_verified'] is True
    config = json.loads((output / 'config.json').read_text())
    assert config['precision'] == runner.PRECISION and config['model_name'] == runner.NAME
    assert config['batch_size'] == 132 and config['optimizer'] == 'AdamW' and config['target_access'] is False
    assert not (output / 'target_scores.csv').exists()
    for domain in runner.FOLDS['f1']['sources']:
        assert (output / f'source_dev_{domain}.csv').exists()
    with pytest.raises(FileExistsError):
        runner.fit(b, prototypes(), output, 'f1', provenance(), 'cpu', logging.getLogger('test_fp32'), max_epochs=2)
    config['precision'] = 'BF16 mixed; sensitive FP32'
    (output / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='precision/identity'):
        runner.verify_completion(output, 'f1')


def test_refuses_target_partition_in_fit(tmp_path):
    b = bank()
    b['speechfake', 'dev'] = b['asv2019', 'dev']
    with pytest.raises(ValueError, match='Only this fold source'):
        runner.fit(b, prototypes(), tmp_path / 'f1' / '1234', 'f1', provenance(), 'cpu', logging.getLogger('test_fp32'))


def test_rejects_gradient_accumulation_recipe_change():
    model = LACFMoE(prototypes())
    with pytest.raises(ValueError, match='accumulation1'):
        runner.train_epoch(model, [None], runner.make_optimizer(model), 'cpu', accumulation_steps=2)
