"""Loss formulas, unchanged MoE, all-fold fitting/reload and strict loss metadata."""
import hashlib
import json
import logging

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from audio_deepfake_detection.experiments.lacf_moe_ablations import loss_model as models
from audio_deepfake_detection.lacf.model import js_divergence, unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches, FeatureSplit, score_probe
from audio_deepfake_detection.lacf_moe.model import LACFMoE
from scripts.experiments.lacf_moe_ablations import train as core
from scripts.experiments.lacf_moe_ablations import evaluate as core_eval
from scripts.experiments.lacf_moe_ablations import train_losses as runner
from scripts.experiments.lacf_moe_ablations import evaluate_losses as evaluator


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def prototypes():
    return unit_vectors(torch.randn(8, 512))


def bank(fold):
    result = {}
    for domain in core.FOLDS[fold]['sources']:
        for split in ('train', 'dev'):
            n = 7; labels = torch.arange(n) % 2
            records = [(str(i), int(label)) for i, label in enumerate(labels)]
            result[domain, split] = FeatureSplit(records, [f'{domain}_{split}_{n-i}' for i in range(n)], labels,
                torch.arange(n)[:, None].repeat(1, 3 if split == 'train' else 1),
                torch.randn(n, 13, 768), torch.randn(n, 512))
    return result


def proof(tmp, fold, variant):
    p = tmp / f'proof_{variant}_{fold}.json'
    core.save_json(p, dict(passed=True, fold=fold, variant=variant, device='cpu',
        trainable_parameters=967338, full_source_dev_reload_bit_exact=True,
        probe_weights_used_for_final_training=False, target_access=False,
        code_sha256={}, cache_completion_sha256=None))
    return dict(code_sha256={}, source_validation_record=str(p), source_validation_sha256=core.checkpoint_digest(p))


@pytest.mark.parametrize('variant', models.VARIANTS)
@pytest.mark.parametrize('labels', [torch.tensor([0, 1, 0, 1]), torch.ones(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)])
def test_original_loss_terms_fp32_formulas_and_finite_gradients(variant, labels):
    model = models.build_model(variant, prototypes())
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = model({'wavlm': torch.randn(4, 13, 768), 'clap': torch.randn(4, 512)})
        losses = model.objective(out, labels)
    a, b = models.WEIGHTS[variant]
    detection = F.binary_cross_entropy_with_logits(out['logit'].float(), labels.float())
    semantic = F.binary_cross_entropy(out['p_w'][:, 3:].sum(-1).clamp(0, 1), labels.float())
    mask = labels == 0
    bf = (js_divergence(out['p_w'], out['p_c']) * mask).sum() / mask.sum().clamp_min(1)
    torch.testing.assert_close(losses['total'], detection + a * semantic + b * bf)
    assert all(v.dtype == torch.float32 for v in losses.values())
    if not a: assert losses['semantic'] == 0
    if not b or not mask.any(): assert losses['bonafide'] == 0
    losses['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 967338


def test_same_architecture_initialization_rng_and_predictions_as_bce_reference():
    fixed = prototypes(); batch = {'wavlm': torch.randn(5, 13, 768), 'clap': torch.randn(5, 512)}
    torch.manual_seed(1234); reference = LACFMoE(fixed).eval(); rng = torch.get_rng_state()
    for variant in models.VARIANTS:
        torch.manual_seed(1234); model = models.build_model(variant, fixed).eval()
        assert torch.equal(rng, torch.get_rng_state())
        assert model.state_dict().keys() == reference.state_dict().keys()
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in reference.state_dict().items())
        with torch.autocast('cpu', dtype=torch.bfloat16):
            expected, actual = reference(batch), model(batch)
        assert all(torch.equal(v, actual[k]) for k, v in expected.items())
        cfg = model.configuration()
        assert cfg['objective'] == models.OBJECTIVES[variant]
        assert (cfg['backend']['semantic_weight'], cfg['backend']['bonafide_weight']) == models.WEIGHTS[variant]
        assert not cfg['foundation_encoders_loaded']


@pytest.mark.parametrize('variant', models.VARIANTS)
@pytest.mark.parametrize('fold', tuple(core.FOLDS))
def test_all_folds_fresh_fit_reload_export_loss_config_and_tamper(tmp_path, variant, fold):
    data = bank(fold); run = tmp_path / f'{variant}_{models.NAMES[variant]}' / fold / '1234'
    provenance = proof(tmp_path, fold, variant)
    with runner.loss_controls():
        done = core.fit(data, prototypes(), run, fold, variant, provenance,
                        'cpu', logging.getLogger('test_loss'), max_epochs=1, patience=1)
        assert done['selected_checkpoint_reloaded_and_source_verified'] and not done['target_access']
        assert core.verify_completion(run, fold, variant, core.labels_for(data, fold)) == done
        cfg = json.loads((run / 'config.json').read_text())
        assert cfg['model']['objective'] == models.OBJECTIVES[variant]
        with pytest.raises(FileExistsError):
            core.fit(data, prototypes(), run, fold, variant, provenance, 'cpu',
                     logging.getLogger('retry'), max_epochs=1)
        # Even internally rehashed metadata cannot silently change a loss weight.
        cfg['model']['backend']['semantic_weight'] = 99.
        (run / 'config.json').write_text(json.dumps(cfg))
        digest = core.checkpoint_digest(run / 'config.json')
        done['config_sha256'] = digest
        state = torch.load(run / 'best.pt', weights_only=True)
        state['config_sha256'] = digest; torch.save(state, run / 'best.pt')
        done['best_checkpoint_sha256'] = core.checkpoint_digest(run / 'best.pt')
        (run / 'training_complete.json').write_text(json.dumps(done))
        with pytest.raises(ValueError, match='Loss ablation architecture or weights'):
            core.verify_completion(run, fold, variant, core.labels_for(data, fold))


def test_registry_restoration_after_failure_preserves_guarded_files():
    names = ('CORE', 'NAMES', 'PARAMETERS', 'FEATURE_INPUTS', 'build_model', 'FAMILY', 'verify_completion')
    old = {k: getattr(core, k) for k in names}
    old_eval = {k: getattr(core_eval, k) for k in ('CORE', 'NAMES', 'build_model', 'FEATURE_INPUTS', 'OUTPUT', 'LAUNCH')}
    paths = [core.__file__, core_eval.__file__]
    before = {p: hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in paths}
    with pytest.raises(RuntimeError):
        with evaluator.loss_evaluation():
            assert core.CORE[1:] == core_eval.CORE[1:] == models.VARIANTS
            assert core.verify_completion is runner.verify_completion
            raise RuntimeError('test restoration')
    assert all(getattr(core, k) is v for k, v in old.items())
    assert all(getattr(core_eval, k) is v for k, v in old_eval.items())
    assert before == {p: hashlib.sha256(open(p, 'rb').read()).hexdigest() for p in paths}


@pytest.mark.parametrize('variant', models.VARIANTS)
def test_bf16_target_batch132_tail_parity(variant):
    n = 135
    data = FeatureSplit([], list(map(str, range(n))), torch.arange(n) % 2, torch.arange(n)[:, None],
                        torch.randn(n, 13, 768), torch.randn(n, 512))
    model = models.build_model(variant, prototypes()).eval()
    expected, _ = score_probe(model, FeatureBatches([data], 'concat', 'cpu', training=False, seed=1234))
    actual = core_eval.score_arrays(model, {'wavlm_layers': data.wavlm.numpy(),
                                          'clap_projected': data.clap.numpy()}, 'cpu')
    assert np.array_equal(expected.numpy(), actual)
    assert np.array_equal(actual, torch.from_numpy(actual).to(torch.bfloat16).float().numpy())


def test_wait_requires_verified_producer_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, 'TRAIN_LAUNCH', tmp_path)
    run = tmp_path / 'C1/f1/1234'; run.mkdir(parents=True)
    (run / 'training_complete.json').write_text('{}')
    (tmp_path / 'exit.json').write_text('{"process_exit":1}')
    with pytest.raises(RuntimeError, match='before C1/f1 verified'):
        evaluator.wait_for_run(run, 'C1/f1', logging.getLogger('wait'))
    core.save_json(tmp_path / 'queue_status.json', {'runs': [{'variant': 'C1', 'fold': 'f1',
        'completion_sha256': core.checkpoint_digest(run / 'training_complete.json')}]})
    evaluator.wait_for_run(run, 'C1/f1', logging.getLogger('wait'))
    (run / 'training_complete.json').write_text('changed')
    with pytest.raises(ValueError, match='completion changed'):
        evaluator.wait_for_run(run, 'C1/f1', logging.getLogger('wait'))
