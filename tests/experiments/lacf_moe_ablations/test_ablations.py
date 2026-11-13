"""CPU-only component, training, source-isolation and failure-gate checks."""

import json
import logging

import pytest
import torch

from audio_deepfake_detection.experiments.lacf_moe_ablations.model import (
    CORE, FEATURE_INPUTS, PARAMETERS, FusionMoE, build_model,
)
from audio_deepfake_detection.experiments.layer_fusion.model import LayerFusionDetector
from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches, FeatureSplit, score_probe
from audio_deepfake_detection.lacf_moe.model import LACFMoE
from scripts.experiments.lacf_moe_ablations import train as runner


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def prototypes():
    return unit_vectors(torch.randn(8, 512))


def bank(fold='f1'):
    return {(d, s): FeatureSplit([(str(i), i % 2) for i in range(7)],
        [f'{d}_{s}_{6-i}' for i in range(7)], torch.tensor([0, 1, 0, 1, 0, 1, 0]),
        torch.arange(7)[:, None].repeat(1, 3 if s == 'train' else 1),
        torch.randn(7, 13, 768), torch.randn(7, 512))
        for d in runner.FOLDS[fold]['sources'] for s in ('train', 'dev')}


def fit_provenance(tmp_path, fold, variant):
    path = tmp_path / f'proof_{variant}_{fold}.json'
    runner.save_json(path, {'passed': True, 'fold': fold, 'variant': variant, 'device': 'cpu',
        'trainable_parameters': PARAMETERS[variant], 'full_source_dev_reload_bit_exact': True,
        'probe_weights_used_for_final_training': False, 'target_access': False,
        'code_sha256': {}, 'cache_completion_sha256': None})
    return {'code_sha256': {}, 'source_validation_record': str(path),
            'source_validation_sha256': runner.checkpoint_digest(path)}


def test_reference_and_uniform_layer_control_preserve_existing_math():
    fixed = prototypes(); batch = {'wavlm': torch.randn(7, 13, 768), 'clap': torch.randn(7, 512)}
    for variant, factory in (('A0', lambda: LACFMoE(fixed)),
                             ('A4', lambda: LayerFusionDetector('C1', fixed))):
        torch.manual_seed(1234); old = factory().eval()
        torch.manual_seed(1234); new = build_model(variant, fixed).eval()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            left, right = old(batch), new(batch)
        for key in ('features', 'logit', 'p_w', 'p_c'):
            assert torch.equal(left[key], right[key]), (variant, key)
        if variant == 'A0':
            assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())


@pytest.mark.parametrize('variant', CORE[1:])
def test_configuration_shapes_finite_gradients_and_only_enabled_parameters(variant):
    fixed = prototypes(); model = build_model(variant, fixed)
    batch = {'labels': torch.tensor([0, 1, 0, 1, 0])}
    if variant != 'A1': batch['wavlm'] = torch.randn(5, 13, 768)
    if variant != 'A3': batch['clap'] = torch.randn(5, 512)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(batch); losses = model.objective(output, batch['labels'])
    assert output['features'].shape == (5, {'A1': 512, 'A2': 1024, 'A3': 8}.get(variant, 35))
    assert output['logit'].shape == (5,) and output['logit'].dtype == torch.bfloat16
    assert losses['total'].dtype == torch.float32 and losses['semantic'] == losses['bonafide'] == 0
    losses['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(not b.requires_grad for b in model.buffers())
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[variant]
    config = model.configuration()
    assert config['variant'] == variant and not config['foundation_encoders_loaded']
    assert config['clap_audio_enabled'] == (variant != 'A3')
    assert config['wavlm_enabled'] == (variant != 'A1')
    assert not config['semantic_loss'] and not config['bonafide_loss']
    if variant == 'A1': assert not hasattr(model, 'adapter') and not hasattr(model, 'layer_fusion')
    if variant in ('A1', 'A2'): assert not hasattr(model, 'prototypes')


def test_fusion_global_control_matches_experts_and_uniform_initial_predictions():
    fixed = prototypes()
    torch.manual_seed(1234); dynamic = build_model('A5', fixed).eval(); rng = torch.random.get_rng_state()
    torch.manual_seed(1234); global_model = build_model('A6', fixed).eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    for a, b in zip(dynamic.fusion_moe.experts, global_model.fusion_moe.experts):
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    batch = {'wavlm': torch.randn(7, 13, 768), 'clap': torch.randn(7, 512)}
    with torch.autocast('cpu', dtype=torch.bfloat16): left, right = dynamic(batch), global_model(batch)
    assert torch.equal(left['fusion_routing'], torch.full((7, 3), 1/3))
    assert torch.equal(left['logit'], right['logit'])
    assert PARAMETERS['A5'] - PARAMETERS['A6'] == 175
    with torch.no_grad(): dynamic.fusion_moe.router[-1].weight[0, 0] = 2
    assert not torch.equal(dynamic(batch)['fusion_routing'][0], dynamic(batch)['fusion_routing'][1])


def test_fusion_weighted_sum_includes_residuals():
    fusion = FusionMoE()
    for expert in fusion.experts:
        for p in expert.parameters(): p.data.zero_()
    values = torch.randn(5, 35)
    result, routing = fusion(values)
    assert torch.allclose(result, values) and torch.equal(routing, torch.full((5, 3), 1/3))


@pytest.mark.parametrize('variant', ('A0', *CORE[1:]))
def test_scoring_matches_shared_scorer_and_route_counts(variant):
    data = bank(); model = build_model(variant, prototypes())
    values, metrics, routes = runner.score_sources(model, data, 'f1', variant, 'cpu', collect_routing=True)
    for domain in runner.FOLDS['f1']['sources']:
        expected, expected_metrics = score_probe(model, FeatureBatches([data[domain, 'dev']],
            FEATURE_INPUTS[variant], 'cpu', training=False, seed=1234))
        assert torch.equal(values[domain], expected) and metrics['by_domain'][domain] == expected_metrics
        for route in routes[domain].values():
            assert route['all']['count'] == 7 and route['bonafide']['count'] == 4 and route['spoof']['count'] == 3
            assert sum(route['all']['top_expert_counts']) == 7


@pytest.mark.parametrize('fold', tuple(runner.FOLDS))
@pytest.mark.parametrize('variant', CORE[1:])
def test_each_control_fits_reloads_exports_and_calibrates_all_folds(tmp_path, fold, variant):
    data = bank(fold); path = tmp_path / variant / fold / '1234'
    provenance = fit_provenance(tmp_path, fold, variant)
    result = runner.fit(data, prototypes(), path, fold, variant, provenance, 'cpu',
                        logging.getLogger('ablation_test'), max_epochs=2, patience=1)
    assert result['selected_checkpoint_reloaded_and_source_verified'] and result['target_access'] is False
    assert runner.verify_completion(path, fold, variant, runner.labels_for(data, fold)) == result
    config = runner.read(path / 'config.json')
    assert config['sources'] == list(runner.FOLDS[fold]['sources']) and config['seed_plan'] == [1234]
    assert config['excluded_target'] == runner.FOLDS[fold]['target']
    with pytest.raises(FileExistsError):
        runner.fit(data, prototypes(), path, fold, variant, provenance, 'cpu', logging.getLogger('repeat'))
    (path / 'source_calibration.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        runner.verify_completion(path, fold, variant, runner.labels_for(data, fold))


def test_wrong_source_or_stale_validation_never_creates_run(tmp_path):
    provenance = fit_provenance(tmp_path, 'f2', 'A5'); path = tmp_path / 'f2/1234'
    with pytest.raises(ValueError, match='Only this fold source'):
        runner.fit(bank('f1'), prototypes(), path, 'f2', 'A5', provenance, 'cpu', logging.getLogger('wrong'))
    Path = type(tmp_path)
    Path(provenance['source_validation_record']).write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        runner.fit(bank('f2'), prototypes(), path, 'f2', 'A5', provenance, 'cpu', logging.getLogger('stale'))
    assert not path.exists()


def test_reload_mismatch_stops_without_completion(tmp_path, monkeypatch):
    original = runner.score_sources; calls = 0
    def changed(*args, **kwargs):
        nonlocal calls
        values, metrics, routes = original(*args, **kwargs); calls += 1
        if calls == 2:
            domain = next(iter(values)); values[domain] = values[domain].clone(); values[domain][0] += 1
        return values, metrics, routes
    monkeypatch.setattr(runner, 'score_sources', changed)
    path = tmp_path / 'f1/1234'; provenance = fit_provenance(tmp_path, 'f1', 'A5')
    with pytest.raises(ValueError, match='Fresh selected-checkpoint'):
        runner.fit(bank(), prototypes(), path, 'f1', 'A5', provenance, 'cpu', logging.getLogger('bad'), max_epochs=1)
    assert not (path / 'training_complete.json').exists()


@pytest.mark.parametrize('failure', ('preflight', 'fit', 'verify'))
def test_queue_stops_on_failure_and_additional_controls_never_run(tmp_path, monkeypatch, failure):
    data = {k: v for fold in runner.FOLDS for k, v in bank(fold).items()}; calls = []
    actual_fit = runner.fit
    def check(stage):
        def stub(*args, **kwargs):
            calls.append(stage)
            if stage == failure: raise ValueError(stage)
            return {'passed': True}
        return stub
    monkeypatch.setattr(runner, 'preflight', check('preflight'))
    monkeypatch.setattr(runner, 'fit', check('fit'))
    monkeypatch.setattr(runner, 'verify_completion', check('verify'))
    with pytest.raises(ValueError, match=failure):
        runner.run_queue(data, prototypes(), {}, 'cpu', tmp_path, logging.getLogger('queue'))
    assert calls == ['preflight', 'fit', 'verify'][:['preflight', 'fit', 'verify'].index(failure) + 1]
    assert tuple(CORE) == tuple(f'A{i}' for i in range(7))
    for variant in ('B1', 'A0', 'A7'):
        with pytest.raises(ValueError):
            actual_fit(data, prototypes(), tmp_path / variant, 'f1', variant, {}, 'cpu', logging.getLogger('held'))


def test_queue_success_has_exactly_24_verified_fits_in_fixed_order(tmp_path, monkeypatch):
    data = {k: v for fold in runner.FOLDS for k, v in bank(fold).items()}; calls = []
    monkeypatch.setattr(runner, 'FAMILY', tmp_path / 'runs')
    def preflight(subset, fixed, fold, variant, *args):
        runner.assert_bank(subset, fold); calls.append(('preflight', variant, fold)); return {'passed': True}
    def fit(subset, fixed, path, fold, variant, *args, **kwargs):
        runner.assert_bank(subset, fold); calls.append(('fit', variant, fold))
        path.mkdir(parents=True); (path / 'training_complete.json').write_text('{}')
    def verify(path, fold, variant, labels):
        calls.append(('verify', variant, fold))
    monkeypatch.setattr(runner, 'preflight', preflight); monkeypatch.setattr(runner, 'fit', fit)
    monkeypatch.setattr(runner, 'verify_completion', verify)
    result = runner.run_queue(data, prototypes(), {}, 'cpu', tmp_path, logging.getLogger('success'))
    assert len(result) == 24
    assert calls == [(stage, variant, fold) for variant in CORE[1:] for fold in runner.FOLDS
                     for stage in ('preflight', 'fit', 'verify')]


def test_ties_keep_earlier_checkpoint(tmp_path, monkeypatch):
    original = runner.score_sources
    def tied(*args, **kwargs):
        values, metrics, routes = original(*args, **kwargs)
        for item in metrics['by_domain'].values(): item['eer'] = .5
        metrics['macro_eer'] = .5
        return values, metrics, routes
    monkeypatch.setattr(runner, 'score_sources', tied)
    path = tmp_path / 'f1/1234'; provenance = fit_provenance(tmp_path, 'f1', 'A5')
    # Deliberately fake EERs must eventually fail the independent calibration gate.
    with pytest.raises(ValueError):
        runner.fit(bank(), prototypes(), path, 'f1', 'A5', provenance, 'cpu', logging.getLogger('ties'),
                   max_epochs=6, patience=2)
    state = torch.load(path / 'best.pt', weights_only=True)
    history = [json.loads(line) for line in (path / 'history.jsonl').read_text().splitlines()]
    assert state['epoch'] == 1 and len(history) == 3 and history[-1]['patience'] == 2
