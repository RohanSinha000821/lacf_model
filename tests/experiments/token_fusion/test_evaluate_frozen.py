"""Synthetic frozen TF2 evaluation through real source gates and shared metrics."""

import hashlib
import json
import logging

import numpy as np
import pytest

from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from scripts.token_fusion import evaluate_frozen as evaluator
from scripts.token_fusion import train_all_folds as trainer
from scripts.experiments.layer_fusion import evaluate_four_experts_cached as cache_reader
from test_frozen_all_folds import bank


def write(path, value):
    path.write_text(json.dumps(value))


def execution(run, producer, fold):
    return {'code_sha256': {},
            'checkpoint_sha256': {fold: checkpoint_digest(run / 'best.pt')},
            'training_completion_sha256': {fold: checkpoint_digest(run / 'training_complete.json')},
            'producer_completion_sha256': {fold: checkpoint_digest(producer / 'evaluation_complete.json')}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Produce real small TF2 fits; replace only local metadata/cache readers."""
    data = bank()
    cache = tmp_path / 'cache'
    cache.mkdir()
    write(cache / 'config.json', {})
    write(cache / 'execution_accelerated.json', {'pretrained': {}})
    write(cache / 'complete.json', {'splits': {f'{d}/dev': {} for d in ('asv2019', 'asv5', 'cfad', 'speechfake')}})
    monkeypatch.setattr(trainer, 'CACHE', cache)
    monkeypatch.setattr(evaluator, 'load_split', lambda c, d, s, m, r, logger=None: data[d, s])
    provenance = {'code_sha256': {}, 'preserved_f1_sha256': {},
                  'cache_sha256': {n: checkpoint_digest(cache / n) for n in
                                   ('config.json', 'complete.json', 'execution_accelerated.json')}}

    def make(fold):
        subset = trainer.source_bank(data, fold)
        run = tmp_path / 'training' / fold / '1234'
        with trainer.fold_context(fold) as original:
            original.fit_run(subset, run, 'TF2', 1234, None, provenance, 'cpu', logging.getLogger('fit'))
        output = tmp_path / 'evaluation' / fold / '1234'
        spec = FOLDS[fold]
        target = {f'target{i:03}': i % 2 for i in range(144)}
        def canonical(domain, split, root):
            if domain == spec['target']:
                assert split == spec['target_split']
                assert (output / 'source_validation.json').is_file(), 'target opened before fresh source validation'
                return target
            assert split == 'dev'
            ds = data[domain, 'dev']
            return dict(zip(ds.ids, ds.labels.tolist()))
        import scripts.evaluate_scores as shared
        monkeypatch.setattr(evaluator, 'canonical_score_labels', canonical)
        monkeypatch.setattr(cache_reader, 'canonical_score_labels', canonical)
        monkeypatch.setattr(shared, 'canonical_score_labels', canonical)
        producer = tmp_path / 'producer' / fold / '1234'
        chunk = producer / 'target_features/chunk_00000'
        chunk.mkdir(parents=True)
        write(producer / 'config.json', {'provenance': {'cache_completion_sha256': checkpoint_digest(cache / 'complete.json')}})
        write(producer / 'training_complete.json', {'config_sha256': checkpoint_digest(producer / 'config.json')})
        inference = {'input': 'centered common interval up to10s; native unpadded WavLM; CLAP repeatpad',
                     'head_batch': 132, 'wavlm_batch': 32, 'clap_batch': 64, 'chunk': 1024,
                     'workers': 8, 'prefetch': 2, 'precision': 'BF16 mixed; sensitive FP32', 'score': 'scalar spoof logit'}
        write(producer / 'evaluation_config.json', {'model_name': 'LACF-MoE', 'fold': fold, 'seed': 1234,
              'target': spec['target'] + '/' + spec['target_split'], 'source_cache': str(cache),
              'frozen_encoder_revisions': {}, 'inference': inference})
        rows = [{'index': i, 'id': f"{spec['target']}/{spec['target_split']}/{key}",
                 'label': label, 'frames': 160000, 'rate': 16000,
                 'intervals': [[0, 160000]], 'view_rows': [0]} for i, (key, label) in enumerate(target.items())]
        write(chunk / 'records.json', rows)
        ds = next(iter(data.values()))
        for name, values in [('wavlm_layers', ds.wavlm[:6].numpy()), ('clap_projected', ds.clap[:6].numpy())]:
            np.save(chunk / f'{name}.npy', np.tile(values, (24,) + (1,) * (values.ndim - 1)), allow_pickle=False)
        write(chunk / 'complete.json', {'records': 144, 'files_sha256': {
              n: checkpoint_digest(chunk / n) for n in ('records.json', 'wavlm_layers.npy', 'clap_projected.npy')}})
        features = producer / 'target_features'
        write(features / 'complete.json', {'records': 144, 'chunks': [{'path': 'chunk_00000',
              'completion_sha256': checkpoint_digest(chunk / 'complete.json')}],
              'canonical_ids_labels_sha256': hashlib.sha256(json.dumps(list(target.items())).encode()).hexdigest(),
              'evaluation_config_sha256': checkpoint_digest(producer / 'evaluation_config.json')})
        digest = checkpoint_digest(features / 'complete.json')
        write(producer / 'score_manifest.json', {'target_features_sha256': digest})
        write(producer / 'metrics.json', {})
        write(producer / 'source_validation.json', {'passed': True})
        (producer / 'best.pt').write_bytes(b'completed unrelated MoE checkpoint')
        (producer / 'target_scores.csv').write_text('preserved producer scores')
        write(producer / 'evaluation_complete.json', {'fold': fold, 'seed': 1234,
              'target_features_sha256': digest, 'source_selected_checkpoint_only': True,
              'metrics_sha256': checkpoint_digest(producer / 'metrics.json'),
              'source_validation_sha256': checkpoint_digest(producer / 'source_validation.json'),
              'checkpoint_sha256': checkpoint_digest(producer / 'best.pt'),
              'score_sha256': {'target_scores.csv': checkpoint_digest(producer / 'target_scores.csv')}})
        return run, output, producer, target
    return make


@pytest.mark.parametrize('fold', ('f2', 'f3', 'f4'))
def test_full_source_reload_cached_target_tail_and_shared_metrics(setup, tmp_path, fold):
    run, output, producer, target = setup(fold)
    before = {p.name: checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    result = evaluator.evaluate(run, output, producer, fold, execution(run, producer, fold), tmp_path,
                                'cpu', logging.getLogger('evaluate'), bootstrap_resamples=5)
    assert result['model_family'] == 'token_fusion'
    assert result['fold'] == fold and result['seed'] == 1234
    assert result['target']['n_bona_fide'] + result['target']['n_spoof'] == 144
    for branch in ('strict_raw_transfer', 'unlabeled_zscore_transfer'):
        assert set(result[branch]['target_operating_points']) == {'1%', '5%', '10%'}
    assert result['unlabeled_zscore_transfer']['transductive'] is True
    assert result['confidence_intervals']['resamples'] == 5
    assert evaluator.read(output / 'evaluation_config.json')['encoder_execution'] is False
    assert evaluator.read(output / 'source_validation.json')['full_source_score_bit_exact'] is True
    assert evaluator.read(output / 'evaluation_complete.json')['metrics_sha256'] == checkpoint_digest(output / 'metrics.json')
    assert before == {p.name: checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        evaluator.evaluate(run, output, producer, fold, {}, tmp_path, 'cpu', logging.getLogger('retry'))


def test_failed_source_reload_prevents_all_target_access(setup, tmp_path, monkeypatch):
    run, output, producer, _ = setup('f2')
    def failure(*args, **kwargs):
        raise ValueError('source reload differs')
    def no_target(*args, **kwargs):
        pytest.fail('target producer accessed after source failure')
    monkeypatch.setattr(evaluator, 'source_reload', failure)
    monkeypatch.setattr(evaluator, 'verify_producer', no_target)
    with pytest.raises(ValueError, match='source reload differs'):
        evaluator.evaluate(run, output, producer, 'f2', execution(run, producer, 'f2'), tmp_path, 'cpu', logging.getLogger('fail'))
    assert not output.exists()


@pytest.mark.parametrize('artifact', ('best.pt', 'source_dev_asv2019.csv', 'source_calibration.json'))
def test_training_artifact_tampering_stops_before_output(setup, tmp_path, artifact):
    run, output, producer, _ = setup('f2')
    with (run / artifact).open('ab') as f:
        f.write(b'changed')
    with pytest.raises((ValueError, RuntimeError)):
        evaluator.evaluate(run, output, producer, 'f2', execution(run, producer, 'f2'), tmp_path, 'cpu', logging.getLogger('tamper'))
    assert not output.exists()


def test_producer_completion_cannot_be_bypassed(setup, tmp_path):
    run, output, producer, _ = setup('f2')
    (producer / 'metrics.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='producer artifact'):
        evaluator.evaluate(run, output, producer, 'f2', execution(run, producer, 'f2'), tmp_path, 'cpu', logging.getLogger('producer'))
    assert (output / 'source_validation.json').is_file()
    assert not (output / 'target_scores.csv').exists()
    assert not (output / 'evaluation_complete.json').exists()


def test_changed_producer_completion_fails_launch_binding(setup, tmp_path):
    run, output, producer, _ = setup('f2')
    bound = execution(run, producer, 'f2')
    done = evaluator.read(producer / 'evaluation_complete.json')
    done['new_revision'] = True
    write(producer / 'evaluation_complete.json', done)
    with pytest.raises(ValueError, match='Launch-bound target-cache'):
        evaluator.evaluate(run, output, producer, 'f2', bound, tmp_path, 'cpu', logging.getLogger('binding'))
    assert not (output / 'target_scores.csv').exists()
