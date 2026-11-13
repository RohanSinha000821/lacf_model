"""CPU-only coverage of frozen TF2 fold isolation, parity and queue gates."""

import json
import logging
from pathlib import Path

import pytest
import torch

from audio_deepfake_detection.lacf.probes import FeatureSplit
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from scripts.token_fusion import train as original
from scripts.token_fusion import train_all_folds as queue


def bank():
    result = {}
    for domain in ('asv2019', 'asv5', 'cfad', 'speechfake'):
        for split in ('train', 'dev'):
            labels = torch.tensor([0, 1, 0, 1, 0, 1])
            rows = torch.arange(18 if split == 'train' else 6).reshape(6, -1)
            result[domain, split] = FeatureSplit(
                [(f'{domain}/{i}.wav', int(labels[i])) for i in range(6)],
                [f'{domain}/{6-i}.wav' for i in range(6)], labels, rows,
                torch.randn(rows.numel(), 13, 768), torch.randn(rows.numel(), 512))
    return result


@pytest.mark.parametrize('fold', FOLDS)
def test_fold_binding_and_source_exclusion(fold):
    before = original.FOLD, original.SOURCES, original.SCOPE, original.SEEDS
    data = queue.source_bank(bank(), fold)
    assert set(data) == {(d, s) for d in FOLDS[fold]['sources'] for s in ('train', 'dev')}
    assert all(d != FOLDS[fold]['target'] for d, _ in data)
    with queue.fold_context(fold) as runner:
        assert runner.FOLD == fold and runner.SEEDS == (1234,)
        assert runner.SOURCES == tuple(FOLDS[fold]['sources'])
        assert f'full {fold.upper()}' in runner.SCOPE
    assert (original.FOLD, original.SOURCES, original.SCOPE, original.SEEDS) == before


def test_binding_restored_on_failure_and_invalid_fold():
    before = original.FOLD, original.SOURCES, original.SCOPE, original.SEEDS
    with pytest.raises(RuntimeError):
        with queue.fold_context('f4'):
            raise RuntimeError('stop')
    assert (original.FOLD, original.SOURCES, original.SCOPE, original.SEEDS) == before
    with pytest.raises(ValueError):
        with queue.fold_context('f5'):
            pass


def test_f1_fitting_is_numerically_identical_to_original(tmp_path):
    data = queue.source_bank(bank(), 'f1')
    logger = logging.getLogger('parity')
    first, second = tmp_path / 'original/f1/1234', tmp_path / 'new/f1/1234'
    original.fit_run(data, first, 'TF2', 1234, None, {'code_sha256': {}},
                     'cpu', logger, max_epochs=1)
    with queue.fold_context('f1') as runner:
        runner.fit_run(data, second, 'TF2', 1234, None, {'code_sha256': {}},
                       'cpu', logger, max_epochs=1)
    a = torch.load(first / 'best.pt', weights_only=True)
    b = torch.load(second / 'best.pt', weights_only=True)
    assert a['source_dev'] == b['source_dev']
    assert all(torch.equal(a['model_state_dict'][k], b['model_state_dict'][k])
               for k in a['model_state_dict'])
    assert json.loads((first / 'source_calibration.json').read_text()) == json.loads(
        (second / 'source_calibration.json').read_text())


@pytest.mark.parametrize('fold', queue.FRESH_FOLDS)
def test_actual_fit_reload_exports_and_no_overwrite(fold, tmp_path):
    data = queue.source_bank(bank(), fold)
    run = tmp_path / fold / '1234'
    with queue.fold_context(fold) as runner:
        runner.fit_run(data, run, 'TF2', 1234, None, {'code_sha256': {}}, 'cpu',
                       logging.getLogger('fit'), max_epochs=1)
    done = queue.reload_sources(run, data, fold, 'cpu', logging.getLogger('reload'))
    assert done['fold'] == fold and done['selected_checkpoint_reloaded_and_source_verified']
    config = json.loads((run / 'config.json').read_text())
    assert config['seed_plan'] == [1234]
    assert config['model']['trainable_parameters'] == 4907745
    assert config['model']['foundation_encoders_loaded'] is False
    assert config['model']['semantic_loss'] is False
    assert config['model']['clap_audio'] is True
    assert not list(run.glob('*target*'))
    with queue.fold_context(fold) as runner:
        with pytest.raises(FileExistsError):
            runner.fit_run(data, run, 'TF2', 1234, None, {'code_sha256': {}}, 'cpu',
                           logging.getLogger('again'), max_epochs=1)
    (run / 'source_calibration.json').write_text('{}')
    with pytest.raises(ValueError):
        queue.verify_run(run, fold, data)


def test_failed_fold_prevents_later_fits(tmp_path, monkeypatch):
    data = bank()
    calls = []
    monkeypatch.setattr(queue, 'guards', lambda _: None)
    monkeypatch.setattr(queue, 'wait_for_resources', lambda *args, **kwargs: None)
    monkeypatch.setattr(queue, 'preflight', lambda *args: {'passed': True})
    def fail(*args, **kwargs):
        calls.append(original.FOLD)
        assert all(d != FOLDS[original.FOLD]['target'] for d, _ in args[0])
        raise RuntimeError('source failure')
    monkeypatch.setattr(original, 'fit_run', fail)
    launch = tmp_path / 'launch'
    launch.mkdir()
    with pytest.raises(RuntimeError, match='source failure'):
        queue.run_folds(data, {}, 'cpu', logging.getLogger('queue'), output=tmp_path, launch=launch)
    assert calls == ['f2']
    assert not (tmp_path / 'f3').exists() and not (tmp_path / 'f4').exists()


def test_guard_rejects_changed_preserved_f1_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'CACHE', tmp_path / 'cache')
    monkeypatch.setattr(queue, 'OUTPUT', tmp_path / 'out')
    queue.CACHE.mkdir()
    f1 = queue.OUTPUT / 'f1/1234'
    f1.mkdir(parents=True)
    (queue.CACHE / 'complete.json').write_text('cache')
    (f1 / 'best.pt').write_text('weights')
    evidence = {'code_sha256': {}, 'cache_sha256': {'complete.json': checkpoint_digest(queue.CACHE / 'complete.json')},
                'preserved_f1_sha256': {'best.pt': checkpoint_digest(f1 / 'best.pt')}}
    queue.guards(evidence)
    (f1 / 'best.pt').write_text('changed')
    with pytest.raises(ValueError, match='Preserved F1'):
        queue.guards(evidence)
    (f1 / 'best.pt').write_text('weights')
    (queue.CACHE / 'complete.json').write_text('changed')
    with pytest.raises(ValueError, match='source-cache'):
        queue.guards(evidence)


def test_resource_wait_uses_live_memory_without_killing_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'CACHE', tmp_path)
    monkeypatch.setattr(queue, 'LAUNCH', tmp_path)
    (tmp_path / 'complete.json').write_text(json.dumps({'splits': {'x': {'unique_rows': 100}}}))
    monkeypatch.setattr(queue, 'guards', lambda _: None)
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *args, **kwargs: '7000')
    old_read = Path.read_text
    def read(path, *args, **kwargs):
        return 'MemAvailable: 90000000 kB\n' if str(path) == '/proc/meminfo' else old_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    queue.wait_for_resources({}, logging.getLogger('resources'), timeout=1)
    assert json.loads((tmp_path / 'resource_status.json').read_text())['free_gpu_mib'] == 7000
