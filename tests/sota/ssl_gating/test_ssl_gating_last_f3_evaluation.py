"""CPU-only last-block recipe and pre-target/no-overwrite execution gates."""

import copy
import json
import sys

import numpy as np
import pytest
import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, dataset_root
from scripts.ssl_gating import evaluate_last_f3 as evaluate
from scripts.ssl_gating import evaluate_partial


def config():
    return {'fold': 'f3', 'seed': 1234, 'run_name': evaluate.RUN_FAMILY, 'model_name': 'SSLFeatureGating',
        'model_config': evaluate.LAST_MODEL_CONFIG, 'precision': 'BF16_autocast_FP32_pooling_losses_scores',
        'trainable_parameters': 15241090, 'total_parameters': 318083586,
        'sources': list(FOLDS['f3']['sources']), 'target_domain': 'asv5', 'target_access': False,
        'train_max_seconds': 20, 'eval_batch_size': 1, 'batch_size': 24, 'learning_rate': 3e-6,
        'weight_decay': 1e-4, 'optimizer': 'Adam', 'scheduler': 'constant', 'maximum_epochs': 10,
        'patience': 3, 'loss': 'weighted_CE_0.9_0.1_plus_CKA_coefficient1',
        'selection': 'strict_equal_domain_macro_source_dev_eer_ties_earlier',
        'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'transformers': __import__('transformers').__version__, 'deterministic_algorithms': True, 'cublas_workspace': ':4096:8', 'allocator': 'max_split_size_mb:128'}


@pytest.mark.parametrize('field,value', [('run_name', 'ssl_gating_20s_top4_bf16'), ('fold', 'f1'),
    ('seed', 1235), ('trainable_parameters', 53029762), ('batch_size', 5), ('learning_rate', 1e-4),
    ('train_max_seconds', 10), ('precision', 'float32'), ('maximum_epochs', 30),
    ('torch', 'changed'), ('cuda', 'changed'), ('transformers', 'changed')])
def test_last_evaluation_rejects_wrong_family_scope_or_recipe(field, value):
    c = config(); done = {'target_access': False, 'best_epoch': 10, 'best_macro_source_dev_eer': .03,
                        'selected_checkpoint_reloaded_and_source_verified': True}
    state = {'config': copy.deepcopy(c), 'fold': 'f3', 'epoch': 10, 'macro_dev_eer': .03}
    evaluate.verify_checkpoint(state, done)
    state['config'][field] = value
    with pytest.raises(ValueError, match='SSL-last checkpoint'):
        evaluate.verify_checkpoint(state, done)


def test_f3_corrupt_source_gate_precedes_gpu_model_and_target(tmp_path, monkeypatch):
    run = tmp_path / 'outputs/ssl_gating_last/f3/1234'; launch = run.parent / 'evaluation'
    preparation = run.parent / 'launch'
    for path in (run, launch, preparation): path.mkdir(parents=True)
    for name in ('readiness.json', 'source_validation.json'): (preparation / name).write_text('{}')
    (preparation / evaluate.PRODUCER_EXIT).write_text(json.dumps({'verified_complete': True, 'trainer_exit': 0, 'logger_exit': 0, 'target_access': False}))
    c = config(); c.update(source_sha256={}, dataset_counts={d: {'dev': 4} for d in c['sources']},
        readiness_sha256=checkpoint_digest(preparation / 'readiness.json'),
        source_validation_sha256=checkpoint_digest(preparation / 'source_validation.json'))
    records, sources = {}, {}
    for domain in c['sources']:
        records[domain] = [(str(dataset_root(tmp_path, domain) / f'{i}.wav'), i % 2) for i in range(4)]
        sources[domain] = (np.array([0, 1, 0, 1]), np.array([-2., 2., -1., 1.]))
        write_score_csv(run / f'source_dev_{domain}.csv',
            ((f'{i}.wav', domain, i % 2, float(sources[domain][1][i])) for i in range(4)))
    (run / 'source_calibration.json').write_text(json.dumps(calibrate_source_scores(sources)))
    (run / 'config.json').write_text(json.dumps(c))
    torch.save({'config': c, 'fold': 'f3', 'epoch': 10, 'macro_dev_eer': 0.}, run / 'best.pt')
    done = {'fold': 'f3', 'seed': 1234, 'best_epoch': 10, 'final_epoch': 10, 'best_macro_source_dev_eer': 0.,
        'best_checkpoint_sha256': checkpoint_digest(run / 'best.pt'), 'target_access': False,
        'selected_checkpoint_reloaded_and_source_verified': True,
        'source_calibration_sha256': checkpoint_digest(run / 'source_calibration.json'),
        'source_scores_sha256': {p.name: checkpoint_digest(p) for p in run.glob('source_dev_*.csv')}}
    (run / 'training_complete.json').write_text(json.dumps(done))
    (launch / 'source_guards.json').write_text('{}'); (launch / 'input_sha256.json').write_text('{}')
    (run / 'source_dev_speechfake.csv').write_text('changed')
    opened = []
    def reader(domain, split, root):
        opened.append((domain, split)); assert split == 'dev'; return records[domain]
    monkeypatch.setattr(evaluate_partial, 'read_dataset', reader)
    monkeypatch.setattr(evaluate, 'RUN', run); monkeypatch.setattr(evaluate, 'LAUNCH', launch)
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path); monkeypatch.setattr(evaluate, 'DATA', tmp_path)
    monkeypatch.setattr(evaluate, 'read_dataset', lambda *a: pytest.fail('Target opened after a failed source gate'))
    monkeypatch.setattr(evaluate.LastSSLFeatureGating, 'from_frontend_config',
                        lambda *a: pytest.fail('Model constructed after a failed source gate'))
    with pytest.raises(ValueError, match='Recorded source scores changed'):
        evaluate.main()
    assert all(split == 'dev' for _, split in opened)
    assert not (run / 'target_scores.csv').exists() and not (launch / 'started.json').exists()



@pytest.mark.parametrize('field,value', [('verified_complete', False), ('trainer_exit', 1),
    ('logger_exit', 1), ('target_access', True)])
def test_failed_producer_is_rejected(field, value):
    producer = dict(verified_complete=True, trainer_exit=0, logger_exit=0, target_access=False)
    evaluate.verify_producer(producer)
    producer[field] = value
    with pytest.raises(ValueError, match='producer'):
        evaluate.verify_producer(producer)


def test_existing_f3_evaluation_is_preserved(tmp_path, monkeypatch):
    run = tmp_path / '1234'; run.mkdir()
    launch = tmp_path / 'evaluation'; launch.mkdir()
    (run / 'metrics.json').write_text('preserve')
    monkeypatch.setattr(evaluate, 'RUN', run)
    monkeypatch.setattr(evaluate, 'LAUNCH', launch)
    with pytest.raises(FileExistsError):
        evaluate.main()
    assert (run / 'metrics.json').read_text() == 'preserve'


def test_f3_null_compatibility_is_narrow_and_does_not_mutate():
    stored = {'frontend_config': {'hidden_size': 1024}, 'source_sha256': {}}
    saved = copy.deepcopy(stored)
    saved['frontend_config']['attn_implementation'] = None
    before = copy.deepcopy(saved)
    evaluate.verify_run_config(saved, stored, {})
    assert saved == before and 'attn_implementation' not in stored['frontend_config']
    saved['frontend_config']['attn_implementation'] = 'sdpa'
    with pytest.raises(ValueError): evaluate.verify_run_config(saved, stored, {})
    saved = copy.deepcopy(before); saved['frontend_config']['hidden_size'] = 768
    with pytest.raises(ValueError): evaluate.verify_run_config(saved, stored, {})


def test_wait_for_memory_uses_cpu_only_and_preserves_target_hold(tmp_path, monkeypatch):
    calls = []
    free = iter(['5769', '11264'])
    monkeypatch.setattr(evaluate, 'check_guards', lambda: calls.append('guards'))
    monkeypatch.setattr(evaluate.subprocess, 'check_output', lambda *a, **k: next(free))
    monkeypatch.setattr(evaluate.time, 'sleep', lambda seconds: calls.append(seconds))
    monkeypatch.setattr(evaluate, 'LAUNCH', tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('CUDA initialized in resource wait'))
    evaluate.wait_for_headroom()
    assert calls == ['guards', 30, 'guards']
    status = json.loads((tmp_path / 'queue_status.json').read_text())
    assert status['target_access'] is False and status['status'] == 'waiting_for_gpu_memory'
