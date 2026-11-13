"""CPU-only last-block recipe and pre-target/no-overwrite execution gates."""

import copy
import json
import sys

import numpy as np
import pytest
import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, dataset_root
from scripts.ssl_gating import evaluate_last as evaluate
from scripts.ssl_gating import evaluate_partial
from scripts.ssl_gating import train_last_fold as trainer


def config():
    return {'fold': 'f1', 'seed': 1234, 'run_name': evaluate.RUN_FAMILY, 'model_name': 'SSLFeatureGating',
        'model_config': evaluate.LAST_MODEL_CONFIG, 'precision': 'BF16_autocast_FP32_pooling_losses_scores',
        'trainable_parameters': 15241090, 'total_parameters': 318083586,
        'sources': list(FOLDS['f1']['sources']), 'target_domain': 'speechfake', 'target_access': False,
        'train_max_seconds': 20, 'eval_batch_size': 1, 'batch_size': 24, 'learning_rate': 3e-6,
        'weight_decay': 1e-4, 'optimizer': 'Adam', 'scheduler': 'constant', 'maximum_epochs': 10,
        'patience': 3, 'loss': 'weighted_CE_0.9_0.1_plus_CKA_coefficient1',
        'selection': 'strict_equal_domain_macro_source_dev_eer_ties_earlier'}


@pytest.mark.parametrize('field,value', [('run_name', 'ssl_gating_20s_top4_bf16'), ('fold', 'f2'),
    ('seed', 1235), ('trainable_parameters', 53029762), ('batch_size', 5), ('learning_rate', 1e-4),
    ('train_max_seconds', 10), ('precision', 'float32'), ('maximum_epochs', 30)])
def test_last_evaluation_rejects_wrong_family_scope_or_recipe(field, value):
    c = config(); done = {'target_access': False, 'best_epoch': 10, 'best_macro_source_dev_eer': .03,
                        'selected_checkpoint_reloaded_and_source_verified': True}
    state = {'config': copy.deepcopy(c), 'fold': 'f1', 'epoch': 10, 'macro_dev_eer': .03}
    evaluate.verify_checkpoint(state, done)
    state['config'][field] = value
    with pytest.raises(ValueError, match='SSL-last checkpoint'):
        evaluate.verify_checkpoint(state, done)


def test_f1_corrupt_source_gate_precedes_gpu_model_and_target(tmp_path, monkeypatch):
    run = tmp_path / 'outputs/ssl_gating_last/f1/1234'; launch = run.parent / 'evaluation'
    preparation = run.parent / 'launch'
    for path in (run, launch, preparation): path.mkdir(parents=True)
    for name in ('readiness.json', 'source_validation.json'): (preparation / name).write_text('{}')
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
    torch.save({'config': c, 'fold': 'f1', 'epoch': 10, 'macro_dev_eer': 0.}, run / 'best.pt')
    done = {'fold': 'f1', 'seed': 1234, 'best_epoch': 10, 'final_epoch': 10, 'best_macro_source_dev_eer': 0.,
        'best_checkpoint_sha256': checkpoint_digest(run / 'best.pt'), 'target_access': False,
        'selected_checkpoint_reloaded_and_source_verified': True,
        'source_calibration_sha256': checkpoint_digest(run / 'source_calibration.json'),
        'source_scores_sha256': {p.name: checkpoint_digest(p) for p in run.glob('source_dev_*.csv')}}
    (run / 'training_complete.json').write_text(json.dumps(done))
    (launch / 'source_guards.json').write_text('{}'); (launch / 'input_sha256.json').write_text('{}')
    (run / 'source_dev_cfad.csv').write_text('changed')
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


def test_existing_training_or_evaluation_is_never_overwritten(tmp_path, monkeypatch):
    run = tmp_path / 'outputs/ssl_gating_last/f2/1234'; run.mkdir(parents=True)
    monkeypatch.setattr(trainer, 'ROOT', tmp_path); monkeypatch.setattr(sys, 'argv', ['train_last_fold', '--fold', 'f2'])
    with pytest.raises(FileExistsError): trainer.main()
    launch = tmp_path / 'evaluation'; launch.mkdir()
    (run / 'metrics.json').write_text('preserve')
    monkeypatch.setattr(evaluate, 'RUN', run); monkeypatch.setattr(evaluate, 'LAUNCH', launch)
    with pytest.raises(FileExistsError): evaluate.main()
    assert (run / 'metrics.json').read_text() == 'preserve'


def test_f2_entrypoint_rejects_unrequested_folds(monkeypatch):
    for fold in ('f1', 'f3', 'f4'):
        monkeypatch.setattr(sys, 'argv', ['train_last_fold', '--fold', fold])
        with pytest.raises(SystemExit): trainer.main()
