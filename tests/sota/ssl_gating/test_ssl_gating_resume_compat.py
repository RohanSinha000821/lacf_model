"""A null serialization key may differ; all actual recipe checks stay strict."""

import copy
import json

import pytest
import torch
from transformers import Wav2Vec2Config

from scripts.ssl_gating import resume_last_f3 as recovery


def fixture(tmp_path, monkeypatch):
    trainer = recovery.trainer
    monkeypatch.setattr(trainer, 'RUN', tmp_path)
    config = dict(fold='f3', seed=1234, model_config=trainer.LAST_MODEL_CONFIG,
                  trainable_parameters=15241090, sources=list(trainer.SOURCES),
                  target_access=False, batch_size=24,
                  precision='BF16_autocast_FP32_pooling_losses_scores',
                  learning_rate=3e-6, weight_decay=1e-4,
                  frontend_config={'hidden_size': 1024})
    (tmp_path / 'config.json').write_text(json.dumps(config))
    state = dict(format_version=1, config=copy.deepcopy(config),
                 config_sha256=trainer.checkpoint_digest(tmp_path / 'config.json'),
                 epoch=8, stage='train', best={'epoch': 7}, patience=0,
                 plan={'offset': 24, 'order': torch.arange(48)})
    return state, config


def test_installed_transformers_adds_only_null_attention_key():
    original = {'hidden_size': 1024}
    mutated = copy.deepcopy(original)
    Wav2Vec2Config.from_dict(mutated)
    assert mutated == dict(original, attn_implementation=None)


def test_accepts_null_without_mutating_artifacts(tmp_path, monkeypatch):
    state, config = fixture(tmp_path, monkeypatch)
    recovery.verify_recovery(state, config)
    state['config']['frontend_config']['attn_implementation'] = None
    before = copy.deepcopy(state['config'])
    recovery.verify_recovery(state, config)
    assert state['config'] == before
    assert 'attn_implementation' not in config['frontend_config']


@pytest.mark.parametrize('change', ['attention', 'hidden_size', 'batch', 'hash', 'stage'])
def test_other_changes_still_rejected(tmp_path, monkeypatch, change):
    state, config = fixture(tmp_path, monkeypatch)
    state['config']['frontend_config']['attn_implementation'] = None
    if change == 'attention': state['config']['frontend_config']['attn_implementation'] = 'sdpa'
    if change == 'hidden_size': state['config']['frontend_config']['hidden_size'] = 768
    if change == 'batch': state['config']['batch_size'] = 12
    if change == 'hash': state['config_sha256'] = 'wrong'
    if change == 'stage': state['stage'] = 'completed'
    with pytest.raises(ValueError): recovery.verify_recovery(state, config)
