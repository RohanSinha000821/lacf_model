"""Exact CPU continuation across prefetch/worker restart, not weights-only recovery."""

import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from audio_deepfake_detection.sota.ssl_gating_resume import (
    EpochPlan, append_epoch, capture_rng, durable_save, restore_rng,
)
from audio_deepfake_detection.sota.ssl_gating import FullAudioDataset
from scripts.ssl_gating import train_last_f3 as trainer


class Audio(Dataset):
    def __init__(self, count, domain): self.count, self.domain = count, domain
    def __len__(self): return self.count
    def __getitem__(self, index):
        # Torch crop randomness matches FullAudioDataset's only random operation.
        return torch.rand(4) + self.domain + index / 100, index % 2


def datasets(): return [Audio(17, 0), Audio(34, 1)]


@pytest.fixture(autouse=True)
def cpu_threads(): torch.set_num_threads(2)


def test_domain_sampler_matches_original_and_worker_crops_are_identical():
    plan = EpochPlan(datasets()); plan.start_epoch()
    expected = torch.multinomial(plan.weights, 51, replacement=True, generator=torch.Generator().manual_seed(1234))
    assert torch.equal(plan.order, expected)
    assert plan.weights[:17].sum() == plan.weights[17:].sum()
    before = torch.get_rng_state()
    zero = list(plan.loader(workers=0)); parallel = list(plan.loader(workers=2))
    assert torch.equal(before, torch.get_rng_state())
    assert len(zero) == 3 and len(zero[-1][1]) == 3
    assert all(torch.equal(a, b) and torch.equal(y, z) for (a, y), (b, z) in zip(zero, parallel))


@pytest.mark.parametrize('restart_workers', (0, 2))
def test_model_optimizer_rng_sampling_and_remaining_steps_match(tmp_path, restart_workers):
    torch.manual_seed(1234); np.random.seed(1234); random.seed(1234)
    plan = EpochPlan(datasets()); plan.start_epoch()
    def model(): return nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Dropout(.2), nn.Linear(16, 1))
    current = model(); optimizer = torch.optim.Adam(current.parameters(), lr=3e-6, weight_decay=1e-4)
    def step(network, opt, batch):
        wave, labels = batch; opt.zero_grad(set_to_none=True)
        # Exercise all main-process random sources used by encoder augmentation.
        wave = wave * (1 + float(np.random.uniform()) + random.random())
        with torch.autocast('cpu', dtype=torch.bfloat16): logits = network(wave).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float())
        loss.backward(); opt.step(); return loss.detach().clone()
    iterator = iter(plan.loader(workers=2)); first = next(iterator)
    step(current, optimizer, first); plan.offset += len(first[1])
    path = tmp_path / 'recovery.pt'
    durable_save(path, {'model': current.state_dict(), 'optimizer': optimizer.state_dict(),
                       'plan': plan.state_dict(), 'rng': capture_rng()})
    expected_losses = [step(current, optimizer, batch) for batch in iterator]
    expected = copy.deepcopy(current.state_dict()); expected_opt = copy.deepcopy(optimizer.state_dict())
    state = torch.load(path, weights_only=True)
    restored = model(); opt = torch.optim.Adam(restored.parameters(), lr=3e-6, weight_decay=1e-4)
    restored.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
    resumed = EpochPlan(datasets()); resumed.load_state_dict(state['plan']); restore_rng(state['rng'])
    actual_losses = [step(restored, opt, batch) for batch in resumed.loader(workers=restart_workers)]
    assert all(torch.equal(a, b) for a, b in zip(expected_losses, actual_losses))
    assert len(actual_losses) == 2 and all(torch.equal(v, restored.state_dict()[k]) for k, v in expected.items())
    for key, values in expected_opt['state'].items():
        assert all(torch.equal(v, opt.state_dict()['state'][key][name]) for name, v in values.items())
    # Private generators continue to the identical next epoch too.
    plan.start_epoch(); resumed.start_epoch()
    assert torch.equal(plan.order, resumed.order) and torch.equal(plan.crops, resumed.crops)


@pytest.mark.parametrize('change', ('offset', 'count', 'index'))
def test_corrupt_or_wrong_draw_plan_is_rejected(change):
    plan = EpochPlan(datasets()); plan.start_epoch(); state = copy.deepcopy(plan.state_dict())
    if change == 'offset': state['offset'] = 1
    if change == 'count': state['order'] = state['order'][:-1]
    if change == 'index': state['order'][0] = 51
    with pytest.raises(ValueError): plan.load_state_dict(state)


def test_durable_save_keeps_last_good_file_and_refuses_interrupted_temp(tmp_path):
    path = tmp_path / 'recovery.pt'; durable_save(path, {'step': 1})
    temporary = path.with_suffix('.pt.tmp'); temporary.write_bytes(b'interrupted')
    with pytest.raises(FileExistsError): durable_save(path, {'step': 2})
    assert torch.load(path, weights_only=True)['step'] == 1 and temporary.read_bytes() == b'interrupted'


def test_history_append_repeated_after_recovery_is_idempotent(tmp_path):
    path = tmp_path / 'history.jsonl'; row = {'epoch': 1, 'source_dev': {'macro_eer': .1}}
    append_epoch(path, row); before = path.read_bytes(); append_epoch(path, row)
    assert path.read_bytes() == before
    with pytest.raises(ValueError): append_epoch(path, {'epoch': 1, 'source_dev': {'macro_eer': .2}})


def test_wrong_checkpoint_scope_never_passes_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, 'RUN', tmp_path)
    config = {'fold': 'f3', 'seed': 1234, 'model_config': trainer.LAST_MODEL_CONFIG, 'trainable_parameters': 15241090,
              'sources': list(trainer.SOURCES), 'target_access': False, 'batch_size': 24,
              'precision': 'BF16_autocast_FP32_pooling_losses_scores', 'learning_rate': 3e-6, 'weight_decay': 1e-4}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    state = {'format_version': 1, 'config': config, 'config_sha256': trainer.checkpoint_digest(tmp_path / 'config.json'),
             'epoch': 1, 'stage': 'train', 'best': None, 'patience': 0,
             'plan': {'offset': 0, 'order': torch.arange(51)}}
    trainer.verify_recovery(state, config)
    state['stage'] = 'source_dev'
    with pytest.raises(ValueError, match='phase progress'): trainer.verify_recovery(state, config)
    state['stage'] = 'fake_completed'
    with pytest.raises(ValueError): trainer.verify_recovery(state, config)


def test_actual_full_audio_crop_survives_worker_restart(tmp_path):
    import soundfile as sf
    path = tmp_path / 'long.wav'
    sf.write(path, np.linspace(-.5, .5, 320097, dtype=np.float32), 16000, subtype='FLOAT')
    datasets = [FullAudioDataset([(str(path), i % 2) for i in range(27)], training=True)]
    plan = EpochPlan(datasets); plan.start_epoch()
    expected = list(plan.loader(workers=2)); plan.offset = 24
    restored = EpochPlan(datasets); restored.load_state_dict(plan.state_dict())
    actual = list(restored.loader(workers=0))
    assert len(actual) == 1 and actual[0][0].shape == (3, 320000)
    assert torch.equal(expected[-1][0], actual[0][0]) and torch.equal(expected[-1][1], actual[0][1])
