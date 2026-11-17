"""SSL-only draw/crop replay and durable training snapshots.

Crop seeds belong to sampled draws, so worker restarts and prefetch do not
change the remaining audio views. Domain sampling and crop distributions are
the same as the original loader; the crop RNG assignment is explicitly new.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from audio_deepfake_detection.sota.ssl_gating import collate_audio


class SeededDraws(Dataset):
    def __init__(self, datasets):
        self.data = ConcatDataset(datasets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, draw):
        index, crop_seed = draw
        # No CUDA RNG changes in workers. FullAudioDataset provides the exact
        # existing read/resample/random-up-to20s operation.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(crop_seed).get_state())
            return self.data[index]


class RemainingDraws(Sampler):
    def __init__(self, plan):
        self.order, self.crops = plan.order[plan.offset:], plan.crops[plan.offset:]

    def __iter__(self):
        return iter(zip(self.order.tolist(), self.crops.tolist()))

    def __len__(self):
        return len(self.order)


class EpochPlan:
    def __init__(self, datasets, seed=1234):
        self.data = SeededDraws(datasets)
        self.weights = torch.cat([torch.full((len(d),), 1 / len(d), dtype=torch.double) for d in datasets])
        self.sampler_rng = torch.Generator().manual_seed(seed)
        self.crop_rng = torch.Generator().manual_seed(seed + 2)
        self.order = self.crops = torch.empty(0, dtype=torch.long)
        self.offset = 0

    def start_epoch(self):
        self.order = torch.multinomial(self.weights, len(self.data), replacement=True, generator=self.sampler_rng)
        self.crops = torch.randint(2**63 - 1, (len(self.data),), generator=self.crop_rng)
        self.offset = 0

    def loader(self, *, workers=8, batch_size=24):
        options = {'num_workers': workers, 'pin_memory': torch.cuda.is_available(), 'batch_size': batch_size,
                   'collate_fn': collate_audio, 'sampler': RemainingDraws(self), 'drop_last': False,
                   'generator': torch.Generator().manual_seed(1235)}
        if workers:
            options.update(prefetch_factor=2, persistent_workers=False)
        return DataLoader(self.data, **options)

    def state_dict(self):
        return {'order': self.order, 'crop_seeds': self.crops, 'offset': self.offset,
                'sampler_rng': self.sampler_rng.get_state(), 'crop_rng': self.crop_rng.get_state()}

    def load_state_dict(self, state):
        order, crops, offset = state['order'], state['crop_seeds'], state['offset']
        if (order.shape != (len(self.data),) or crops.shape != order.shape or order.dtype != torch.long
                or crops.dtype != torch.long or not 0 <= offset <= len(self.data)
                or (offset % 24 and offset != len(self.data))
                or (order < 0).any() or (order >= len(self.data)).any()):
            raise ValueError('Recovery draw plan/count/offset differs')
        self.order, self.crops, self.offset = order, crops, offset
        self.sampler_rng.set_state(state['sampler_rng']); self.crop_rng.set_state(state['crop_rng'])


def capture_rng():
    state = np.random.get_state()
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate(),
            'numpy': {'name': state[0], 'keys': torch.tensor(state[1].astype(np.int64)),
                      'position': state[2], 'gaussian': state[3], 'cached': state[4]}}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['name'], n['keys'].numpy().astype(np.uint32), n['position'], n['gaussian'], n['cached']))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def durable_save(path, value):
    """A crash leaves either the previous full checkpoint or an explicit temp file."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_epoch(path, row):
    """Idempotent if interruption occurred after history write but before snapshot."""
    import json
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if len(rows) == row['epoch'] and rows[-1] == row:
        return
    if len(rows) != row['epoch'] - 1:
        raise ValueError('Recovery history is inconsistent; preserve it for review')
    with path.open('a') as stream:
        stream.write(json.dumps(row) + '\n'); stream.flush(); os.fsync(stream.fileno())
