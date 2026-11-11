"""Synthetic CPU tests: crop identity, CLAP bypass, gradients and source-only completion."""

import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import WavLMConfig, WavLMModel

from audio_deepfake_detection.experiments.tf2_frozen_clap import model_data as data
from audio_deepfake_detection.lacf.model import ComponentConfig, LACFLoss
from audio_deepfake_detection.protocol import checkpoint_digest
from audio_deepfake_detection.token_fusion.adaptation import AdaptationConfig, AdaptedTF2
from scripts.experiments.tf2_frozen_clap import train as runner


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(13*32+512, 1)
        self.objective = LACFLoss(ComponentConfig(semantic_loss=False, bonafide_loss=False))

    def forward(self, batch):
        return {"logit": self.linear(torch.cat((batch["wavlm"].flatten(1), batch["clap"]), 1)).squeeze(1)}


class Audio(nn.Module):
    def forward(self, input_features, **kwargs):
        return SimpleNamespace(pooler_output=input_features)


class Clap(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_model = Audio()
        self.audio_projection = nn.Linear(4, 512)


def tiny_wavlm():
    return WavLMModel(WavLMConfig(hidden_size=32, num_hidden_layers=12, num_attention_heads=4,
        intermediate_size=64, conv_dim=(8, 8), conv_stride=(5, 2), conv_kernel=(10, 3),
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=4, mask_time_prob=0))


def test_exact_primary_trainable_inventory_without_loading_encoders():
    wavlm = nn.Module()
    wavlm.encoder = nn.Module()
    wavlm.encoder.layers = nn.ModuleList()
    for _ in range(12):
        block = nn.Module()
        block.attention = nn.Module()
        block.attention.q_proj = nn.Linear(768, 768)
        block.attention.v_proj = nn.Linear(768, 768)
        wavlm.encoder.layers.append(block)
    model = data.CachedCLAPTF2(wavlm)
    assert model.configuration()['counts'] == {'wavlm_lora': 294912, 'clap': 0, 'fusion': 4907745}
    assert model.configuration()['trainable_parameters'] == 5202657
    assert not model.configuration()['clap_encoder_loaded']


def test_cached_forward_matches_frozen_live_clap_and_keeps_lora_gradients():
    wavlm, fusion = tiny_wavlm(), Fusion()
    reference = AdaptedTF2(copy.deepcopy(wavlm), Clap(), AdaptationConfig(False), fusion=copy.deepcopy(fusion)).eval()
    cached = data.CachedCLAPTF2(copy.deepcopy(wavlm), fusion=copy.deepcopy(fusion)).eval()
    cached.load_trainable_state(reference.trainable_state())
    waves = torch.randn(2, 1000)
    masks = torch.ones_like(waves, dtype=torch.bool)
    masks[0, 800:] = False
    inputs = {"wavlm": {"input_values": waves, "attention_mask": masks},
              "clap": {"input_features": torch.randn(2, 4)}}
    features = reference.clap_features(inputs["clap"]).detach().requires_grad_()
    observed = cached({**inputs, "clap": features})
    assert torch.equal(observed["logit"], reference(inputs)["logit"])
    observed["logit"].sum().backward()
    assert features.grad is None and not list(cached.clap.parameters())
    parameters = dict(cached.wavlm.named_parameters())
    assert all(p.grad is None for p in parameters.values() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values() if p.requires_grad)
    assert any(p.grad.abs().sum() > 0 for name, p in parameters.items() if 'lora_b' in name)
    assert cached.fusion.linear.weight.grad.abs().sum() > 0
    assert not cached.wavlm.training and not cached.clap.training
    assert {g['name'] for g in cached.optimizer().param_groups} == {'wavlm_lora', 'tf2'}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path/'cache'
    files = []
    for name, length, label in [('long', 176000, 1), ('short', 800, 0)]:
        path = tmp_path/f'{name}.wav'
        sf.write(path, np.linspace(-.5, .5, length, dtype=np.float32), 16000, subtype='FLOAT')
        files.append((str(path), label))
    monkeypatch.setattr(data, 'read_dataset', lambda d, s, r: files)
    monkeypatch.setattr(data, 'utterance_id', lambda p, *args: Path(p).name)
    metadata, values = [], []
    for i, (path, label) in enumerate(files):
        frames = sf.info(path).frames
        identifier = f'asv2019/train/{Path(path).name}'
        intervals, aliases = data.crop_intervals(frames, 16000, identifier, training=True)
        base = len(values)
        metadata.append({'index': i, 'id': identifier, 'label': label, 'frames': frames, 'rate': 16000,
                         'intervals': [list(v) for v in intervals], 'view_rows': [base+v for v in aliases]})
        for v in range(len(intervals)):
            value = np.arange(512, dtype=np.float32)+v
            value[0] = 1 if label else -1
            values.append(value)
    directory = root/'asv2019/train/chunk_00000'
    directory.mkdir(parents=True)
    (directory/'records.json').write_text(json.dumps(metadata))
    np.save(directory/'clap_projected.npy', np.stack(values))
    # Any accidental WavLM cache load must fail immediately.
    (directory/'wavlm_layers.npy').write_bytes(b'not a numpy array; must never be opened')
    state = SimpleNamespace(root=root, directory=directory, records=files, metadata=metadata, values=np.stack(values))

    def seal():
        marker = {'records': len(files), 'unique_rows': len(values), 'files_sha256': {
            name: checkpoint_digest(directory/name) for name in ('records.json', 'clap_projected.npy')}}
        (directory/'complete.json').write_text(json.dumps(marker))
        ids = [Path(p).name for p, _ in files]
        membership = hashlib.sha256(json.dumps(list(zip(ids, [y for _, y in files]))).encode()).hexdigest()
        state.manifest = {'records': len(files), 'unique_rows': len(values), 'logical_views': len(files)*3,
            'canonical_ids_labels_sha256': membership, 'chunks': [{'path': 'chunk_00000', 'records': len(files),
            'unique_rows': len(values), 'completion_sha256': checkpoint_digest(directory/'complete.json')}]}
        (directory.parent/'complete.json').write_text(json.dumps(state.manifest))
    state.seal = seal
    seal()
    return state


class Processor:
    def __call__(self, waves, **kwargs):
        waves = [torch.from_numpy(w) for w in waves]
        lengths = torch.tensor([len(w) for w in waves])
        values = pad_sequence(waves, batch_first=True)
        return {'input_values': values, 'attention_mask': torch.arange(values.shape[1])[None] < lengths[:, None]}


def test_cache_only_reads_clap_and_matches_all_fixed_audio_views(cache):
    split = data.load_clap_split(cache.root, 'asv2019', 'train', cache.manifest, '.')
    dataset = data.CachedCLAPDataset([split])
    for i, row in enumerate(cache.metadata):
        for view in range(3):
            wave, feature, label = dataset[i, view]
            interval = row['intervals'][row['view_rows'][view]-row['view_rows'][0]]
            audio, _ = sf.read(cache.records[i][0], start=interval[0], frames=interval[1], dtype='float32')
            assert torch.equal(wave, torch.from_numpy(audio))
            assert np.array_equal(feature.numpy(), cache.values[row['view_rows'][view]])
            assert label == row['label']
    batch = data.CachedCLAPCollator(Processor())([dataset[0, 2], dataset[1, 0]])
    assert batch['clap'].shape == (2, 512)
    assert set(batch['wavlm']) == {'input_values', 'attention_mask'}
    assert batch['labels'].tolist() == [1, 0]


@pytest.mark.parametrize('change', ['bytes', 'aliases', 'label', 'shape', 'nan'])
def test_cache_rejects_tampered_or_misaligned_features(cache, change):
    if change == 'bytes':
        with (cache.directory/'clap_projected.npy').open('ab') as f:
            f.write(b'tampered')
    elif change in ('aliases', 'label'):
        rows = json.loads((cache.directory/'records.json').read_text())
        if change == 'aliases':
            rows[0]['view_rows'][1] = rows[-1]['view_rows'][0]
        else:
            rows[0]['label'] = 0
        (cache.directory/'records.json').write_text(json.dumps(rows))
        cache.seal()
    else:
        values = cache.values[:, :511] if change == 'shape' else cache.values.copy()
        if change == 'nan':
            values[0, 1] = np.nan
        np.save(cache.directory/'clap_projected.npy', values)
        cache.seal()
    with pytest.raises(ValueError):
        data.load_clap_split(cache.root, 'asv2019', 'train', cache.manifest, '.')


def test_target_and_changed_audio_are_rejected(cache):
    with pytest.raises(ValueError, match='source train/dev'):
        data.load_clap_split(cache.root, 'speechfake', 'test', {}, '.')
    split = data.load_clap_split(cache.root, 'asv2019', 'train', cache.manifest, '.')
    sf.write(cache.records[0][0], np.zeros(900, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match='length/rate'):
        data.CachedCLAPDataset([split])[0, 0]


def test_combined_dataset_maps_global_indices_to_correct_partition(cache):
    first = data.load_clap_split(cache.root, 'asv2019', 'train', cache.manifest, '.')
    second = copy.copy(first)
    second.features = first.features + 100
    dataset = data.CachedCLAPDataset([first, second])
    assert torch.equal(dataset[len(first.records), 2][1], second.features[second.view_rows[0, 2]])
    for selection in ((-1, 0), (len(dataset), 0), (0, 3)):
        with pytest.raises(ValueError): dataset[selection]


@pytest.mark.parametrize('change', [None, 'recipe', 'target', 'failed', 'throughput', 'memory'])
def test_training_gate_binds_source_gpu_evidence(tmp_path, change):
    config = {'gpu_memory_budget_gib': 10, 'physical_batch': 22}
    record = {'passed': True, 'target_access': False, 'recipe_sha256': runner.digest(config),
        'checks': dict.fromkeys(runner.CHECKS, True),
        'measurements': {'train_examples_per_second': 10, 'source_dev_examples_per_second': 5,
                         'peak_allocated_bytes': 2**30}}
    if change == 'recipe': record['recipe_sha256'] = 'wrong'
    if change == 'target': record['target_access'] = True
    if change == 'failed': record['checks']['finite_gradients'] = False
    if change == 'throughput': record['measurements']['train_examples_per_second'] = float('nan')
    if change == 'memory': record['measurements']['peak_allocated_bytes'] = 10*1024**3
    path = tmp_path/'validation.json'
    path.write_text(json.dumps(record))
    if change:
        with pytest.raises(ValueError): runner.verify_validation(path, config)
    else:
        assert runner.verify_validation(path, config)['sha256'] == checkpoint_digest(path)
    with pytest.raises(ValueError, match='Training held'):
        runner.verify_validation(None, config)


def test_cli_rejects_training_before_cache_model_or_gpu_access(monkeypatch):
    monkeypatch.setattr('sys.argv', ['train.py', '--mode', 'train'])
    monkeypatch.setattr(runner, 'cache_identity', lambda *a: pytest.fail('must stop before cache/model/GPU'))
    with pytest.raises(SystemExit): runner.main()
    monkeypatch.setattr('sys.argv', ['train.py', '--output', '/tmp/not_an_experiment'])
    with pytest.raises(SystemExit): runner.main()


def test_recipe_hash_treats_default_and_explicit_memory_limit_identically(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'checkpoint_digest', lambda path: 'fixed-sha')
    args = SimpleNamespace(seed=1234, physical_batch=22, workers=8, gpu_memory_budget_gib=10,
                           cache_root=tmp_path, data_root=tmp_path)
    manifest = {'config_sha256': 'config', 'accelerated_execution_sha256': 'execution',
                'splits': {f'{domain}/{part}': {'records': 2, 'canonical_ids_labels_sha256': 'membership'}
                           for domain in ('asv2019', 'asv5', 'cfad') for part in ('train', 'dev')}}
    execution = {'pretrained': {'wavlm_revision': 'wavlm', 'clap_revision': 'clap'}}
    default = runner.recipe(args, manifest, execution)
    args.gpu_memory_budget_gib = 10.0
    explicit = runner.recipe(args, manifest, execution)
    assert runner.digest(default) == runner.digest(explicit)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.objective = LACFLoss(ComponentConfig(semantic_loss=False, bonafide_loss=False))

    def forward(self, batch):
        return {'logit': self.weight*batch['clap'][:, 0]}

    def optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=.01)

    def trainable_state(self):
        return {'weight': self.weight.detach().clone()}

    def load_trainable_state(self, state):
        self.load_state_dict(state, strict=True)


@pytest.mark.parametrize('reload_mismatch', [False, True])
def test_fit_keeps_earlier_ties_and_reloads_before_source_completion(cache, tmp_path, monkeypatch, reload_mismatch):
    split = data.load_clap_split(cache.root, 'asv2019', 'train', cache.manifest, '.')
    train = data.CachedCLAPDataset([split])
    dev_split = copy.copy(split)
    dev_split.part = 'dev'
    dev = data.CachedCLAPDataset([dev_split])
    original_loader = data.make_loader
    monkeypatch.setattr(runner, 'make_loader', lambda dataset, collator, **kw:
        original_loader(dataset, collator, **{**kw, 'workers': 0}))
    monkeypatch.setattr(runner, 'build_model', lambda *a: (Toy(), None))
    out = tmp_path/'run'
    out.mkdir()
    config = {'provenance': {'code_sha256': {}}}
    (out/'config.json').write_text(json.dumps(config))
    args = SimpleNamespace(output=out, seed=1234, workers=0, physical_batch=22)
    if reload_mismatch:
        def wrong_reload(model, state):
            model.load_trainable_state({'weight': state['weight']+1})
        monkeypatch.setattr(runner, 'restore_adaptation', wrong_reload)
        with pytest.raises(ValueError, match='source reload differs'):
            runner.fit(Toy(), [train], {'asv2019': dev}, data.CachedCLAPCollator(Processor()), args,
                       config, {}, logging.getLogger('cached_clap_test'))
        assert not (out/'complete.json').exists()
        return
    runner.fit(Toy(), [train], {'asv2019': dev}, data.CachedCLAPCollator(Processor()), args,
               config, {}, logging.getLogger('cached_clap_test'))
    complete = json.loads((out/'complete.json').read_text())
    assert complete['best_epoch'] == 1 and complete['final_epoch'] == 6
    assert complete['selected_checkpoint_reloaded_and_source_verified'] and not complete['target_access']
    assert len((out/'history.jsonl').read_text().splitlines()) == 6
    assert not (out/'target_scores.csv').exists()
