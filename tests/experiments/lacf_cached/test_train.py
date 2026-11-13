"""Synthetic checks for the authorized cached F4 switch and source completion."""
import json
import logging

import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureSplit
from scripts.experiments.lacf_cached import train


def bank():
    result = {}
    for domain in train.SOURCES:
        for split in ('train','dev'):
            n = 8
            labels = torch.tensor([0,1]*4)
            result[domain,split] = FeatureSplit([(str(i),int(y)) for i,y in enumerate(labels)],
                [f'{domain}_{split}_{i}' for i in range(n)],labels,
                torch.arange(n)[:,None].repeat(1,3 if split=='train' else 1),
                torch.randn(n,768),torch.randn(n,512))
    return result


def test_original_backend_all_losses_and_parameter_count():
    model = train.new_model(unit_vectors(torch.randn(8,512)),'cpu')
    data = {'wavlm':torch.randn(4,768),'clap':torch.randn(4,512),'labels':torch.tensor([0,1,0,1])}
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output = model(data); losses = model.objective(output,data['labels'])
    assert output['features'].shape == (4,35)
    assert losses['semantic'] > 0 and losses['bonafide'] > 0
    assert torch.allclose(losses['total'],losses['detection']+.25*losses['semantic']+.10*losses['bonafide'])
    losses['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    spoof = model.objective(model(data),torch.ones(4,dtype=torch.long))
    assert spoof['bonafide'] == 0
    assert sum(p.numel() for p in model.parameters()) == 661479


def test_source_only_fresh_fit_reload_export_and_tamper_gate(tmp_path):
    torch.set_num_threads(2)
    run = tmp_path/'f4/1234'
    done = train.fit(bank(),unit_vectors(torch.randn(8,512)),run,{'source_sha256':{}},'cpu',
                     logging.getLogger('synthetic'),epochs=2,patience=1)
    assert done['selected_checkpoint_reloaded_and_source_verified'] and done['target_access'] is False
    config = json.loads((run/'config.json').read_text())
    assert config['model_name']=='LACF-Frozen' and config['run_name']=='lacf'
    assert config['execution_mode']==train.MODE and config['eval_batch_size']==1
    assert config['components']['semantic_loss'] and config['components']['bonafide_loss']
    assert train.verify_completion(run)==done
    with pytest.raises(FileExistsError):
        train.fit(bank(),unit_vectors(torch.randn(8,512)),run,{'source_sha256':{}},'cpu',logging.getLogger('synthetic'))
    (run/'source_dev_cfad.csv').write_text('changed')
    with pytest.raises(ValueError,match='Source export changed'):train.verify_completion(run)


def test_nonfinite_training_never_completes(tmp_path):
    source = bank();source['asv5','train'].wavlm.fill_(float('nan'))
    run = tmp_path/'f4/1234'
    with pytest.raises(RuntimeError):
        train.fit(source,unit_vectors(torch.randn(8,512)),run,{'source_sha256':{}},'cpu',logging.getLogger('synthetic'))
    assert not (run/'training_complete.json').exists()


def test_f4_sources_only_and_guard_failure(tmp_path,monkeypatch):
    assert train.SOURCES == ('asv5','cfad','speechfake')
    monkeypatch.setattr(train,'ROOT',tmp_path)
    (tmp_path/'protected.py').write_text('preserve')
    with pytest.raises(ValueError,match='source changed'):
        train.check_guards({'source_sha256':{'protected.py':'wrong'}})
