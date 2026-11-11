"""SAM architecture, source isolation, strict selection and completion gates."""
import copy
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from audio_deepfake_detection.sota.aasist_arch import Model
from audio_deepfake_detection.sota.aasist import NUM_SAMPLES
from audio_deepfake_detection.sota.sam_aasist import MODEL_CONFIG, PARAMETER_COUNT, SAMAASIST
from scripts.sam_aasist import train


class Tiny(nn.Linear):
    def __init__(self):
        super().__init__(6, 2)

    @staticmethod
    def spoof_score(logits):
        return logits[:, 1] - logits[:, 0]


def test_network_cached_sinc_preserves_full_architecture_logits_and_gradients():
    torch.manual_seed(3)
    original = Model(copy.deepcopy(MODEL_CONFIG))
    torch.manual_seed(3)
    cached = SAMAASIST()
    assert sum(p.numel() for p in cached.parameters()) == PARAMETER_COUNT == 297866
    assert MODEL_CONFIG['pool_ratios'] == [0.5]*4
    assert all(torch.equal(v, cached.state_dict()[k]) for k, v in original.state_dict().items())
    waves = torch.randn(2, NUM_SAMPLES)
    y = torch.tensor([0, 1])
    original.eval(); cached.eval()
    with torch.no_grad():
        assert torch.equal(original(waves)[1].flip(1), cached(waves))
    original.train(); cached.train()
    torch.manual_seed(8)
    nn.CrossEntropyLoss(weight=torch.tensor([.1,.9]))(original(waves)[1], 1-y).backward()
    torch.manual_seed(8)
    nn.CrossEntropyLoss(weight=torch.tensor([.9,.1]))(cached(waves), y).backward()
    for a, b in zip(original.parameters(), cached.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_source_records_never_read_target(monkeypatch):
    calls = []
    monkeypatch.setattr(train, 'read_dataset', lambda d,s,r: calls.append((d,s)) or [])
    train.source_records('f1', '/unused')
    assert sorted(calls) == sorted((d,s) for d in ('asv2019','asv5','cfad') for s in ('train','dev'))


def test_arguments_keep_approved_recipe():
    args = train.parse_args(['--fold','f1'])
    assert (args.seed,args.num_workers,args.eval_workers,args.prefetch_factor,args.cpu_threads) == (1234,16,8,2,2)
    for extra in (['--seed','1'], ['--num-workers','-1'], ['--prefetch-factor','0']):
        with pytest.raises(SystemExit):
            train.parse_args(['--fold','f1',*extra])


def test_strict_macro_selection_keeps_earlier_tie_and_refuses_existing_directory(tmp_path, monkeypatch):
    model = Tiny()
    batches = [(torch.randn(8,6),torch.tensor([0,1]*4))]*2
    values = iter([.3,.2,.2,.25,.1])
    monkeypatch.setattr(train, 'source_metrics', lambda _: {'macro_eer':next(values)})
    def score(*args):
        return {'asv2019':(np.array([0,1]),np.array([.0,1.],dtype=np.float32))}
    run = tmp_path/'run'
    result = train.fit(model,batches,{},'cpu',run,{'fold':'f1','seed':1234},score=score,
                       max_epochs=8,patience=2,guard=lambda c:None,log=lambda *a,**k:None)
    assert (result['best_epoch'],result['final_epoch']) == (2,4)
    best = torch.load(run/'best.pt',weights_only=True)
    assert best['epoch'] == 2 and best['optimizer_updates'] == 4
    assert best['sam_state']['rho'] == .05
    assert not (run/'training_complete.json').exists()
    assert (run/'recovery.pt').exists()
    before = (run/'best.pt').read_bytes()
    with pytest.raises(FileExistsError):
        train.fit(model,batches,{},'cpu',run,{},score=score)
    assert (run/'best.pt').read_bytes() == before


@pytest.mark.parametrize('mismatch',[False,True])
def test_fresh_source_reload_required_before_completion(tmp_path,monkeypatch,mismatch):
    torch.manual_seed(9)
    run = tmp_path/'run'
    model = Tiny()
    waves = torch.randn(8,6)
    labels = torch.tensor([0,1]*4)
    loader = DataLoader(TensorDataset(waves,labels),batch_size=1)
    dev = {'asv2019':loader}
    config = {'fold':'f1','seed':1234,'data_root':str(tmp_path),'source_sha256':{}}
    monkeypatch.setattr(train,'source_guards',lambda: {})
    result = train.fit(model,[(waves,labels)],dev,'cpu',run,config,max_epochs=1,log=lambda *a,**k:None)
    root = tmp_path/'asvspoof2019'/'extracted'
    records = {'asv2019':{'dev':[(root/f'{i}.flac',int(y)) for i,y in enumerate(labels)]}}
    def score(model,dev,device):
        values = train.score_sources(model,dev,device)
        if mismatch:
            values['asv2019'][1][0] += .01
        return values
    if mismatch:
        with pytest.raises(RuntimeError,match='reload differs'):
            train.finalize(run,config,records,dev,'cpu',result,model_factory=Tiny,score=score)
        assert not (run/'training_complete.json').exists()
    else:
        train.finalize(run,config,records,dev,'cpu',result,model_factory=Tiny,score=score)
        assert json.loads((run/'training_complete.json').read_text())['selected_checkpoint_reloaded_and_source_verified']
        assert (run/'source_dev_asv2019.csv').exists() and (run/'source_calibration.json').exists()


def test_guard_and_interrupted_checkpoint_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(train,'source_guards',lambda:{'x':'new'})
    with pytest.raises(RuntimeError,match='runtime changed'):
        train.check_guards({'source_sha256':{'x':'old'}})
    path = tmp_path/'best.pt'
    path.with_suffix('.pt.tmp').write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        train.save_checkpoint(path,{})
    assert path.with_suffix('.pt.tmp').read_bytes() == b'preserve'
