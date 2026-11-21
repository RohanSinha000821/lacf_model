"""Synthetic/offline CPU checks; these do not establish real-data/GPU readiness."""

import json
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from audio_deepfake_detection.sota.frozen_frontend import FrozenSpeechEncoder
from audio_deepfake_detection.sota.lhcc import LHCCConfig, LHCCDetector, classification_loss
from audio_deepfake_detection.sota.nsf_pinn import NSFPINNConfig, NSFPINNDetector, physics_loss
from audio_deepfake_detection.sota import frozen_training as train
from scripts.sam_aasist import train as shared


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class FakeOfficial(nn.Module):
    def __init__(self, layers, width):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers, hidden_size=width)
        self.weight = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(0.9)
        self.calls = 0

    def forward(self, waves, *, output_hidden_states):
        self.calls += 1
        h = self.dropout(waves[:, :3, None].expand(-1, -1, self.config.hidden_size)) * self.weight
        return SimpleNamespace(last_hidden_state=h,
            hidden_states=tuple(h + i for i in range(self.config.num_hidden_layers + 1)))


@pytest.mark.parametrize("method,layers,width", [("lhcc",24,1024),("nsf_pinn",12,768)])
def test_official_encoder_stays_frozen_eval_and_layer_order(method, layers, width):
    raw = FakeOfficial(layers, width)
    encoder = FrozenSpeechEncoder(method, raw)
    encoder.train()
    waves = torch.ones(2, 400, requires_grad=True)
    a, b = encoder(waves), encoder(waves)
    assert torch.equal(a,b) and not a.requires_grad
    assert not raw.training and not encoder.training
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    if method == "lhcc":
        assert a.shape == (2,24,3,1024)
        assert a[0,0,0,0] == 2 and a[0,-1,0,0] == 25  # excludes state0
    else:
        assert a.shape == (2,3,768)


def test_wrong_backbone_rejected():
    with pytest.raises(ValueError, match="geometry"):
        FrozenSpeechEncoder("lhcc", FakeOfficial(12,768))


@pytest.mark.parametrize('method',['lhcc','nsf_pinn'])
def test_inspect_never_accesses_gpu_encoder_or_data(method,monkeypatch,capsys):
    def forbidden(*a,**k):
        raise AssertionError('Inspect must not access GPU/weights/data')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(train,'FrozenSpeechEncoder',forbidden)
    monkeypatch.setattr(train,'source_records',forbidden)
    monkeypatch.setitem(train.DETECTORS,method,tiny_nsf)
    train.main(method,['--fold','f1','--inspect'])
    result=json.loads(capsys.readouterr().out)
    assert result['gpu_validation']=='pending' and not result['target_access']


def tiny_lhcc():
    return LHCCDetector(LHCCConfig(width=12,frames=9,channels=4,res2_scale=2,
        attention_heads=2,ffn_hidden=12,se_reduction=4,classifier_hidden=8))


def test_lhcc_asymmetry_sharing_gradient_and_reload():
    torch.manual_seed(4)
    head = tiny_lhcc()
    assert len(head.fusion.low) == 5 and len(head.fusion.high) == 4
    assert head.blocks[0].attention is head.blocks[2].attention
    assert head.blocks[0].attention is not head.blocks[1].attention
    assert head.blocks[0].ffn is not head.blocks[2].ffn
    x = torch.randn(2,24,9,12)
    logits = head(x)
    assert logits.shape == (2,2)
    classification_loss(logits,torch.tensor([0,1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    clone = tiny_lhcc()
    clone.load_state_dict(head.state_dict(),strict=True)
    head.eval(); clone.eval()
    with torch.no_grad():
        assert torch.equal(head(x),clone(x))
    with pytest.raises(ValueError):
        head(x[:, :23])


def test_lhcc_binary_weight_and_spoof_direction():
    logits = torch.tensor([[0.,2.],[0.,-2.]])
    y = torch.tensor([1,0])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:,1]-logits[:,0],y.float(),reduction="none") * torch.tensor([.1,.9])
    torch.testing.assert_close(classification_loss(logits,y),expected.mean())
    assert LHCCDetector.spoof_score(logits)[0] > LHCCDetector.spoof_score(logits)[1]


def tiny_nsf():
    return NSFPINNDetector(NSFPINNConfig(width=2,hidden=4,embedding=3,classifier_hidden=4,dropout=0.0))


def test_nsf_physics_matches_explicit_reductions_and_warmup():
    head = tiny_nsf()
    x = torch.tensor([[[1.,2.],[3.,4.],[2.,0.]],[[0.,1.],[2.,1.],[3.,3.]]])
    o = head(x,return_physics=True)
    loss1, terms = physics_loss(o,torch.tensor([0,1]),head.cfg,epoch=1)
    loss2, _ = physics_loss(o,torch.tensor([0,1]),head.cfg,epoch=2)
    smooth = ((o['source'][:,1:]-o['source'][:,:-1]).square().mean()
              +(o['filter'][:,1:]-o['filter'][:,:-1]).square().mean()).clamp(0,10)
    torch.testing.assert_close(terms['smoothness'],smooth)
    cos = torch.nn.functional.cosine_similarity(o['s'],o['f']).square().mean()
    torch.testing.assert_close(terms['orthogonality'],cos)
    aux = .1 * sum(terms[k] for k in ('reconstruction','smoothness','orthogonality'))
    torch.testing.assert_close(loss1,terms['classification']+.5*aux)
    torch.testing.assert_close(loss2,terms['classification']+aux)
    loss2.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert head.spoof_score(o['logits']).shape == (2,)


def test_nsf_zero_vectors_and_single_frame_have_finite_gradients():
    head = tiny_nsf()
    o = head(torch.zeros(2,1,2),return_physics=True)
    for key in ('s','f'):
        o[key] = torch.zeros_like(o[key],requires_grad=True)
    loss, terms = physics_loss(o,torch.tensor([0,1]),head.cfg,epoch=1)
    assert terms['smoothness'] == 0 and terms['orthogonality'] == 0
    loss.backward()
    assert torch.isfinite(o['s'].grad).all() and torch.isfinite(o['f'].grad).all()


@pytest.mark.parametrize("method,size",[("lhcc",64600),("nsf_pinn",64000)])
def test_native_padding_and_development_crop(method,size):
    short = torch.tensor([1.,2.,3.])
    actual = train.fixed_audio(short,method,training=False)
    assert len(actual) == size
    if method == 'lhcc':
        assert torch.equal(actual[:6],short.repeat(2))
    else:
        assert torch.equal(actual[:3],short) and actual[3:].count_nonzero() == 0
    long = torch.arange(size+10,dtype=torch.float32)
    start = 0 if method == 'lhcc' else 5
    assert torch.equal(train.fixed_audio(long,method,training=False),long[start:start+size])
    for invalid in (torch.empty(0),torch.tensor([float('nan')]),torch.ones(1,5)):
        with pytest.raises(ValueError):
            train.fixed_audio(invalid,method,training=True)


def test_random_nsf_crop_includes_last_valid_start(monkeypatch):
    monkeypatch.setattr(torch,'randint',lambda high,*a,**k:torch.tensor(high-1))
    long = torch.arange(64009,dtype=torch.float32)
    assert torch.equal(train.fixed_audio(long,'nsf_pinn',training=True),long[9:])


@pytest.mark.parametrize('method',['lhcc','nsf_pinn'])
@pytest.mark.parametrize('rate',[16000,44100])
@pytest.mark.parametrize('training',[False,True])
def test_seek_audio_exactly_matches_full_decode_resample_crop(tmp_path,method,rate,training):
    path=tmp_path/'synthetic.wav'
    torch.manual_seed(8)
    wave=torch.randn(rate*5,2)*.01
    sf.write(path,wave.numpy(),rate,subtype='FLOAT')
    complete,sr=train.load_audio_segment(path)
    torch.manual_seed(12)
    expected=train.fixed_audio(train.resample_audio(complete,sr,16000),method,training=training)
    torch.manual_seed(12)
    actual=train.load_fixed_audio(path,method,training=training)
    assert torch.equal(actual,expected)


@pytest.mark.parametrize('fold',['f1','f2','f3','f4'])
def test_canonical_source_only_reads(fold,monkeypatch):
    calls=[]
    monkeypatch.setattr(shared,'read_dataset',lambda d,s,r:calls.append((d,s)) or [('unused',0)])
    train.source_records(fold,'unused')
    assert sorted(calls)==sorted((d,s) for d in train.FOLDS[fold]['sources'] for s in ('train','dev'))
    assert all(d!=train.FOLDS[fold]['target'] for d,s in calls)


def test_sam_reuses_frozen_features_and_does_not_update_encoder():
    raw = FakeOfficial(12,768)
    encoder = FrozenSpeechEncoder('nsf_pinn',raw)
    head = NSFPINNDetector(NSFPINNConfig(hidden=4,embedding=3,classifier_hidden=4,dropout=0))
    wrapper,opt,sched,sam = train.make_update(head,'nsf_pinn',total_steps=4)
    frozen_before=raw.weight.detach().clone()
    detector_before=head.source.weight.detach().clone()
    features=encoder(torch.randn(2,400)).float()
    losses=train.update_batch(wrapper,opt,sched,sam,features,torch.tensor([0,1]),epoch=1)
    assert raw.calls==1 and torch.equal(raw.weight,frozen_before) and raw.weight.grad is None
    assert not torch.equal(head.source.weight,detector_before)
    assert torch.isfinite(losses).all() and not sam.perturbed
    assert sched.last_epoch==1


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls=0
    def forward(self,waves):
        self.calls+=1
        return waves.reshape(-1,2,2)


def fixture_loaders():
    waves=torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]])
    labels=torch.tensor([0,1])
    training=DataLoader(TensorDataset(waves,labels),batch_size=2)
    dev={d:DataLoader(TensorDataset(waves,labels),batch_size=1) for d in ('asv2019','asv5','cfad')}
    return training,dev


def test_strict_macro_selection_ties_keep_earlier(tmp_path,monkeypatch):
    monkeypatch.setitem(train.RECIPES,'nsf_pinn',{**train.RECIPES['nsf_pinn'],'max_epochs':4,'patience':1})
    monkeypatch.setattr(train,'check_guards',lambda _:None)
    values=iter([.3,.2,.2])
    monkeypatch.setattr(train,'source_metrics',lambda _: {'macro_eer':next(values)})
    training,dev=fixture_loaders()
    head=tiny_nsf()
    result=train.fit(TinyEncoder(),head,'nsf_pinn',training,dev,'cpu',tmp_path,{'fold':'f1','seed':1234})
    assert (result['best_epoch'],result['final_epoch'])==(2,3)
    saved=torch.load(tmp_path/'best.pt',weights_only=True)
    assert saved['epoch']==2 and 'encoder_state_dict' not in saved
    assert saved['format']=='frozen_encoder_detector_only_v1'
    assert not (tmp_path/'training_complete.json').exists()


@pytest.mark.parametrize('corrupt',[False,True])
def test_full_source_completion_export_calibration_and_refusal(tmp_path,monkeypatch,corrupt):
    monkeypatch.setitem(train.RECIPES,'nsf_pinn',{**train.RECIPES['nsf_pinn'],'max_epochs':1})
    monkeypatch.setattr(train,'check_guards',lambda _:None)
    monkeypatch.setattr(train,'pretrained_identity',lambda _: {'fixture':'no-real-weights'})
    training,dev=fixture_loaders()
    records={d:{'dev':[(f'/fixture/{train.FOLDS["f1"]["sources"][i]}/a',0),
                       (f'/fixture/{train.FOLDS["f1"]["sources"][i]}/b',1)]}
             for i,d in enumerate(dev)}
    # Replace only ID formatting; no real dataset access belongs in this CPU test.
    monkeypatch.setattr(train,'utterance_id',lambda p,*a:p.rsplit('/',1)[-1])
    config={'fold':'f1','seed':1234,'method':'nsf_pinn','run_name':'nsf_pinn_frozen_v1',
            'pretrained':{'fixture':'no-real-weights'},'data_root':'/fixture','source_identity':{'synthetic':True}}
    train.write_json(tmp_path/'config.json',config)
    head=tiny_nsf(); encoder=TinyEncoder()
    result=train.fit(encoder,head,'nsf_pinn',training,dev,'cpu',tmp_path,config)
    if corrupt:
        checkpoint=torch.load(tmp_path/'best.pt',weights_only=True)
        checkpoint['source_dev_scores']['asv2019'][0]+=1
        torch.save(checkpoint,tmp_path/'best.pt')
        with pytest.raises(RuntimeError,match='reload differs'):
            train.finalize(encoder,tiny_nsf(),'nsf_pinn',tmp_path,config,records,dev,'cpu',result)
        assert not (tmp_path/'training_complete.json').exists()
    else:
        train.finalize(encoder,tiny_nsf(),'nsf_pinn',tmp_path,config,records,dev,'cpu',result)
        complete=json.loads((tmp_path/'training_complete.json').read_text())
        assert complete['selected_checkpoint_reloaded_and_source_verified'] and not complete['target_access']
        assert len(complete['source_scores_sha256'])==3
        calibration=json.loads((tmp_path/'source_calibration.json').read_text())
        assert calibration
        assert not (tmp_path/'target_scores.csv').exists()
