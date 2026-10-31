"""Synthetic CPU checks for weight-aware all-layer LoRA and CLAP scope."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import WavLMConfig, WavLMModel

from audio_deepfake_detection.lacf.model import ComponentConfig, LACFLoss
from audio_deepfake_detection.lacf.diagnostic_cache import FixedCropDataset
from audio_deepfake_detection.token_fusion.adaptation import AdaptationConfig, AdaptedTF2, add_wavlm_lora
from audio_deepfake_detection.token_fusion.adaptation_data import SelectedCropDataset, ViewSampler, inner_sources, verify_source_membership
from scripts.token_fusion.train_adaptation import train_audio_epoch


def tiny_wavlm():
    return WavLMModel(WavLMConfig(hidden_size=32, num_hidden_layers=12, num_attention_heads=4,
        intermediate_size=64, conv_dim=(8,8), conv_stride=(5,2), conv_kernel=(10,3),
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=4, mask_time_prob=0))


class TinyAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = nn.Linear(16,32)
        self.batch_norm = nn.BatchNorm2d(1)
        self.audio_encoder = nn.Module()
        self.audio_encoder.layers = nn.ModuleList([nn.Linear(32,32) for _ in range(4)])
        self.audio_encoder.norm = nn.LayerNorm(32)

    def forward(self, input_features, **kwargs):
        x = self.front(self.batch_norm(input_features).flatten(1))
        for layer in self.audio_encoder.layers:
            x = F.gelu(layer(x))
        return SimpleNamespace(pooler_output=self.audio_encoder.norm(x))


class TinyClap(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_model = TinyAudio()
        self.audio_projection = nn.Sequential(nn.Linear(32,512), nn.ReLU(), nn.Linear(512,512))
        self.text_model = nn.Linear(2,2)


class TinyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(13*32+512,1)
        self.objective = LACFLoss(ComponentConfig(semantic_loss=False,bonafide_loss=False))

    def forward(self, batch):
        values = torch.cat((batch['wavlm'].flatten(1),batch['clap']),1)
        return {'logit':self.linear(values).squeeze(-1)}


def model(clap=True):
    torch.manual_seed(1234)
    return AdaptedTF2(tiny_wavlm(), TinyClap(), AdaptationConfig(clap), fusion=TinyFusion())


def batch():
    x = torch.randn(2,500)
    x[0,400:] = 0
    mask = torch.arange(500)[None] < torch.tensor([400,500])[:,None]
    return {'wavlm':{'input_values':x,'attention_mask':mask.long()},
            'clap':{'input_features':torch.randn(2,1,4,4)},'labels':torch.tensor([0,1])}


@pytest.mark.parametrize('clap',[False,True])
def test_all12_qv_lora_get_gradients_and_only_approved_clap_weights_change(clap):
    m = model(clap); inputs = batch()
    assert len(m.lora_targets)==24
    enabled = {n for n,p in m.clap.named_parameters() if p.requires_grad}
    assert all(n.startswith(('audio_model.audio_encoder.layers.3.','audio_projection.')) for n in enabled)
    assert bool(enabled)==clap
    assert all('parametrizations.weight.0.lora_' in n for n,p in m.wavlm.named_parameters() if p.requires_grad)
    frozen = {n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    buffers = {n:p.clone() for n,p in m.named_buffers()}
    optimizer = m.optimizer()
    for step in range(2):
        m.train()
        assert not m.wavlm.training and not m.clap.training and m.fusion.training
        optimizer.zero_grad(set_to_none=True)
        loss = m.objective(m(inputs),inputs['labels'])['total']
        loss.backward()
        for name,p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
                if '.lora_b' in name or step>0 and '.lora_a' in name:
                    assert p.grad.abs().sum()>0,name
            else:
                assert p.grad is None,name
        optimizer.step()
    for name,p in m.named_parameters():
        if name in frozen: assert torch.equal(p,frozen[name]),name
    for name,p in m.named_buffers():assert torch.equal(p,buffers[name]),name
    lrs={g['name']:g['lr'] for g in optimizer.param_groups}
    assert lrs['wavlm_lora']==1e-4 and lrs['tf2']==3e-4
    if clap:assert lrs['clap_final_stage_projection']==1e-5


def test_zero_delta_matches_original_functional_attention_and_native_padding():
    torch.manual_seed(6)
    original=tiny_wavlm().requires_grad_(False).eval()
    m=AdaptedTF2(copy.deepcopy(original),TinyClap(),AdaptationConfig(False),fusion=TinyFusion()).eval()
    inputs=batch()['wavlm']
    actual=m.wavlm_features(inputs)
    for i,length in enumerate([400,500]):
        native={k:v[i:i+1,:length] for k,v in inputs.items()}
        captured={}
        hook=original.feature_projection.register_forward_hook(lambda m,i,o:captured.update(front=o[0]))
        with torch.no_grad():
            output=original(**native,output_hidden_states=True)
            expected=torch.stack([v.mean(1) for v in (captured['front'],*output.hidden_states[1:])],1)
        hook.remove()
        torch.testing.assert_close(actual[i:i+1],expected,atol=1e-6,rtol=1e-6)
    with pytest.raises(ValueError,match='double injection'):
        add_wavlm_lora(m.wavlm,m.config)


def test_sparse_checkpoint_loads_on_fresh_identical_base_and_rejects_missing_state():
    first=model();inputs=batch();opt=first.optimizer()
    first.objective(first(inputs),inputs['labels'])['total'].backward();opt.step()
    first.eval();expected=first(inputs)['logit'].detach()
    fresh=model().eval();state=first.trainable_state();fresh.load_trainable_state(state)
    assert torch.equal(fresh(inputs)['logit'],expected)
    assert not any('parametrizations.weight.original' in n or 'text_model' in n for n in state)
    state.pop(next(iter(state)))
    with pytest.raises(ValueError,match='missing/unexpected'):fresh.load_trainable_state(state)


def test_selected_crop_reader_matches_existing_cache_views(tmp_path):
    p=tmp_path/'wave.wav';sf.write(p,np.random.default_rng(2).normal(size=192000).astype(np.float32),16000,subtype='FLOAT')
    records=[(str(p),1)];names=['asv5/train/example']
    reference=FixedCropDataset(records,names,training=True)[0]
    selected=SelectedCropDataset(records,names,training=True)
    for view,row in enumerate(reference['view_rows']):
        for a,b in zip(selected[0,view][:2],reference['views'][row][:2]):assert torch.equal(a,b)


def test_view_sampler_balances_domains_classes_and_preserves_all_draws():
    domains=[SimpleNamespace(records=[('a',0),('b',1),('c',1)]),SimpleNamespace(records=[('d',0),('e',0),('f',1)])]
    sampler=ViewSampler(domains,1234);weights=sampler.sampler.weights
    assert weights.tolist()==[1,.5,.5,.5,.5,1]
    assert list(sampler)==list(ViewSampler(domains,1234))


def test_source_rotation_cannot_open_target_or_wrong_partitions(monkeypatch,tmp_path):
    import audio_deepfake_detection.token_fusion.adaptation_data as d
    calls=[]
    def read(domain,part,root):
        calls.append((domain,part));return [(str(tmp_path/domain/'a'),0),(str(tmp_path/domain/'b'),1)]
    monkeypatch.setattr(d,'read_dataset',read)
    monkeypatch.setattr(d,'utterance_id',lambda p,domain,root: p.rsplit('/',1)[-1])
    fit,train,dev=inner_sources(tmp_path,'asv2019')
    assert calls==[('asv2019','dev'),('asv5','train'),('cfad','train')]
    assert len(train)==4 and len(dev)==2
    with pytest.raises(ValueError,match='Only F1'):inner_sources(tmp_path,'speechfake')


def test_accumulation_short_last_microbatch_matches_sample_weighted_updates():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(1,1)
            self.objective=LACFLoss(ComponentConfig(semantic_loss=False,bonafide_loss=False))
        def forward(self,b):
            with torch.autocast('cpu',enabled=False):return {'logit':self.linear(b['x']).squeeze(-1)}
    torch.manual_seed(9);a=Linear();b=copy.deepcopy(a)
    data=[{'x':torch.tensor([float(i)/10]),'labels':torch.tensor(i%2)} for i in range(11)]
    opt=torch.optim.SGD(a.parameters(),lr=.1)
    stats=train_audio_epoch(a,DataLoader(data,batch_size=3),opt,'cpu',effective_batch=6)
    ref=torch.optim.SGD(b.parameters(),lr=.1)
    for batch in DataLoader(data,batch_size=6):
        ref.zero_grad();b.objective(b(batch),batch['labels'])['total'].backward();ref.step()
    assert stats['samples']==11 and stats['updates']==2
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-7)


def test_bounded_timing_consumes_only_requested_batches():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.ones(()))
            self.objective=LACFLoss(ComponentConfig(semantic_loss=False,bonafide_loss=False))
        def forward(self,b):
            return {"logit": self.weight.float()*b["x"].float()}
    class Loader:
        batch_size=2
        dataset=list(range(20))
        seen=0
        def __iter__(self):
            for _ in range(10):
                self.seen+=1
                yield {"x":torch.tensor([.2,.5]),"labels":torch.tensor([0,1])}
    m=Linear();loader=Loader();opt=torch.optim.SGD(m.parameters(),lr=.1)
    result=train_audio_epoch(m,loader,opt,"cpu",effective_batch=4,max_batches=3)
    assert loader.seen==3 and result["samples"]==6 and result["updates"]==2


def test_source_identity_check_binds_reference_labels_and_order():
    dataset=SelectedCropDataset([("a",0),("b",1)],["asv5/train/a","asv5/train/b"],training=True)
    digest=hashlib.sha256(json.dumps([("a",0),("b",1)]).encode()).hexdigest()
    marker={"splits":{"asv5/train":{"records":2,"canonical_ids_labels_sha256":digest}}}
    assert verify_source_membership([dataset],marker)["asv5/train"]["records"]==2
    dataset.records[0]=("a",1)
    with pytest.raises(ValueError,match="Source membership differs"):
        verify_source_membership([dataset],marker)
