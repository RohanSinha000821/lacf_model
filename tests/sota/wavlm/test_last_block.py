"""F4 freeze/gradient boundaries, unchanged WA inference and preparation guards."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel

from audio_deepfake_detection.sota.wavlm_wa import WavLMWA
from audio_deepfake_detection.sota.wavlm_wa_last import freeze_except_last_block, PRETRAINED_REVISION
from scripts.wavlm import train, last_block_support as support


def tiny_model():
    config = WavLMConfig(hidden_size=16, num_hidden_layers=12, num_attention_heads=2,
        intermediate_size=32, conv_dim=(8,8,8), conv_stride=(5,2,2), conv_kernel=(10,3,3),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, num_buckets=32,
        hidden_dropout=0, attention_dropout=0, activation_dropout=0, feat_proj_dropout=0,
        layerdrop=0, apply_spec_augment=False)
    model = WavLMWA.__new__(WavLMWA); nn.Module.__init__(model)
    model.wavlm = WavLMModel(config)
    model.layer_weights = nn.Parameter(torch.zeros(13)); model.classifier = nn.Linear(16,2)
    return model


def test_freeze_boundary_actual_transformers_forward_and_update():
    torch.set_num_threads(2)
    model = tiny_model(); original = copy.deepcopy(model)
    freeze_except_last_block(model); model.train()
    assert model.wavlm.encoder.layers[0].training
    assert model.wavlm.feature_extractor._requires_grad is False
    for name, p in model.named_parameters():
        assert p.requires_grad == (name == 'layer_weights' or name.startswith('classifier.')
                                  or name.startswith('wavlm.encoder.layers.11.'))
    before = {k:p.detach().clone() for k,p in model.named_parameters()}
    observations = []
    handle = model.wavlm.encoder.layers[0].register_forward_hook(
        lambda _, inputs, output: observations.append(output[0].requires_grad))
    waves = torch.randn(2,1600); mask = torch.ones_like(waves,dtype=torch.bool)
    expected = original.eval()(waves,mask)
    actual = model(waves,mask)
    torch.testing.assert_close(actual,expected)
    nn.CrossEntropyLoss()(actual,torch.tensor([0,1])).backward()
    assert observations == [False]
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001).step()
    assert all(torch.equal(before[k],p) for k,p in model.named_parameters() if not p.requires_grad)
    assert any(not torch.equal(before[k],p) for k,p in model.named_parameters() if p.requires_grad)
    handle.remove()
    # Existing full-model WavLM evaluator can load the exact same checkpoint keys.
    original.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(original.eval()(waves,mask),model.eval()(waves,mask))


def test_last_block_cli_keeps_existing_family_and_recipe(monkeypatch,tmp_path):
    args=['train','--fold','f4','--last-block-only','--source-validation-record',str(tmp_path/'proof.json')]
    monkeypatch.setattr(sys,'argv',args)
    c=train.parse_args()
    assert c.last_block_only and (c.run_name,c.seed,c.batch_size)==('wavlm_bs96',1234,96)
    assert (c.num_workers,c.eval_workers,c.prefetch_factor)==(16,4,2)
    for other in ['f1','f2','f3']:
        monkeypatch.setattr(sys,'argv',[s if s!='f4' else other for s in args])
        with pytest.raises(SystemExit):train.parse_args()


def test_last_block_refuses_missing_source_evidence(monkeypatch):
    monkeypatch.setattr(sys,'argv',['train','--fold','f4','--last-block-only'])
    with pytest.raises(SystemExit):train.parse_args()


def test_preparation_requires_fresh_discarded_correct_source_evidence(tmp_path,monkeypatch):
    weights=tmp_path/'weights';weights.mkdir();(weights/'config.json').write_text('original')
    monkeypatch.setattr(support,'ROOT',tmp_path);monkeypatch.setattr(support,'PRETRAINED_PATH',weights)
    (tmp_path/'model.py').write_text('original')
    proof=dict(passed=True,training_started=False,fold='f4',seed=1234,batch_size=96,
        pretrained_revision=PRETRAINED_REVISION,data_root=str(tmp_path),probe_weights_discarded=True,
        fresh_long_source_reload_bit_exact=True,code_sha256={'model.py':support.checkpoint_digest(tmp_path/'model.py')},
        pretrained_sha256={'config.json':support.checkpoint_digest(weights/'config.json')})
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof))
    assert support.verify_preparation(path,tmp_path)==proof
    (tmp_path/'model.py').write_text('changed')
    with pytest.raises(ValueError,match='implementation changed'):support.verify_preparation(path,tmp_path)
    proof['fold']='f1';path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='Matching source-only'):support.verify_preparation(path,tmp_path)


@pytest.mark.parametrize('tamper',[False,True])
def test_selected_full_source_exports_bit_exact_and_manifest(tmp_path,monkeypatch,tamper):
    from audio_deepfake_detection.protocol import dataset_root
    class Detector(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(1,1,bias=False)
            with torch.no_grad():self.linear.weight.fill_(1.)
        def forward(self,waves,mask):
            s=self.linear(waves)
            return torch.cat((torch.zeros_like(s),s),dim=-1)
        spoof_score=staticmethod(WavLMWA.spoof_score)
    monkeypatch.setattr(support,'LastBlockWavLMWA',Detector)
    output=tmp_path/'outputs/wavlm_bs96/f4/1234';output.mkdir(parents=True)
    records={};loaders={};selected={}
    for domain in support.FOLDS['f4']['sources']:
        records[domain]=[(str(dataset_root(tmp_path,domain)/name),label)
                         for name,label in [('z.wav',1),('a.wav',0)]]
        loaders[domain]=[(torch.tensor([[v]]),torch.ones(1,1,dtype=torch.bool),torch.tensor([label]))
                         for v,label in [(1.,1),(-1.,0)]]
        selected[domain]={'z.wav':1.,'a.wav':-1.}
    state={'model_state_dict':Detector().state_dict(),'dev_eers':{d:0. for d in records},
           'config':{'adaptation':{'fine_tuning':'last_block'}}}
    torch.save(state,output/'best.pt')
    before=(output/'best.pt').read_bytes()
    if tamper:
        selected['asv5']['z.wav']=99.
        with pytest.raises(ValueError,match='bit-for-bit'):
            support.finalize_sources(output,loaders,records,tmp_path,torch.device('cpu'),0.,selected)
        assert not (output/'source_calibration.json').exists()
    else:
        done=support.finalize_sources(output,loaders,records,tmp_path,torch.device('cpu'),0.,selected)
        assert done['full_source_scores_bit_exact'] and not done['target_access']
        assert len(done['source_scores_sha256'])==3
        (output/'training_complete.json').write_text('{}')
        support.register_source_exports(output,done)
        manifest=json.loads((output/'score_manifest.json').read_text())
        assert manifest['score_sha256']==done['source_scores_sha256']
        assert 'target_scores.csv' not in manifest['score_sha256']
    assert (output/'best.pt').read_bytes()==before
