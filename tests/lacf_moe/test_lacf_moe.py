"""CPU-only parity, source-isolation and completion gates for main LACF-MoE."""

import json
import logging

import pytest
import torch

from audio_deepfake_detection.experiments.layer_fusion.model import LayerFusionDetector
from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureSplit
from audio_deepfake_detection.lacf_moe.model import GROUPS, LACFMoE, TRAINABLE_PARAMETERS
from scripts.lacf_moe import train as runner


def prototypes():
    return unit_vectors(torch.randn(8,512))


def bank(fold):
    return {(d,s):FeatureSplit([(str(i),i%2) for i in range(8)],
        [f'{d}_{s}_{7-i}' for i in range(8)],torch.tensor([0,1]*4),
        torch.arange(8)[:,None].repeat(1,3 if s=='train' else 1),torch.randn(8,13,768),torch.randn(8,512))
        for d in runner.FOLDS[fold]['sources'] for s in ('train','dev')}


def labels(data,fold):
    return {d:dict(zip(data[d,'dev'].ids,data[d,'dev'].labels.tolist())) for d in runner.FOLDS[fold]['sources']}


def test_reference_state_configuration_rng_and_bf16_update_bit_exact():
    torch.set_num_threads(2)
    fixed=prototypes()
    torch.manual_seed(1234);old=LayerFusionDetector('C4',fixed);rng=torch.random.get_rng_state()
    torch.manual_seed(1234);new=LACFMoE(fixed)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert old.configuration()==new.configuration()
    assert TRAINABLE_PARAMETERS==967338==sum(p.numel() for p in new.parameters())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    inputs={'wavlm':torch.randn(8,13,768),'clap':torch.randn(8,512),'labels':torch.tensor([0,1]*4)}
    optimizers=[runner.make_optimizer(model) for model in (old,new)]
    outputs=[]
    for model,optimizer in zip((old,new),optimizers):
        torch.manual_seed(77)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            output=model(inputs);loss=model.objective(output,inputs['labels'])
        assert loss['semantic']==loss['bonafide']==0 and loss['total'].dtype==torch.float32
        loss['total'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();outputs.append(output)
    assert all(torch.equal(v,outputs[1][k]) for k,v in outputs[0].items())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    assert new.backend.prototypes.requires_grad is False


def test_exact_three_groups_and_uniform_initial_routing():
    assert GROUPS==((0,4),(4,8),(8,13))
    model=LACFMoE(prototypes())
    for expert in model.layer_fusion.experts:
        for p in expert.parameters():p.data.zero_()
    inputs=torch.arange(13,dtype=torch.float32)[None,:,None].expand(2,13,768)
    values,weights=model.layer_fusion(inputs)
    assert values.shape==(2,768) and weights.shape==(2,3)
    assert torch.equal(weights,torch.full((2,3),1/3))
    assert torch.allclose(values,torch.full((2,768),(1.5+5.5+10)/3))
    with pytest.raises(ValueError,match='thirteen'):model.layer_fusion(torch.zeros(2,12,768))


@pytest.mark.parametrize('fold',tuple(runner.FOLDS))
def test_all_fold_fresh_fit_reload_canonical_calibration_and_no_overwrite(tmp_path,fold):
    torch.set_num_threads(2)
    data=bank(fold);run=tmp_path/fold/'1234'
    done=runner.fit(data,prototypes(),run,fold,{'code_sha256':{}},'cpu',logging.getLogger('main_moe'),max_epochs=2,patience=1)
    assert done['target_access'] is False and done['selected_checkpoint_reloaded_and_source_verified']
    assert runner.verify_completion(run,fold,expected_labels=labels(data,fold))==done
    config=runner.read(run/'config.json')
    assert config['sources']==list(runner.FOLDS[fold]['sources']) and config['excluded_target']==runner.FOLDS[fold]['target']
    routing=runner.read(run/'routing.json')
    assert set(routing['domains'])==set(config['sources'])
    assert all(len(v['all']['mean_weights'])==3 for v in routing['domains'].values())
    with pytest.raises(FileExistsError):runner.fit(data,prototypes(),run,fold,{'code_sha256':{}},'cpu',logging.getLogger('repeat'))
    path=run/f"source_dev_{config['sources'][0]}.csv";path.write_text('changed')
    with pytest.raises(ValueError,match='Source export changed'):runner.verify_completion(run,fold,expected_labels=labels(data,fold))


@pytest.mark.parametrize('fold',('f2','f3','f4'))
def test_other_fold_data_never_creates_run(tmp_path,fold):
    run=tmp_path/fold/'1234'
    with pytest.raises(ValueError,match='Only this fold source'):
        runner.fit(bank('f1'),prototypes(),run,fold,{'code_sha256':{}},'cpu',logging.getLogger('wrong'))
    assert not run.exists()


def test_reload_mismatch_never_completes(tmp_path,monkeypatch):
    score=runner.score_sources;calls=0
    def mismatch(*args,**kwargs):
        nonlocal calls
        values,metrics=score(*args,**kwargs);calls+=1
        if calls==2:
            domain=next(iter(values));values[domain]=values[domain].clone();values[domain][0]+=1
        return values,metrics
    monkeypatch.setattr(runner,'score_sources',mismatch)
    run=tmp_path/'f2/1234'
    with pytest.raises(ValueError,match='Fresh selected-checkpoint'):
        runner.fit(bank('f2'),prototypes(),run,'f2',{'code_sha256':{}},'cpu',logging.getLogger('bad'),max_epochs=1)
    assert not (run/'training_complete.json').exists()


def test_ties_keep_first_checkpoint(tmp_path,monkeypatch):
    original=runner.score_sources
    def tied(*args,**kwargs):
        values,metrics=original(*args,**kwargs)
        for domain in metrics['by_domain']:metrics['by_domain'][domain]['eer']=.5
        metrics['macro_eer']=.5
        return values,metrics
    monkeypatch.setattr(runner,'score_sources',tied)
    # Calibration still verifies the actual saved source scores; inspect selection
    # before that independent gate, which correctly rejects synthetic fake EERs.
    with pytest.raises(ValueError):
        runner.fit(bank('f2'),prototypes(),tmp_path/'f2/1234','f2',{'code_sha256':{}},'cpu',logging.getLogger('ties'),max_epochs=6,patience=2)
    state=torch.load(tmp_path/'f2/1234/best.pt',weights_only=True)
    history=[json.loads(line) for line in (tmp_path/'f2/1234/history.jsonl').read_text().splitlines()]
    assert state['epoch']==1 and len(history)==3 and history[-1]['patience']==2


@pytest.mark.parametrize('failure',(None,'f2','f3'))
def test_queue_advances_only_after_verified_fold(tmp_path,monkeypatch,failure):
    folds=('f2','f3','f4');calls=[]
    data={k:v for fold in folds for k,v in bank(fold).items()}
    def preflight(subset,proto,fold,*args):
        runner.assert_bank(subset,fold);calls.append(('preflight',fold));return {'passed':True}
    def fit(subset,proto,output,fold,*args):
        runner.assert_bank(subset,fold);calls.append(('fit',fold))
    def verify(output,fold,**kwargs):
        calls.append(('verify',fold))
        if fold==failure:raise ValueError('completion failed')
    monkeypatch.setattr(runner,'preflight',preflight);monkeypatch.setattr(runner,'fit',fit);monkeypatch.setattr(runner,'verify_completion',verify)
    args=(data,prototypes(),folds,{},'cpu',logging.getLogger('queue'),tmp_path/'runs',tmp_path)
    if failure:
        with pytest.raises(ValueError,match='completion failed'):runner.run_folds(*args)
    else:runner.run_folds(*args)
    expected=folds if failure is None else folds[:folds.index(failure)+1]
    assert calls==[(stage,fold) for fold in expected for stage in ('preflight','fit','verify')]
