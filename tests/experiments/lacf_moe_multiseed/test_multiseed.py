"""Actual seed propagation, historical parity, checkpoint bindings and aggregation."""
import json
import logging
from pathlib import Path

import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureSplit,FeatureBatches
from scripts.experiments.lacf_moe_multiseed import train as runner


def bank():
    return {(d,s):FeatureSplit([(str(i),i%2)for i in range(8)],
        [f'{d}_{s}_{i}'for i in range(8)],torch.tensor([0,1]*4),
        torch.arange(8)[:,None].repeat(1,3 if s=='train'else 1),
        torch.randn(8,13,768),torch.randn(8,512))
        for d in runner.FOLDS['f1']['sources']for s in('train','dev')}


@pytest.mark.parametrize('model',runner.MODELS)
def test_seed1234_matches_historical_training_arithmetic(model):
    b=bank();p=unit_vectors(torch.randn(8,512))
    runner.seed_all(1234);original=runner.FACTORIES[model](p)
    old_loader=FeatureBatches([b[d,'train']for d in runner.FOLDS['f1']['sources']],'concat','cpu',training=True,seed=1234)
    a=runner.train_epoch(original,old_loader,runner.make_optimizer(original),'cpu',accumulation_steps=1)
    runner.seed_all(1234);new=runner.build_model(model,p,'cpu')
    new_loader=runner.loader(b,'f1',1234,'cpu',training=True)
    z=runner.train_epoch(new,new_loader,runner.make_optimizer(new),'cpu',accumulation_steps=1)
    assert a==z and all(torch.equal(v,new.state_dict()[k])for k,v in original.state_dict().items())
    values,metrics=runner.score_sources(new,b,'f1',1234,'cpu')
    if model=='three_experts':old_values,old_metrics=runner.original3.score_sources(original,b,'f1','cpu')
    else:old_values,old_metrics=runner.original4.score_sources(original,b,'cpu',fold='f1')
    assert metrics==old_metrics and all(torch.equal(values[d],old_values[d])for d in values)


def test_seed_reaches_sampler_view_choice_initialization_and_dropout():
    b=bank();p=unit_vectors(torch.randn(8,512));weights=[];rngs=[];batches=[]
    for seed in runner.NEW_SEEDS:
        runner.seed_all(seed);net=runner.build_model('three_experts',p,'cpu')
        weights.append(net.backend.classifier[1].weight.detach().clone());rngs.append(torch.random.get_rng_state())
        data=runner.loader(b,'f1',seed,'cpu',training=True)
        assert data.sampler.generator.initial_seed()==seed
        assert data.view_generator.initial_seed()==seed+1
        batches.append(next(iter(data))['wavlm'].clone())
    assert not torch.equal(weights[0],weights[1])and not torch.equal(rngs[0],rngs[1])
    assert not torch.equal(batches[0],batches[1])
    runner.seed_all(1235);net=runner.build_model('three_experts',p,'cpu')
    assert torch.equal(net.backend.classifier[1].weight,weights[0])


@pytest.mark.parametrize('model',runner.MODELS)
@pytest.mark.parametrize('seed',runner.NEW_SEEDS)
def test_complete_fit_uses_actual_seed_and_strict_source_artifacts(tmp_path,monkeypatch,model,seed):
    b=bank();p=unit_vectors(torch.randn(8,512));logger=logging.getLogger('test_multiseed')
    cache=tmp_path/'cache';cache.mkdir();(cache/'complete.json').write_text('{}');monkeypatch.setattr(runner,'CACHE',cache)
    prov={'code_sha256':{},'cache':str(cache),'cache_completion_sha256':runner.checkpoint_digest(cache/'complete.json')}
    proof=runner.preflight(b,p,model,'f1',seed,prov,'cpu',logger)
    assert proof['seed']==seed and proof['serialized_full_source_reload_bit_exact'] and proof['actual_logit_dtype']=='torch.bfloat16'
    path=tmp_path/'proof.json';runner.save_json(path,proof)
    prov.update(source_validation_record=str(path),source_validation_sha256=runner.checkpoint_digest(path))
    run=tmp_path/model/'f1'/str(seed)
    done=runner.fit(b,p,run,model,'f1',seed,prov,'cpu',logger,max_epochs=1)
    cfg,verified,state=runner.verify_training(run,model,'f1',seed,runner.labels_for(b,'f1'))
    assert done==verified and cfg['seed']==done['seed']==seed and cfg['seed_plan']==[1234,1235,1236]
    assert cfg['physical_cache_crop_seed']==1234 and cfg['target_access']is False
    assert runner.read(run/'score_manifest.json')['seed']==seed
    with pytest.raises(ValueError):runner.verify_training(run,model,'f1',1236 if seed==1235 else 1235,runner.labels_for(b,'f1'))
    with pytest.raises(FileExistsError):runner.fit(b,p,run,model,'f1',seed,prov,'cpu',logger,max_epochs=1)


def test_seed1234_never_retrained(tmp_path):
    with pytest.raises(ValueError,match='reused'):
        runner.fit({},None,tmp_path,'three_experts','f1',1234,{},'cpu',logging.getLogger('test'))
    assert not list(tmp_path.iterdir())


def test_source_failure_stops_before_target(tmp_path,monkeypatch):
    p=unit_vectors(torch.randn(8,512));state={'model_state_dict':runner.FACTORIES['three_experts'](p).state_dict()}
    monkeypatch.setattr(runner,'verify_training',lambda *a:({}, {},state))
    def fail(*a,**k):raise ValueError('fresh source failed')
    monkeypatch.setattr(runner,'score_sources',fail)
    monkeypatch.setattr(runner.target_helpers,'verify_producer',lambda *a:pytest.fail('target accessed'))
    with pytest.raises(ValueError,match='source failed'):
        runner.evaluate(tmp_path,tmp_path/'new','three_experts','f1',1235,bank(),{},'cpu',logging.getLogger('test'))
    assert not(tmp_path/'new').exists()


def test_summary_uses_all_three_seeds_sample_std_and_equal_fold_weight(tmp_path,monkeypatch):
    for model in runner.MODELS:
        monkeypatch.setitem(runner.EVALUATIONS,model,tmp_path/model)
        for f_index,fold in enumerate(runner.FOLDS):
            for seed in runner.SEEDS:
                p=runner.paths(model,fold,seed)[1];p.mkdir(parents=True)
                runner.save_json(p/'metrics.json',{'fold':fold,'seed':seed,'value':(seed-1234)*.01+f_index*.1})
    monkeypatch.setattr(runner,'flat_metrics',lambda r:{'target.eer':r['value']})
    summary=runner.summarize()
    for model in runner.MODELS:
        m=summary['models'][model]
        assert m['by_fold']['f1']['target.eer']['mean']==pytest.approx(.01)
        assert m['by_fold']['f1']['target.eer']['sample_std']==pytest.approx(.01)
        assert m['equal_fold_mean']['target.eer']['mean']==pytest.approx(.16)
        assert m['equal_fold_mean']['target.eer']['sample_std']==pytest.approx(.01)
