"""Additional models, isolated registry binding and existing fit/export gates."""
import hashlib
import json
import logging

import numpy as np
import pytest
import torch

from audio_deepfake_detection.experiments.lacf_moe_ablations import additional_model as models
from audio_deepfake_detection.experiments.lacf_moe_ablations.model import CORE,build_model as build_core
from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches,FeatureSplit,score_probe
from audio_deepfake_detection.lacf_moe.model import LACFMoE
from scripts.experiments.lacf_moe_ablations import train as core
from scripts.experiments.lacf_moe_ablations import evaluate as core_eval
from scripts.experiments.lacf_moe_ablations import train_additional as runner
from scripts.experiments.lacf_moe_ablations import evaluate_additional as evaluator


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(2)


def prototypes():return unit_vectors(torch.randn(8,512))


def bank(fold):
    result={}
    for d in core.FOLDS[fold]['sources']:
        for split in ('train','dev'):
            n=7;y=torch.arange(n)%2
            result[d,split]=FeatureSplit([(str(i),int(label)) for i,label in enumerate(y)],
                [f'{d}_{split}_{n-i}' for i in range(n)],y,
                torch.arange(n)[:,None].repeat(1,3 if split=='train' else 1),torch.randn(n,13,768),torch.randn(n,512))
    return result


def proof(tmp,fold,variant):
    p=tmp/f'proof_{variant}_{fold}.json'
    core.save_json(p,{'passed':True,'fold':fold,'variant':variant,'device':'cpu',
        'trainable_parameters':models.PARAMETERS[variant],'full_source_dev_reload_bit_exact':True,
        'probe_weights_used_for_final_training':False,'target_access':False,
        'code_sha256':{},'cache_completion_sha256':None})
    return {'code_sha256':{},'source_validation_record':str(p),'source_validation_sha256':core.checkpoint_digest(p)}


@pytest.mark.parametrize('variant',models.VARIANTS)
def test_dimensions_parameters_enabled_branches_gradients_and_logits(variant):
    fixed=prototypes();model=models.build_model(variant,fixed)
    batch={'labels':torch.tensor([0,1,0,1,0])}
    if variant!='B2':batch['wavlm']=torch.randn(5,13,768)
    if variant!='B1':batch['clap']=torch.randn(5,512)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=model(batch);loss=model.objective(output,batch['labels'])
    assert output['features'].shape==(5,{'B1':512,'B2':8,'B3':16,'B4':32}.get(variant,35))
    assert output['logit'].dtype==torch.bfloat16
    assert loss['total'].dtype==torch.float32 and loss['semantic']==loss['bonafide']==0
    loss['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters())==models.PARAMETERS[variant]
    config=model.configuration()
    assert config['wavlm_enabled']==(variant!='B2') and config['clap_audio_enabled']==(variant!='B1')
    assert not config['foundation_encoders_loaded'] and config['objective']=='detection BCE only'
    if variant!='B1':assert torch.equal(model.prototypes,fixed)
    if variant=='B2':assert not hasattr(model,'adapter') and not hasattr(model,'layer_fusion')


def test_shared_initialization_views_and_global_route_control():
    fixed=prototypes();batch={'wavlm':torch.randn(7,13,768),'clap':torch.randn(7,512)}
    torch.manual_seed(1234);reference=LACFMoE(fixed).eval();rng=torch.get_rng_state()
    with torch.autocast('cpu',dtype=torch.bfloat16):expected=reference(batch)
    for v in models.VARIANTS:
        torch.manual_seed(1234);model=models.build_model(v,fixed).eval()
        assert torch.equal(torch.get_rng_state(),rng)
        with torch.autocast('cpu',dtype=torch.bfloat16):out=model(batch)
        if v!='B2':
            assert all(torch.equal(p,model.adapter.state_dict()[k]) for k,p in reference.backend.adapter.state_dict().items())
        for i in range(4,len(model.classifier)):
            assert all(torch.equal(p,model.classifier[i].state_dict()[k]) for k,p in reference.backend.classifier[i].state_dict().items())
        if v in ('B3','B4','B5'):
            for k in ('v_w','v_c','p_w','p_c'):assert torch.equal(expected[k],out[k]),(v,k)
        if v=='B5':
            assert torch.equal(out['routing'],out['routing'][0].expand(7,-1))
            assert not hasattr(model.layer_fusion,'router') or model.layer_fusion.router is None
            with torch.no_grad():model.layer_fusion.expert_weights[0]=1
            assert model(batch)['routing'][0,0]>.5


def test_feature_groups_and_fusion_only_math():
    fixed=prototypes();batch={'wavlm':torch.randn(5,13,768),'clap':torch.randn(5,512)}
    for v in ('B3','B4'):
        out=models.build_model(v,fixed)(batch);p,q=out['p_w'],out['p_c']
        expected=torch.cat([p,q] if v=='B3' else [p,q,abs(p-q),p*q],-1)
        assert torch.equal(out['features'],expected)
    model=models.build_model('B6',fixed).eval()
    assert not hasattr(model,'layer_fusion')
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=model(batch);p=unit_vectors(model.adapter(batch['wavlm'].mean(1)))
        assert torch.equal(p,out['v_w'])
        joint=model.relations(out);features,routes=model.fusion_moe(joint)
        assert torch.equal(features,out['features']) and torch.equal(routes,out['fusion_routing'])


def test_registries_restored_after_exception_and_sources_unchanged():
    original={k:getattr(core,k) for k in ('CORE','NAMES','PARAMETERS','FEATURE_INPUTS','build_model','FAMILY')}
    original_eval={k:getattr(core_eval,k) for k in ('CORE','NAMES','build_model','FEATURE_INPUTS','OUTPUT','LAUNCH')}
    paths=[core.__file__,core_eval.__file__]
    before={p:hashlib.sha256(open(p,'rb').read()).hexdigest() for p in paths}
    with pytest.raises(RuntimeError):
        with evaluator.additional_evaluation():
            assert core.CORE[1:]==models.VARIANTS and core_eval.CORE[1:]==models.VARIANTS
            assert core.build_model is models.build_model
            raise RuntimeError('synthetic failure')
    for k,v in original.items():assert getattr(core,k) is v
    for k,v in original_eval.items():assert getattr(core_eval,k) is v
    assert before=={p:hashlib.sha256(open(p,'rb').read()).hexdigest() for p in paths}


@pytest.mark.parametrize('variant',models.VARIANTS)
@pytest.mark.parametrize('fold',tuple(core.FOLDS))
def test_all_folds_fit_fresh_reload_canonical_export_and_tamper(tmp_path,variant,fold):
    data=bank(fold);run=tmp_path/f'{variant}_{models.NAMES[variant]}'/fold/'1234'
    provenance=proof(tmp_path,fold,variant)
    with runner.additional_controls():
        done=core.fit(data,prototypes(),run,fold,variant,provenance,'cpu',logging.getLogger('fit'),max_epochs=1,patience=1)
        assert done['selected_checkpoint_reloaded_and_source_verified'] and not done['target_access']
        assert core.verify_completion(run,fold,variant,core.labels_for(data,fold))==done
        with pytest.raises(FileExistsError):
            core.fit(data,prototypes(),run,fold,variant,provenance,'cpu',logging.getLogger('retry'),max_epochs=1)
        source=run/f"source_dev_{core.FOLDS[fold]['sources'][0]}.csv";source.write_text('changed')
        with pytest.raises(ValueError,match='Source export changed'):core.verify_completion(run,fold,variant,core.labels_for(data,fold))
    assert core.CORE==CORE


def test_wait_requires_verified_status_and_stops_on_producer_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluator,'TRAIN_LAUNCH',tmp_path)
    run=tmp_path/'B1/f1/1234';run.mkdir(parents=True);(run/'training_complete.json').write_text('{}')
    (tmp_path/'exit.json').write_text('{"evaluator_exit":1}')
    with pytest.raises(RuntimeError,match='before B1/f1 verified'):
        evaluator.wait_for_run(run,'B1/f1',logging.getLogger('wait'))
    core.save_json(tmp_path/'queue_status.json',{'runs':[{'variant':'B1','fold':'f1',
        'completion_sha256':core.checkpoint_digest(run/'training_complete.json')}]})
    evaluator.wait_for_run(run,'B1/f1',logging.getLogger('wait'))
    (run/'training_complete.json').write_text('changed')
    with pytest.raises(ValueError,match='completion changed'):evaluator.wait_for_run(run,'B1/f1',logging.getLogger('wait'))


@pytest.mark.parametrize('variant',models.VARIANTS)
def test_cached_evaluation_preserves_BF16_head_batches_and_tail(variant):
    n=135;data=FeatureSplit([],list(map(str,range(n))),torch.arange(n)%2,torch.arange(n)[:,None],
                            torch.randn(n,13,768),torch.randn(n,512))
    model=models.build_model(variant,prototypes()).eval()
    expected,_=score_probe(model,FeatureBatches([data],models.FEATURE_INPUTS[variant],'cpu',training=False,seed=1234))
    arrays={'wavlm_layers':data.wavlm.numpy(),'clap_projected':data.clap.numpy()}
    actual=core_eval.score_arrays(model,arrays,'cpu',models.FEATURE_INPUTS[variant])
    assert np.array_equal(actual,expected.numpy())
    assert np.array_equal(actual,torch.from_numpy(actual).to(torch.bfloat16).float().numpy())
