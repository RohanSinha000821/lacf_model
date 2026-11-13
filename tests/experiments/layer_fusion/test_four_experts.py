"""CPU checks for the isolated four-expert F1 control."""

import json
import logging
import math

import pytest
import torch

from audio_deepfake_detection.experiments.layer_fusion.four_experts import FourExpertDetector, FourExpertFusion, GROUPS, TRAINABLE_PARAMETERS
from audio_deepfake_detection.experiments.layer_fusion.model import LayerFusionDetector
from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureSplit
from scripts.experiments.layer_fusion import train_four_experts as runner


def bank(fold='f1'):
    return {(d,s):FeatureSplit([(str(i),i%2) for i in range(8)],
        [f'{d}_{s}_{i}' for i in range(8)],torch.tensor([0,1]*4),
        torch.arange(8)[:,None].repeat(1,3 if s=='train' else 1),torch.randn(8,13,768),torch.randn(8,512))
        for d in runner.sources_for(fold) for s in ('train','dev')}


def prototypes():
    return unit_vectors(torch.randn(8,512))


def test_partition_uniform_router_and_exact_weighted_sum():
    assert [i for start,end in GROUPS for i in range(start,end)]==list(range(13))
    fusion=FourExpertFusion()
    for expert in fusion.experts:
        for parameter in expert.parameters():parameter.data.zero_()
    layers=torch.arange(13,dtype=torch.float32)[None,:,None].expand(2,13,768)
    fused,weights=fusion(layers)
    assert fused.shape==(2,768) and weights.shape==(2,4)
    assert torch.equal(weights,torch.full((2,4),.25))
    assert torch.equal(fused,torch.full((2,768),(1.5+5+8+11)/4))
    with pytest.raises(ValueError,match='thirteen'):fusion(torch.zeros(2,12,768))


def test_common_backend_initialization_and_rng_match_reference():
    fixed=prototypes()
    torch.manual_seed(1234);old=LayerFusionDetector('C4',fixed);old_rng=torch.random.get_rng_state()
    torch.manual_seed(1234);new=FourExpertDetector(fixed);new_rng=torch.random.get_rng_state()
    assert torch.equal(old_rng,new_rng)
    assert all(torch.equal(value,new.backend.state_dict()[key]) for key,value in old.backend.state_dict().items())
    for i in range(3):
        assert all(torch.equal(value,new.layer_fusion.experts[i].state_dict()[key])
                   for key,value in old.layer_fusion.experts[i].state_dict().items())
    assert new.configuration()['trainable_parameters']==TRAINABLE_PARAMETERS==1068779
    assert old.configuration()['trainable_parameters']==967338
    assert not new.backend.config.semantic_loss and not new.backend.config.bonafide_loss


def test_recording_dependent_router_and_finite_detection_gradients():
    model=FourExpertDetector(prototypes())
    with torch.no_grad():model.layer_fusion.router[-1].weight[0,0]=2
    batch={'wavlm':torch.randn(6,13,768),'clap':torch.randn(6,512),'labels':torch.tensor([0,1]*3)}
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=model(batch);losses=model.objective(output,batch['labels'])
    assert output['routing'].shape==(6,4) and output['features'].shape==(6,35)
    assert torch.allclose(output['routing'].sum(-1),torch.ones(6))
    assert not torch.equal(output['routing'][0],output['routing'][1])
    assert losses['total'].dtype==torch.float32
    assert losses['semantic']==losses['bonafide']==0
    losses['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not model.backend.prototypes.requires_grad


def test_four_column_routing_statistics_include_classes_and_ties():
    source=bank();model=FourExpertDetector(prototypes())
    result=runner.routing_summary(model,source,'cpu')
    for domain in runner.SOURCES:
        for name,count in (('all',8),('bonafide',4),('spoof',4)):
            row=result['domains'][domain][name]
            assert row['count']==count and row['mean_weights']==[.25]*4
            assert row['top_expert_counts']==[count,0,0,0]
            assert abs(row['mean_entropy_nats']-math.log(4))<1e-6


@pytest.mark.parametrize('fold',tuple(runner.FOLDS))
def test_fresh_fit_source_reload_exports_and_tamper_gate(tmp_path,fold):
    torch.set_num_threads(2)
    run=tmp_path/fold/'1234'
    done=runner.fit(bank(fold),prototypes(),run,{'code_sha256':{}},'cpu',logging.getLogger('synthetic_moe4'),fold=fold,max_epochs=2,patience=1)
    assert done['selected_checkpoint_reloaded_and_source_verified'] and done['target_access'] is False
    config=json.loads((run/'config.json').read_text())
    assert config['model']['expert_groups_half_open']==[[0,4],[4,7],[7,10],[10,13]]
    assert config['sources']==list(runner.sources_for(fold))
    assert config['excluded_target']==runner.FOLDS[fold]['target']
    assert runner.verify_completion(run,fold)==done
    with pytest.raises(FileExistsError):
        runner.fit(bank(fold),prototypes(),run,{'code_sha256':{}},'cpu',logging.getLogger('existing'),fold=fold)
    (run/f'source_dev_{runner.sources_for(fold)[0]}.csv').write_text('changed')
    with pytest.raises(ValueError,match='Source export changed'):runner.verify_completion(run,fold)


def test_reloaded_source_mismatch_never_completes(tmp_path,monkeypatch):
    score=runner.score_sources;calls=0
    def changed(*args,**kwargs):
        nonlocal calls
        values,metrics=score(*args,**kwargs);calls+=1
        if calls==2:
            values['asv2019']=values['asv2019'].clone()
            values['asv2019'][0]+=1
        return values,metrics
    monkeypatch.setattr(runner,'score_sources',changed)
    run=tmp_path/'f1/1234'
    with pytest.raises(ValueError,match='source reload differs'):
        runner.fit(bank(),prototypes(),run,{'code_sha256':{}},'cpu',logging.getLogger('mismatch'),max_epochs=1)
    assert not (run/'training_complete.json').exists()


def test_completion_reuses_sorted_id_calibration_without_relaxing_equality(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    run=tmp_path/'f2/1234'
    data=bank('f2')
    for split in data.values():
        split.ids.reverse()  # CSV order differs from the canonical calibration order.
    runner.fit(data,prototypes(),run,{'code_sha256':{}},'cpu',logging.getLogger('ordering'),fold='f2',max_epochs=1)
    original_read=runner.read_score_csv
    calls=[]
    def ordered(path,domain,**kwargs):
        assert kwargs.get('expected_labels') is not None
        calls.append(domain)
        return original_read(path,domain,**kwargs)
    monkeypatch.setattr(runner,'read_score_csv',ordered)
    before={p.name:runner.checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    runner.verify_completion(run,'f2')
    assert calls==list(runner.sources_for('f2'))
    assert before=={p.name:runner.checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    calibrate=runner.calibrate_source_scores
    def changed(source):
        values=calibrate(source)
        # Even a single representable step remains an error once order is fixed.
        values['normalized_thresholds']['5%']=math.nextafter(values['normalized_thresholds']['5%'],math.inf)
        return values
    monkeypatch.setattr(runner,'calibrate_source_scores',changed)
    with pytest.raises(ValueError,match='Frozen source calibration differs'):
        runner.verify_completion(run,'f2')


@pytest.mark.parametrize('fold',('f2','f3','f4'))
def test_wrong_fold_bank_rejected_before_creation(tmp_path,fold):
    run=tmp_path/fold/'1234'
    with pytest.raises(ValueError,match='selected fold source'):
        runner.fit(bank('f1'),prototypes(),run,{'code_sha256':{}},'cpu',logging.getLogger('wrong'),fold=fold)
    assert not run.exists()


def test_union_cache_loads_only_required_train_dev_once(tmp_path,monkeypatch):
    from audio_deepfake_detection.protocol import checkpoint_digest
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'execution_accelerated.json').write_text('{}')
    domains=set(d for fold in ('f2','f3','f4') for d in runner.sources_for(fold))
    manifest={'step':2,'config_sha256':checkpoint_digest(tmp_path/'config.json'),
              'accelerated_execution_sha256':checkpoint_digest(tmp_path/'execution_accelerated.json'),
              'splits':{f'{d}/{s}':{} for d in domains for s in ('train','dev')}}
    (tmp_path/'complete.json').write_text(json.dumps(manifest))
    all_data={key:value for fold in ('f2','f3','f4') for key,value in bank(fold).items()}
    calls=[]
    def load(root,domain,split,*args,**kwargs):
        calls.append((domain,split));return all_data[domain,split]
    monkeypatch.setattr(runner,'load_split',load)
    result=runner.load_sources(tmp_path,tmp_path,['f2','f3','f4'],logging.getLogger('cache'))
    assert set(result)==set(calls)=={(d,s) for d in domains for s in ('train','dev')}
    assert len(calls)==8


@pytest.mark.parametrize('failed_gate',(None,'f2','f3'))
def test_queue_advances_only_after_verified_completion(tmp_path,monkeypatch,failed_gate):
    folds=['f2','f3','f4'];calls=[]
    source={key:value for fold in folds for key,value in bank(fold).items()}
    monkeypatch.setattr(runner,'FAMILY',tmp_path/'runs')
    def preflight(subset,proto,device,logger,*,fold,launch):
        runner.assert_bank(subset,fold);calls.append(('preflight',fold));return {'passed':True,'fold':fold}
    def fit(subset,proto,output,provenance,device,logger,*,fold):
        runner.assert_bank(subset,fold);calls.append(('fit',fold))
        assert provenance['source_validation_sha256']
    def verify(output,fold):
        calls.append(('verify',fold))
        if fold==failed_gate:raise ValueError('Completion gate failed')
    monkeypatch.setattr(runner,'preflight',preflight)
    monkeypatch.setattr(runner,'fit',fit)
    monkeypatch.setattr(runner,'verify_completion',verify)
    if failed_gate:
        with pytest.raises(ValueError,match='Completion gate failed'):
            runner.run_folds(source,prototypes(),folds,{},'cpu',logging.getLogger('queue'),tmp_path)
    else:runner.run_folds(source,prototypes(),folds,{},'cpu',logging.getLogger('queue'),tmp_path)
    expected=folds if failed_gate is None else folds[:folds.index(failed_gate)+1]
    assert calls==[(stage,fold) for fold in expected for stage in ('preflight','fit','verify')]
