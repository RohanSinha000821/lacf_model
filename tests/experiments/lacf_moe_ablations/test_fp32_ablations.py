"""FP32 execution for all controls and fail-closed producer dependency gates."""
import json
import logging
from pathlib import Path

import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches, FeatureSplit
from scripts.experiments.lacf_moe_ablations import train_fp32 as runner


def bank():
    return {(d,s):FeatureSplit([(str(i), i%2) for i in range(8)],
        [f'{d}_{s}_{i}' for i in range(8)], torch.tensor([0,1]*4),
        torch.arange(8)[:,None].repeat(1,3 if s=='train' else 1),
        torch.randn(8,13,768), torch.randn(8,512))
        for d in runner.FOLDS['f1']['sources'] for s in ('train','dev')}


def provenance():
    return {'code_sha256':{}, 'target_access':False}


@pytest.mark.parametrize('variant',runner.CORE[1:])
def test_all_fifteen_controls_fp32_forward_backward_source_fit_and_reload(tmp_path, variant):
    b=bank();prototypes=unit_vectors(torch.randn(8,512))
    model=runner.build_model(variant,prototypes)
    dtypes=[]
    handles=[m.register_forward_hook(lambda m,args,out:dtypes.append(out.dtype))
             for m in model.modules() if isinstance(m,torch.nn.Linear)]
    train=FeatureBatches([b[d,'train'] for d in runner.FOLDS['f1']['sources']],
        runner.FEATURE_INPUTS[variant],'cpu',training=True,seed=1234,batch_size=4,max_batches=2)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        losses,updates=runner.train_epoch(model,train,runner.make_optimizer(model),'cpu')
        scores,metrics,_=runner.score_sources(model,b,'f1',variant,'cpu',collect_routing=True)
    for h in handles:h.remove()
    assert dtypes and set(dtypes)=={torch.float32}
    assert updates==2 and torch.isfinite(torch.tensor(list(losses.values()))).all()
    assert sum(p.numel() for p in model.parameters())==runner.PARAMETERS[variant]
    assert model.configuration()['objective']==runner.OBJECTIVES[variant]
    assert all(s.dtype==torch.float32 for s in scores.values())
    if variant not in ('C1','C3'):assert losses['semantic']==0
    if variant not in ('C2','C3'):assert losses['bonafide']==0
    proof=runner.preflight(b,prototypes,'f1',variant,provenance(),'cpu',logging.getLogger('test'))
    assert proof['linear_output_dtypes']==['torch.float32'] and proof['serialized_source_reload_bit_exact']
    path=tmp_path/f'{variant}_preflight.json';runner.save_json(path,proof)
    evidence={**provenance(),'source_validation_record':str(path),'source_validation_sha256':runner.checkpoint_digest(path)}
    output=tmp_path/variant/'f1'/'1234'
    done=runner.fit(b,prototypes,output,'f1',variant,evidence,'cpu',logging.getLogger('test'),max_epochs=1)
    assert done['selected_checkpoint_reloaded_and_source_verified'] and not done['target_access']
    config=runner.read(output/'config.json')
    assert config['precision']==runner.PRECISION and config['model']['score_precision']==runner.PRECISION
    assert config['model']['objective']==runner.OBJECTIVES[variant]
    assert not (output/'target_scores.csv').exists()
    assert runner.verify_completion(output,'f1',variant,runner.labels_for(b,'f1'))==done
    with pytest.raises(FileExistsError):
        runner.fit(b,prototypes,output,'f1',variant,evidence,'cpu',logging.getLogger('test'),max_epochs=1)


def make_dependency(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'REFERENCE',tmp_path)
    (tmp_path/'launch').mkdir()
    return {'dependency_inputs_sha256':{}}


def test_waiting_has_no_gpu_or_cache_access(tmp_path,monkeypatch):
    evidence=make_dependency(tmp_path,monkeypatch)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('GPU initialized while waiting'))
    monkeypatch.setattr(runner,'load_bank',lambda *a,**k:pytest.fail('Cache loaded while waiting'))
    assert runner.dependency_ready(evidence) is False


@pytest.mark.parametrize('exit_status',[{'trainer_exit':1,'logger_exit':0,'verified_complete':False},
    {'trainer_exit':0,'logger_exit':1,'verified_complete':False},
    {'trainer_exit':0,'logger_exit':0,'verified_complete':False}])
def test_failed_or_unverified_parent_stops_dependency(tmp_path,monkeypatch,exit_status):
    evidence=make_dependency(tmp_path,monkeypatch)
    runner.save_json(tmp_path/'launch'/'exit.json',exit_status)
    with pytest.raises(ValueError,match='failed/stopped'):runner.dependency_ready(evidence)


def test_success_requires_all_four_bound_completions(tmp_path,monkeypatch):
    evidence=make_dependency(tmp_path,monkeypatch)
    runner.save_json(tmp_path/'launch'/'exit.json',{'trainer_exit':0,'logger_exit':0,'verified_complete':True})
    done={'folds':list(runner.FOLDS),'precision':runner.PRECISION,'target_access':False,'completion_sha256':{}}
    runner.save_json(tmp_path/'launch'/'complete.json',done)
    with pytest.raises(ValueError,match='all four'):runner.dependency_ready(evidence)
    for fold in runner.FOLDS:
        p=tmp_path/fold/'1234';p.mkdir(parents=True)
        runner.save_json(p/'training_complete.json',{'fold':fold})
        done['completion_sha256'][fold]=runner.checkpoint_digest(p/'training_complete.json')
    (tmp_path/'launch'/'complete.json').write_text(json.dumps(done))
    assert runner.dependency_ready(evidence) is True
    (tmp_path/'f1'/'1234'/'training_complete.json').write_text('{}')
    with pytest.raises(ValueError,match='completion changed'):runner.dependency_ready(evidence)


def test_launch_dependency_hash_change_rejected(tmp_path,monkeypatch):
    evidence=make_dependency(tmp_path,monkeypatch)
    p=tmp_path/'launch'/'run.sh';p.write_text('original')
    evidence['dependency_inputs_sha256']={'run.sh':runner.checkpoint_digest(p)}
    p.write_text('changed')
    with pytest.raises(ValueError,match='launch changed'):runner.dependency_ready(evidence)


def test_precision_preflight_mismatch_rejected(tmp_path):
    p=tmp_path/'proof.json';p.write_text(json.dumps({'precision':'BF16'}))
    evidence={'source_validation_record':str(p),'source_validation_sha256':runner.checkpoint_digest(p)}
    with pytest.raises(ValueError,match='preflight evidence'):runner.validate_preflight(evidence,'f1','A1','cpu')
