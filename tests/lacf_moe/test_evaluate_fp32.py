"""FP32 target scores and fail-closed source/producer evaluation gates."""
import json
import logging

import numpy as np
import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureSplit
from audio_deepfake_detection.lacf_moe.model import LACFMoE
from audio_deepfake_detection.metrics import write_score_csv
from scripts.lacf_moe import evaluate_fp32 as runner


def bank():
    return {(d,'dev'):FeatureSplit([(str(i),i%2)for i in range(8)],
        [f'{d}_{i}'for i in range(8)],torch.tensor([0,1]*4),torch.arange(8)[:,None],
        torch.randn(8,13,768),torch.randn(8,512))for d in runner.FOLDS['f1']['sources']}


def test_real_fp32_target_linears_inside_outer_bf16_context():
    model=LACFMoE(unit_vectors(torch.randn(8,512))).eval()
    arrays={'wavlm_layers':np.random.default_rng(1234).standard_normal((135,13,768)).astype('float32'),
            'clap_projected':np.random.default_rng(1235).standard_normal((135,512)).astype('float32')}
    dtypes=[]
    handles=[m.register_forward_hook(lambda m,args,out:dtypes.append(out.dtype))
             for m in model.modules()if isinstance(m,torch.nn.Linear)]
    with torch.autocast('cpu',dtype=torch.bfloat16):scores=runner.score_arrays(model,arrays,'cpu')
    for h in handles:h.remove()
    assert set(dtypes)=={torch.float32} and scores.dtype==np.float32 and scores.shape==(135,)
    assert not torch.equal(torch.from_numpy(scores),torch.from_numpy(scores).bfloat16().float())
    with torch.inference_mode():
        expected=torch.cat([model({'wavlm':torch.from_numpy(arrays['wavlm_layers'][s:s+132]),
            'clap':torch.from_numpy(arrays['clap_projected'][s:s+132])})['logit']for s in range(0,135,132)])
    assert np.array_equal(scores,expected.numpy())


@pytest.mark.parametrize('bad', ['dtype','shape','nan','batch'])
def test_invalid_target_inputs_stop_without_skipping(bad):
    model=LACFMoE(unit_vectors(torch.randn(8,512))).eval()
    arrays={'wavlm_layers':np.ones((2,13,768),np.float32),'clap_projected':np.ones((2,512),np.float32)}
    if bad=='dtype':arrays['clap_projected']=arrays['clap_projected'].astype('float64')
    if bad=='shape':arrays['wavlm_layers']=arrays['wavlm_layers'][:,:12]
    if bad=='nan':arrays['clap_projected'][0,0]=np.nan
    with pytest.raises(ValueError):runner.score_arrays(model,arrays,'cpu',batch_size=1 if bad == 'batch' else 132)


def test_source_full_score_parity_not_just_unchanged_eer(tmp_path,monkeypatch):
    b=bank();model=LACFMoE(unit_vectors(torch.randn(8,512))).eval()
    scores,metrics=runner.training.score_sources(model,b,'f1','cpu')
    runner.save_json(tmp_path/'metrics.json',{'source_dev':metrics})
    for d in runner.FOLDS['f1']['sources']:
        ds=b[d,'dev'];write_score_csv(tmp_path/f'source_dev_{d}.csv',zip(ds.ids,[d]*8,ds.labels.tolist(),scores[d].tolist()))
    monkeypatch.setattr(runner,'source_bank',lambda *a,**k:b)
    validation=runner.source_reload(model,tmp_path,{},'f1','cpu',logging.getLogger('test'),{})
    assert validation['full_source_score_bit_exact'] and len(validation['checks'])==3
    # A constant score shift can leave EER identical; exact score comparison must still fail.
    with torch.no_grad():model.backend.classifier[-1].bias.add_(0.125)
    with pytest.raises(ValueError,match='source (scores|metrics) differ'):
        runner.source_reload(model,tmp_path,{},'f1','cpu',logging.getLogger('test'),{})


def test_source_failure_prevents_target_access(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir()
    for n in ['best.pt','training_complete.json']:(run/n).write_text('placeholder')
    provenance={'training_completion_sha256':{'f1':runner.checkpoint_digest(run/'training_complete.json')},
                'checkpoint_sha256':{'f1':runner.checkpoint_digest(run/'best.pt')}}
    monkeypatch.setattr(runner,'verify_training',lambda *a:({},{}))
    monkeypatch.setattr(runner,'load_detector',lambda *a:None)
    def source_failure(*a):raise ValueError('source parity failed')
    monkeypatch.setattr(runner,'source_reload',source_failure)
    monkeypatch.setattr(runner,'finish_evaluation',lambda *a,**k:pytest.fail('Target touched before source gate'))
    output=tmp_path/'evaluation'
    with pytest.raises(ValueError,match='source parity failed'):
        runner.evaluate(run,output,'f1',provenance,'cpu',logging.getLogger('test'),{})
    assert not output.exists()


def test_producer_failure_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'FAMILY',tmp_path)
    p=tmp_path/'launch';p.mkdir()
    runner.save_json(p/'exit.json',{'trainer_exit':1,'logger_exit':0,'verified_complete':False})
    runner.save_json(p/'complete.json',{})
    with pytest.raises(ValueError,match='not fully verified'):
        runner.verify_main_producer({'training_producer_sha256':{},'training_completion_sha256':{}})


def test_finish_refuses_missing_source_proof_before_target_access(tmp_path):
    with pytest.raises(ValueError,match='Full source verification'):
        runner.finish_evaluation(None,tmp_path,tmp_path/'new','f1','name','family',{}, {},
            {'passed':False},{},'cpu',logging.getLogger('test'))
    assert not (tmp_path/'new').exists()
