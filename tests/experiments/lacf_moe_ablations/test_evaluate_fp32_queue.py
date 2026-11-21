"""Actual FP32 heads, registry isolation, producer and source-to-target gates."""
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureSplit
from scripts.experiments.lacf_moe_ablations import evaluate_fp32 as runner

CASES = [('ablations',v) for v in runner.VARIANTS['ablations']] + [
    ('embedding_paths',v) for v in runner.VARIANTS['embedding_paths']]


@pytest.mark.parametrize('family,variant', CASES)
def test_every_head_actual_fp32_target_and_full_source_parity(tmp_path, family, variant):
    prototypes = unit_vectors(torch.randn(8,512))
    arrays = {'wavlm_layers':np.random.default_rng(1).normal(size=(135,13,768)).astype('float32'),
              'clap_projected':np.random.default_rng(2).normal(size=(135,512)).astype('float32')}
    bank = {(d,'dev'):FeatureSplit([(str(i),i%2) for i in range(135)],
        [f'{d}_{i:03}' for i in range(135)],torch.arange(135)%2,torch.arange(135)[:,None],
        torch.from_numpy(arrays['wavlm_layers']),torch.from_numpy(arrays['clap_projected']))
        for d in runner.FOLDS['f1']['sources']}
    with runner.registry(family):
        model = runner.core.build_model(variant,prototypes).eval()
        inputs = runner.core.FEATURE_INPUTS[variant]
        dtypes = []
        hooks = [m.register_forward_hook(lambda m,a,out:dtypes.append(out.dtype))
                 for m in model.modules() if isinstance(m,torch.nn.Linear)]
        with torch.autocast('cpu',dtype=torch.bfloat16):
            scores = runner.score_arrays(model,arrays,'cpu',inputs)
            source,metrics,_ = runner.core.score_sources(model,bank,'f1',variant,'cpu')
        for hook in hooks:hook.remove()
        assert set(dtypes)=={torch.float32} and scores.dtype==np.float32
        assert not torch.equal(torch.from_numpy(scores),torch.from_numpy(scores).bfloat16().float())
        for domain in runner.FOLDS['f1']['sources']:
            assert np.array_equal(scores,source[domain].numpy())
            ds=bank[domain,'dev']
            runner.shared.write_score_csv(tmp_path/f'source_dev_{domain}.csv',
                zip(ds.ids,[domain]*135,ds.labels.tolist(),source[domain].tolist()))
        runner.save_json(tmp_path/'metrics.json',{'source_dev':metrics})
        proof=runner.source_reload(model,tmp_path,bank,'f1',variant,'cpu',logging.getLogger('test'))
        assert proof['passed'] and proof['full_source_score_bit_exact'] and len(proof['checks'])==3
        # Preserve rank/EER but shift absolute scores: calibration parity must catch this.
        with torch.no_grad():
            head = model.backend.classifier if hasattr(model,'backend') else model.classifier
            head[-1].bias.add_(0.125)
        with pytest.raises(ValueError,match='source (scores|metrics) differ'):
            runner.source_reload(model,tmp_path,bank,'f1',variant,'cpu',logging.getLogger('test'))


def test_exporter_and_model_registries_restore_after_failure():
    old_scorer,old_factory=runner.shared.score_arrays,runner.core.build_model
    with pytest.raises(RuntimeError):
        with runner.registry('embedding_paths'), runner.fp32_exporter():
            assert runner.shared.score_arrays is runner.score_arrays
            assert runner.core.build_model is runner.paths.models.build_model
            raise RuntimeError('synthetic failure')
    assert runner.shared.score_arrays is old_scorer and runner.core.build_model is old_factory


def producer(tmp_path,monkeypatch):
    root=tmp_path/'producer';launch=root/'launch';launch.mkdir(parents=True)
    monkeypatch.setitem(runner.FAMILIES,'ablations',root)
    runner.save_json(launch/'provenance.json',{'code_sha256':{}})
    evidence={'training_producers':{'ablations':{'launch_sha256':{},'pid':os.getpid(),'module':'python'}}}
    run,_=runner.case_paths('ablations','A1','f1');run.mkdir(parents=True)
    runner.save_json(run/'training_complete.json',{'fold':'f1'})
    entry={'variant':'A1','fold':'f1','completion_sha256':runner.checkpoint_digest(run/'training_complete.json')}
    runner.save_json(launch/'queue_status.json',{'completed_fits':1,'runs':[entry]})
    return launch,run,evidence


def test_producer_entry_is_required_and_immutable(tmp_path,monkeypatch):
    launch,run,evidence=producer(tmp_path,monkeypatch);observed={}
    assert runner.ready_cases('ablations',evidence,observed)[0][:3]==('ablations','A1','f1')
    (launch/'queue_status.json').write_text(json.dumps({'completed_fits':0,'runs':[]}))
    with pytest.raises(ValueError,match='removed or changed'):runner.ready_cases('ablations',evidence,observed)


@pytest.mark.parametrize('failure',['exit','completion','scope','disappeared'])
def test_failed_changed_or_unauthorized_producer_stops(tmp_path,monkeypatch,failure):
    launch,run,evidence=producer(tmp_path,monkeypatch)
    if failure=='exit':runner.save_json(launch/'exit.json',{'trainer_exit':1,'logger_exit':0,'verified_complete':False})
    if failure=='completion':(run/'training_complete.json').write_text('{}')
    if failure=='scope':
        status=runner.read(launch/'queue_status.json');status['runs'][0]['variant']='A0'
        (launch/'queue_status.json').write_text(json.dumps(status))
    if failure=='disappeared':evidence['training_producers']['ablations']['pid']=999999999
    with pytest.raises((ValueError,RuntimeError)):runner.ready_cases('ablations',evidence,{})


def test_source_failure_prevents_target_access(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'verify_training',lambda *a:(tmp_path,tmp_path/'new',{}, {},torch.nn.Linear(1,1),'concat'))
    def fail(*a):raise ValueError('source parity failed')
    monkeypatch.setattr(runner,'source_reload',fail)
    monkeypatch.setattr(runner.shared,'finish_evaluation',lambda *a,**k:pytest.fail('Target accessed'))
    with pytest.raises(ValueError,match='source parity'):
        runner.evaluate_case(('ablations','A1','f1','digest'),{}, {},logging.getLogger('test'),device='cpu')
    assert not (tmp_path/'new').exists()


def test_low_resources_releases_dev_cache_without_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'CACHE',tmp_path)
    runner.save_json(tmp_path/'complete.json',{'splits':{f'{d}/dev':{'unique_rows':100}
        for d in runner.FOLDS['f1']['sources']}})
    monkeypatch.setattr(runner.paths,'available_ram',lambda:0)
    monkeypatch.setattr(runner.shared,'source_bank',lambda *a,**k:pytest.fail('Cache loaded without headroom'))
    retained={('asv2019','dev'):object()}
    assert runner.load_dev_bank(retained,'f1',{},logging.getLogger('test')) is None
    assert not retained
