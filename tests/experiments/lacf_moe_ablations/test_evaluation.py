"""Cached evaluation preserves trained readouts, source gates and target metrics."""
import hashlib
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from audio_deepfake_detection.experiments.lacf_moe_ablations.model import CORE, FEATURE_INPUTS, build_model
from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches, FeatureSplit, score_probe
from audio_deepfake_detection.protocol import checkpoint_digest
from scripts.experiments.lacf_moe_ablations import evaluate as ev
from scripts.experiments.lacf_cached import evaluate as primary
from scripts.experiments.lacf_cached import train as primary_train


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize('variant', CORE[1:])
def test_target_scoring_matches_trained_batch_rounding_and_tail(variant):
    torch.manual_seed(7);n=135
    data=FeatureSplit([],list(map(str,range(n))),torch.arange(n)%2,torch.arange(n)[:,None],
                      torch.randn(n,13,768),torch.randn(n,512))
    model=build_model(variant,unit_vectors(torch.randn(8,512))).eval()
    original,_=score_probe(model,FeatureBatches([data],FEATURE_INPUTS[variant],'cpu',training=False,seed=1234))
    arrays={'wavlm_layers':data.wavlm.numpy(),'clap_projected':data.clap.numpy()}
    actual=ev.score_arrays(model,arrays,'cpu',FEATURE_INPUTS[variant])
    assert np.array_equal(actual,original.numpy())
    assert actual.shape==(135,) and actual.dtype==np.float32
    assert np.array_equal(actual,torch.from_numpy(actual).to(torch.bfloat16).float().numpy())


def test_primary_target_matches_original_batch_one_source_scoring():
    model=primary_train.new_model(unit_vectors(torch.randn(8,512)),'cpu').eval()
    bank={};arrays={}
    for d in primary_train.SOURCES:
        n=7;wave=torch.randn(n,13,768);clap=torch.randn(n,512)
        bank[d,'dev']=FeatureSplit([],list(map(str,range(n))),torch.arange(n)%2,torch.arange(n)[:,None],wave[:,-1],clap)
        arrays[d]={'wavlm_layers':wave.numpy(),'clap_projected':clap.numpy()}
    original,_=primary_train.score_sources(model,bank,'cpu',logging.getLogger('test'))
    for d in primary_train.SOURCES:
        assert np.array_equal(ev.score_arrays(model,arrays[d],'cpu',batch_size=1,last_layer_only=True),original[d].numpy())
    with pytest.raises(ValueError,match='head batch'):ev.score_arrays(model,arrays[d],'cpu',batch_size=24)


def test_failed_source_gate_blocks_all_target_reads(tmp_path,monkeypatch):
    output=tmp_path/'out'
    monkeypatch.setattr(ev,'verify_producer',lambda *a:pytest.fail('target read before source gate'))
    with pytest.raises(ValueError,match='Full source verification'):
        ev.finish_evaluation(None,tmp_path,output,'f1','x','x',{}, {},{'passed':False},{},'cpu',logging.getLogger('test'))
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        ev.finish_evaluation(None,tmp_path,output,'f1','x','x',{}, {},{}, {},'cpu',logging.getLogger('test'))


def test_checkpoint_change_fails_before_source_or_target_reads(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();(run/'best.pt').write_bytes(b'changed');(run/'training_complete.json').write_text('{}')
    provenance={'training_completion_sha256':{'A1/f1':'wrong'},'checkpoint_sha256':{'A1/f1':'wrong'}}
    monkeypatch.setattr(ev,'verify_training',lambda *a:pytest.fail('verification started after changed launch binding'))
    with pytest.raises(ValueError,match='Launch-bound training changed'):
        ev.evaluate(run,tmp_path/'out','f1','A1',provenance,'cpu',logging.getLogger('test'),{})


def test_cached_export_uses_shared_metrics_and_preserves_originals(tmp_path,monkeypatch):
    from audio_deepfake_detection.metrics import calibrate_source_scores,write_score_csv,compute_eer
    import scripts.evaluate_scores as shared
    run=tmp_path/'train/f1/1234';run.mkdir(parents=True);output=tmp_path/'eval/f1/1234'
    model=build_model('A3',unit_vectors(torch.randn(8,512))).eval()
    torch.save({'model_state_dict':model.state_dict()},run/'best.pt')
    (run/'config.json').write_text('{}');sources=ev.FOLDS['f1']['sources'];labels={f's{i:03}':i%2 for i in range(10)}
    inputs={};hashes={}
    for d in sources:
        values=np.linspace(-2,2,10);file=run/f'source_dev_{d}.csv'
        write_score_csv(file,[(key,d,y,float(s)) for (key,y),s in zip(labels.items(),values)])
        inputs[d]=(np.array(list(labels.values())),values);hashes[file.name]=checkpoint_digest(file)
    cal=calibrate_source_scores(inputs);(run/'source_calibration.json').write_text(json.dumps(cal))
    (run/'metrics.json').write_text(json.dumps({'source_dev':cal['source_dev']}))
    done={'fold':'f1','seed':1234,'best_epoch':1,'final_epoch':1,'best_macro_source_dev_eer':cal['source_dev']['macro_eer'],
          'best_checkpoint_sha256':checkpoint_digest(run/'best.pt')}
    (run/'training_complete.json').write_text(json.dumps(done))
    producer=tmp_path/'producer';features=producer/'target_features';features.mkdir(parents=True)
    (features/'complete.json').write_text('{}')
    target={f't{i:03}':i%2 for i in range(135)}
    def canonical(d,split,root):
        if d=='speechfake':
            assert (output/'source_validation.json').is_file()
            return target
        return labels
    monkeypatch.setattr(ev,'canonical_score_labels',canonical);monkeypatch.setattr(shared,'canonical_score_labels',canonical)
    monkeypatch.setattr(ev,'verify_producer',lambda *a:(producer,features))
    arrays={'wavlm_layers':np.random.default_rng(7).normal(size=(135,13,768)).astype(np.float32),
            'clap_projected':np.zeros((135,512),np.float32)}
    monkeypatch.setattr(ev,'f1_chunks',lambda *a:iter([(list(target),arrays)]))
    before={p.name:checkpoint_digest(p) for p in run.iterdir()}
    result=ev.finish_evaluation(model,run,output,'f1','MoE-A3','lacf_moe_ablations',{'precision':'BF16 mixed'},
        done,{'passed':True,'full_source_score_bit_exact':True},{'code_sha256':{}},'cpu',logging.getLogger('test'),
        inputs='weighted',bootstrap_resamples=3)
    assert result['target']['n_bona_fide']+result['target']['n_spoof']==135
    assert result['confidence_intervals']['resamples']==3
    assert result['unlabeled_zscore_transfer']['transductive'] is True
    for branch in ('strict_raw_transfer','unlabeled_zscore_transfer'):
        assert set(result[branch]['target_operating_points'])=={'1%','5%','10%'}
    assert ev.read(output/'evaluation_complete.json')['metrics_sha256']==checkpoint_digest(output/'metrics.json')
    assert before=={p.name:checkpoint_digest(p) for p in run.iterdir()}
    assert (output/'source_metrics.json').read_text()==(run/'metrics.json').read_text()


@pytest.mark.parametrize('fold',('f1','f4'))
def test_target_producer_completion_and_payload_identity(tmp_path,monkeypatch,fold):
    producer=tmp_path/fold/'1234';producer.mkdir(parents=True)
    monkeypatch.setattr(ev,'PRODUCERS',tmp_path)
    features=tmp_path/'old_f1_features' if fold=='f1' else producer/'target_features'
    features.mkdir();monkeypatch.setattr(ev,'TARGET_FEATURES',features)
    (features/'complete.json').write_text('{}')
    for name in ('metrics.json','best.pt','target_scores.csv'):(producer/name).write_text('saved')
    (producer/'source_validation.json').write_text('{"passed":true}')
    scores={'target_scores.csv':checkpoint_digest(producer/'target_scores.csv')}
    done={'fold':fold,'seed':1234,'source_selected_checkpoint_only':True,
          'score_sha256':scores,'checkpoint_sha256':checkpoint_digest(producer/'best.pt'),
          'metrics_sha256':checkpoint_digest(producer/'metrics.json'),
          'source_validation_sha256':checkpoint_digest(producer/'source_validation.json')}
    (producer/'evaluation_complete.json').write_text(json.dumps(done))
    manifest={'checkpoint_sha256':done['checkpoint_sha256'],'score_sha256':scores,
              'target_features_sha256':checkpoint_digest(features/'complete.json')}
    (producer/'score_manifest.json').write_text(json.dumps(manifest))
    provenance={'producer_completion_sha256':{fold:checkpoint_digest(producer/'evaluation_complete.json')}}
    assert ev.verify_producer(fold,provenance)==(producer,features)
    (producer/'metrics.json').write_text('changed')
    with pytest.raises(ValueError,match='producer artifact changed'):ev.verify_producer(fold,provenance)


def test_primary_preserves_batch_one_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(primary,'verify_completed_run',lambda *a:({},'unused'))
    (tmp_path/'config.json').write_text(json.dumps({'fold':'f4','seed':1234,'eval_batch_size':132,
        'precision':'bfloat16_mixed','target_access':False,'provenance':{'source_sha256':{}}}))
    with pytest.raises(ValueError,match='F4 recipe differs'):primary.verify_training(tmp_path)


def test_primary_canonical_source_calibration_and_identity(tmp_path,monkeypatch):
    # Fit real synthetic primary weights and retain reverse IDs to exercise canonical ordering.
    bank={}
    for d in primary_train.SOURCES:
        n=8;labels=torch.arange(n)%2
        bank[d,'train']=FeatureSplit([(str(i),int(y)) for i,y in enumerate(labels)], [f'{d}_train_{n-i}' for i in range(n)],labels,
            torch.arange(n)[:,None].repeat(1,3),torch.randn(n,768),torch.randn(n,512))
        bank[d,'dev']=FeatureSplit([(str(i),int(y)) for i,y in enumerate(labels)], [f'{d}_dev_{n-i}' for i in range(n)],labels,
            torch.arange(n)[:,None],torch.randn(n,768),torch.randn(n,512))
    cache=tmp_path/'cache';cache.mkdir()
    for file in ('config.json','execution_accelerated.json','complete.json'):(cache/file).write_text('{}')
    monkeypatch.setattr(primary_train,'CACHE',cache)
    provenance={'source_sha256':{},'cache_config_sha256':checkpoint_digest(cache/'config.json'),
       'cache_completion_sha256':checkpoint_digest(cache/'complete.json'),
       'cache_execution_sha256':checkpoint_digest(cache/'execution_accelerated.json')}
    run=tmp_path/'f4/1234'
    done=primary_train.fit(bank,unit_vectors(torch.randn(8,512)),run,provenance,'cpu',logging.getLogger('fit'),epochs=1,patience=1)
    monkeypatch.setattr(primary,'canonical_score_labels',lambda d,s,r:dict(zip(bank[d,'dev'].ids,bank[d,'dev'].labels.tolist())))
    config,observed,model=primary.verify_training(run)
    assert observed==done and not model.training
    (run/'source_calibration.json').write_text('{}')
    with pytest.raises(ValueError,match='Completion-bound artifact changed'):primary.verify_training(run)
