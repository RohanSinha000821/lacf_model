"""Missing-fold continuation must preserve previous fits and all active sources."""
import json
import logging
from types import SimpleNamespace

import pytest
import torch

from audio_deepfake_detection.protocol import checkpoint_digest
from scripts.experiments.layer_fusion import complete_four_experts as e
from test_evaluate_four_experts import completed
from test_evaluate_four_experts_cached import target_cache


@pytest.mark.parametrize('fold',('f3','f4'))
def test_cached_remaining_fold_full_source_reload_and_shared_metrics(completed,tmp_path,monkeypatch,fold):
    run,data=completed(fold)
    before={p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    out=tmp_path/'evaluation'/fold/'1234';spec=e.FOLDS[fold]
    target={f'target{i}':i%2 for i in range(8)}
    def canonical(domain,split,root,**kw):
        if domain==spec['target']:
            assert split==spec['target_split'] and (out/'source_validation.json').is_file()
            return target
        ds=data[domain,'dev'];return dict(zip(ds.ids,ds.labels.tolist()))
    monkeypatch.setattr(e.four,'canonical_score_labels',canonical);monkeypatch.setattr(e,'canonical_score_labels',canonical)
    import scripts.evaluate_scores as shared
    monkeypatch.setattr(shared,'canonical_score_labels',canonical)
    monkeypatch.setattr(e.four,'load_split',lambda cache,d,s,marker,root,logger=None:data[d,s])
    launch=tmp_path/'launch';launch.mkdir();(launch/'provenance.json').write_text('{"code_sha256":{}}')
    producer=tmp_path/'producer';(producer/'target_features').mkdir(parents=True)
    (producer/'target_features/complete.json').write_text('{}')
    ds=next(iter(data.values()));arrays={'wavlm_layers':ds.wavlm.numpy(),'clap_projected':ds.clap.numpy()}
    def chunks(*args):
        canonical(spec['target'],spec['target_split'],tmp_path)
        yield list(target),arrays
    monkeypatch.setattr(e,'target_chunks',chunks)
    result=e.evaluate(run,out,producer,tmp_path,'cpu',logging.getLogger('remaining'),fold=fold,launch=launch,bootstrap_resamples=5)
    assert result['fold']==fold and result['model_family']=='lacf_layer_moe4'
    assert e.read(out/'evaluation_config.json')['encoder_execution'] is False
    assert before=={p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        e.evaluate(run,out,producer,tmp_path,'cpu',logging.getLogger('overwrite'),fold=fold,launch=launch)


@pytest.mark.parametrize('fold',('f3','f4'))
def test_existing_cache_reader_supports_remaining_canonical_folds(target_cache,tmp_path,fold):
    producer,cache,chunk,bind=target_cache;spec=e.FOLDS[fold]
    config=e.read(producer/'evaluation_config.json');config['fold']=fold
    config['target']=spec['target']+'/'+spec['target_split']
    (producer/'evaluation_config.json').write_text(json.dumps(config))
    rows=e.read(chunk/'records.json')
    for row,name in zip(rows,('a','b')):row['id']=f"{spec['target']}/{spec['target_split']}/{name}"
    (chunk/'records.json').write_text(json.dumps(rows));bind()
    done=e.read(producer/'evaluation_complete.json');done['fold']=fold
    (producer/'evaluation_complete.json').write_text(json.dumps(done))
    assert list(e.target_chunks(producer,cache,fold,tmp_path))[0][0]==['a','b']


@pytest.mark.parametrize('failed_training',(False,True))
def test_continuation_trains_only_missing_folds_and_stops_on_failure(tmp_path,monkeypatch,failed_training):
    train=tmp_path/'train';evaluate=tmp_path/'eval';launch=evaluate/'launch_f3_f4';launch.mkdir(parents=True)
    train_launch=train/'launch_f3_f4';train_launch.mkdir(parents=True)
    for folder in (launch,train_launch):(folder/'provenance.json').write_text('{"code_sha256":{}}')
    for fold in ('f1','f2'):
        run=evaluate/fold/'1234';run.mkdir(parents=True);(run/'evaluation_complete.json').write_text('{}')
    existing=tmp_path/'existing_f2';existing.mkdir();(existing/'exit.json').write_text('{"verified_complete":true}')
    monkeypatch.setattr(e,'LAUNCH',launch);monkeypatch.setattr(e,'TRAIN_LAUNCH',train_launch)
    monkeypatch.setattr(e.four,'FAMILY',train);monkeypatch.setattr(e.four,'OUTPUT',evaluate)
    monkeypatch.setattr(e,'F2_LAUNCH',existing)
    calls=[]
    monkeypatch.setattr(e,'verify_preserved',lambda:calls.append('preserved'))
    monkeypatch.setattr(e.four,'verify_training',lambda run,fold,*a:calls.append(('verify',fold)))
    monkeypatch.setattr(e,'wait_for_record',lambda path,*a,**kw:calls.append(('wait',str(path))))
    monkeypatch.setattr(e.subprocess,'check_output',lambda *a,**kw:'8192\n')
    def train_command(command,**kw):
        assert command[-3:]==['--folds','f3','f4']
        assert command[3]=='scripts.experiments.layer_fusion.train_four_experts'
        calls.append('train')
        if not failed_training:
            for fold in ('f3','f4'):
                run=train/fold/'1234';run.mkdir(parents=True);(run/'training_complete.json').write_text('{}')
        return SimpleNamespace(returncode=int(failed_training))
    monkeypatch.setattr(e.subprocess,'run',train_command)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda **kw:True)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:SimpleNamespace(total_memory=35*1024**3))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    def score(training,output,*a,fold,launch,**kw):
        calls.append(('evaluate',fold));output.mkdir(parents=True)
        (output/'evaluation_complete.json').write_text('{}')
    monkeypatch.setattr(e,'evaluate',score)
    if failed_training:
        with pytest.raises(RuntimeError,match='Fresh training failed'):e.main()
        assert not any(isinstance(x,tuple) and x[0]=='evaluate' for x in calls)
        assert not (launch/'complete.json').exists()
    else:
        e.main()
        assert [x for x in calls if isinstance(x,tuple) and x[0]=='evaluate']==[('evaluate','f3'),('evaluate','f4')]
        assert e.read(launch/'complete.json')['preserved_f1_f2']
        with pytest.raises(FileExistsError):e.main()


def test_dependency_failure_prevents_training(tmp_path,monkeypatch):
    launch=tmp_path/'launch';launch.mkdir();training=tmp_path/'training';training.mkdir()
    (training/'provenance.json').write_text('{"code_sha256":{}}')
    f2=tmp_path/'f2';f2.mkdir();(f2/'exit.json').write_text('{"verified_complete":false}')
    monkeypatch.setattr(e,'LAUNCH',launch);monkeypatch.setattr(e,'TRAIN_LAUNCH',training)
    monkeypatch.setattr(e.four,'FAMILY',tmp_path/'fits');monkeypatch.setattr(e,'F2_LAUNCH',f2)
    monkeypatch.setattr(e,'verify_preserved',lambda:None)
    monkeypatch.setattr(e.subprocess,'run',lambda *a,**kw:pytest.fail('Training before prior evaluation verified'))
    with pytest.raises(RuntimeError,match='Existing F2 evaluation failed'):e.main()


def test_gpu_gate_releases_with_spare_memory_without_waiting_for_encoder_exit(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'LAUNCH',tmp_path)
    memory=iter(('1024\n','8807\n'))
    monkeypatch.setattr(e.subprocess,'check_output',lambda *a,**kw:next(memory))
    sleeps=[];monkeypatch.setattr(e.time,'sleep',lambda seconds:sleeps.append(seconds))
    free=e.wait_for_gpu_headroom(logging.getLogger('headroom'))
    assert free==8807 and sleeps==[30]
    assert e.read(tmp_path/'gpu_headroom_release.json')['minimum_free_mib']==4096
    assert e.read(tmp_path/'progress.json')['phase']=='waiting for GPU headroom'


def test_gpu_gate_times_out_without_starting_training(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'LAUNCH',tmp_path)
    monkeypatch.setattr(e.subprocess,'check_output',lambda *a,**kw:'1024\n')
    ticks=iter((0,2));monkeypatch.setattr(e.time,'monotonic',lambda:next(ticks))
    with pytest.raises(TimeoutError,match='headroom'):
        e.wait_for_gpu_headroom(logging.getLogger('headroom'),timeout=1)
    assert not (tmp_path/'gpu_headroom_release.json').exists()
