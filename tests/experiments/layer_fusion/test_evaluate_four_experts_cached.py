"""CPU checks for cached target reuse without encoder execution or retraining."""
import json
import hashlib
import logging

import numpy as np
import pytest
import torch

from audio_deepfake_detection.protocol import checkpoint_digest
from scripts.experiments.layer_fusion import evaluate_four_experts_cached as e
from scripts.experiments.layer_fusion import evaluate_four_experts as four
from test_evaluate_four_experts import completed


@pytest.fixture
def target_cache(tmp_path,monkeypatch):
    producer=tmp_path/'producer';producer.mkdir()
    cache=tmp_path/'cache';cache.mkdir(exist_ok=True)
    (cache/'complete.json').write_text('{}')
    (cache/'execution_accelerated.json').write_text('{"pretrained":{}}')
    labels={'a':0,'b':1}
    monkeypatch.setattr(e,'canonical_score_labels',lambda *a,**kw:labels)
    (producer/'config.json').write_text(json.dumps({'provenance':{'cache_completion_sha256':checkpoint_digest(cache/'complete.json')}}))
    (producer/'training_complete.json').write_text(json.dumps({'config_sha256':checkpoint_digest(producer/'config.json')}))
    inference={'input':'centered common interval up to10s; native unpadded WavLM; CLAP repeatpad',
               'head_batch':132,'wavlm_batch':32,'clap_batch':64,'chunk':1024,'workers':8,'prefetch':2,
               'precision':'BF16 mixed; sensitive FP32','score':'scalar spoof logit'}
    (producer/'evaluation_config.json').write_text(json.dumps({'model_name':'LACF-MoE','fold':'f2','seed':1234,
        'target':'cfad/test_unseen','source_cache':str(cache),'frozen_encoder_revisions':{},'inference':inference}))
    features=producer/'target_features';chunk=features/'chunk_00000';chunk.mkdir(parents=True)
    rows=[{'index':i,'id':f'cfad/test_unseen/{key}','label':label,'frames':160000,'rate':16000,
           'intervals':[[0,160000]],'view_rows':[0]} for i,(key,label) in enumerate(labels.items())]
    (chunk/'records.json').write_text(json.dumps(rows))
    for name,shape in [('wavlm_layers',(2,13,768)),('clap_projected',(2,512))]:
        np.save(chunk/f'{name}.npy',np.ones(shape,np.float32),allow_pickle=False)
    def bind():
        (chunk/'complete.json').write_text(json.dumps({'records':2,'files_sha256':{
            name:checkpoint_digest(chunk/name) for name in ('records.json','wavlm_layers.npy','clap_projected.npy')}}))
        (features/'complete.json').write_text(json.dumps({'records':2,'chunks':[{'path':'chunk_00000',
            'completion_sha256':checkpoint_digest(chunk/'complete.json')}],
            'canonical_ids_labels_sha256':hashlib.sha256(json.dumps(list(labels.items())).encode()).hexdigest(),
            'evaluation_config_sha256':checkpoint_digest(producer/'evaluation_config.json')}))
        digest=checkpoint_digest(features/'complete.json')
        (producer/'evaluation_complete.json').write_text(json.dumps({'fold':'f2','seed':1234,'target_features_sha256':digest}))
        (producer/'score_manifest.json').write_text(json.dumps({'target_features_sha256':digest}))
    bind()
    return producer,cache,chunk,bind


def test_cache_preserves_ids_features_and_tail(target_cache,tmp_path):
    producer,cache,chunk,bind=target_cache
    results=list(e.target_chunks(producer,cache,'f2',tmp_path))
    assert len(results)==1 and results[0][0]==['a','b']
    assert results[0][1]['wavlm_layers'].shape==(2,13,768)


@pytest.mark.parametrize('mutation',['payload','crop','label','revision','precision','count','nan'])
def test_cache_corruption_or_policy_change_rejected(target_cache,tmp_path,mutation):
    producer,cache,chunk,bind=target_cache
    if mutation=='payload':
        (chunk/'clap_projected.npy').write_bytes(b'changed')
    elif mutation in ('crop','label'):
        rows=e.read(chunk/'records.json')
        if mutation=='crop':rows[0]['intervals']=[[1,159999]]
        else:rows[0]['label']=1
        (chunk/'records.json').write_text(json.dumps(rows));bind()
    elif mutation=='nan':
        values=np.ones((2,512),np.float32);values[0,0]=np.nan
        np.save(chunk/'clap_projected.npy',values,allow_pickle=False);bind()
    elif mutation=='count':
        rows=e.read(chunk/'records.json');rows.pop()
        (chunk/'records.json').write_text(json.dumps(rows));bind()
    else:
        config=e.read(producer/'evaluation_config.json')
        if mutation=='revision':config['frozen_encoder_revisions']={'wrong':'revision'}
        else:config['inference']['precision']='FP32'
        (producer/'evaluation_config.json').write_text(json.dumps(config));bind()
    with pytest.raises(ValueError):list(e.target_chunks(producer,cache,'f2',tmp_path))


def test_producer_failure_stops_wait_without_target_reads(tmp_path,monkeypatch):
    producer=tmp_path/'producer';producer.mkdir()
    launch=tmp_path/'launch';launch.mkdir();(launch/'exit.json').write_text('{}')
    monkeypatch.setattr(e,'PRODUCER',producer);monkeypatch.setattr(e,'PRODUCER_LAUNCH',launch)
    with pytest.raises(RuntimeError,match='exited'):e.wait_for_features(logging.getLogger('wait'))


def test_full_four_expert_cached_export_and_no_training_changes(completed,tmp_path,monkeypatch):
    torch.set_num_threads(2)
    run,data=completed('f2')
    before={p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    out=tmp_path/'evaluation/f2/1234';target={f'target{i}':i%2 for i in range(8)}
    def canonical(domain,split,root,**kw):
        if domain=='cfad':
            assert (out/'source_validation.json').is_file();return target
        ds=data[domain,'dev'];return dict(zip(ds.ids,ds.labels.tolist()))
    monkeypatch.setattr(four,'canonical_score_labels',canonical);monkeypatch.setattr(e,'canonical_score_labels',canonical)
    import scripts.evaluate_scores as shared
    monkeypatch.setattr(shared,'canonical_score_labels',canonical)
    monkeypatch.setattr(four,'load_split',lambda cache,d,s,marker,root,logger=None:data[d,s])
    launch=tmp_path/'launch';launch.mkdir();(launch/'provenance.json').write_text('{"code_sha256":{}}')
    monkeypatch.setattr(e,'LAUNCH',launch)
    producer=tmp_path/'producer';(producer/'target_features').mkdir(parents=True)
    (producer/'target_features/complete.json').write_text('{}')
    ds=next(iter(data.values()));arrays={'wavlm_layers':ds.wavlm.numpy(),'clap_projected':ds.clap.numpy()}
    def chunks(*args):
        canonical('cfad','test_unseen',tmp_path)
        yield list(target),arrays
    monkeypatch.setattr(e,'target_chunks',chunks)
    result=e.evaluate(run,out,producer,tmp_path,'cpu',logging.getLogger('evaluate'),bootstrap_resamples=5)
    assert result['model_family']=='lacf_layer_moe4'
    assert e.read(out/'evaluation_config.json')['encoder_execution'] is False
    assert e.read(out/'evaluation_complete.json')['checkpoint_sha256']==checkpoint_digest(run/'best.pt')
    assert before=={p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):e.evaluate(run,out,producer,tmp_path,'cpu',logging.getLogger('retry'))
