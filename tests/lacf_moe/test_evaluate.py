"""Synthetic CPU evaluation gates for the main three-expert model."""
import json
import logging

import numpy as np
import pytest
import torch

from audio_deepfake_detection.protocol import checkpoint_digest
from scripts.lacf_moe import evaluate as e
from scripts.lacf_moe import train as t
from scripts.experiments.layer_fusion import evaluate_four_experts as helpers
from test_lacf_moe import bank, prototypes


@pytest.fixture
def completed(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    def make(fold):
        data = bank(fold)
        cache = tmp_path / 'cache'
        cache.mkdir(exist_ok=True)
        (cache/'config.json').write_text('{}')
        (cache/'execution_accelerated.json').write_text('{"pretrained":{}}')
        (cache/'complete.json').write_text(json.dumps({'splits':{f'{d}/dev':{} for d in t.FOLDS[fold]['sources']}}))
        provenance = {'code_sha256':{}, 'cache':str(cache)}
        for name,key in [('config.json','cache_config_sha256'),('complete.json','cache_completion_sha256'),
                         ('execution_accelerated.json','cache_execution_sha256')]:
            provenance[key] = checkpoint_digest(cache/name)
        run = tmp_path/'training'/fold/'1234'
        t.fit(data,prototypes(),run,fold,provenance,'cpu',logging.getLogger('fixture'),max_epochs=1)
        output = tmp_path/'evaluation'
        (output/'launch').mkdir(parents=True,exist_ok=True)
        (output/'launch/provenance.json').write_text('{"code_sha256":{}}')
        monkeypatch.setattr(e,'OUTPUT',output)
        return run,data
    return make


@pytest.mark.parametrize('fold',tuple(t.FOLDS))
def test_full_source_reload_export_and_shared_metrics(completed,tmp_path,monkeypatch,fold):
    run,data = completed(fold)
    before = {p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    spec = t.FOLDS[fold]
    target = {f'target_{i}':i%2 for i in range(8)}
    out = e.OUTPUT/fold/'1234'
    def canonical(domain,split,root,**kw):
        if domain == spec['target']:
            assert split == spec['target_split']
            assert (out/'source_validation.json').is_file()
            assert (out/'source_encoder_check.json').is_file()
            return target
        assert split == 'dev'
        ds = data[domain,'dev']
        return dict(zip(ds.ids,ds.labels.tolist()))
    monkeypatch.setattr(t,'canonical_score_labels',canonical)
    monkeypatch.setattr(e,'canonical_score_labels',canonical)
    import scripts.evaluate_scores as shared
    monkeypatch.setattr(shared,'canonical_score_labels',canonical)
    monkeypatch.setattr(helpers,'load_split',lambda cache,d,s,marker,root,logger=None:data[d,s])
    monkeypatch.setattr(e,'load_encoders',lambda cache:(None,None))
    def encoder_check(encoders,collator,cache,domain,root,output,logger):
        assert domain in spec['sources'] and domain != spec['target']
        e.save_json(output/'source_encoder_check.json',{'passed':True})
    monkeypatch.setattr(e,'source_encoder_check',encoder_check)
    ds = next(iter(data.values()))
    arrays = {'wavlm_layers':ds.wavlm.numpy(),'clap_projected':ds.clap.numpy()}
    def extract(*args):
        canonical(spec['target'],spec['target_split'],tmp_path)
        features = out/'target_features';features.mkdir()
        (features/'complete.json').write_text('{}')
        yield list(target),arrays
    monkeypatch.setattr(e,'extract_chunks',extract)
    result = e.evaluate(run,out,fold,tmp_path,'cpu',logging.getLogger('evaluate'),bootstrap_resamples=5)
    assert result['model_family']=='lacf_moe'
    assert e.read(out/'evaluation_complete.json')['no_target_tuning']
    assert e.load_detector(run,'cpu').configuration()['trainable_parameters']==967338
    assert before == {p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):e.evaluate(run,out,fold,tmp_path,'cpu',logging.getLogger('overwrite'))
    (out/'target_scores.csv').write_text('changed')
    with pytest.raises(ValueError,match='score changed'):e.verify_evaluation(out,fold,run,tmp_path)


def test_source_failure_prevents_target_access(completed,tmp_path,monkeypatch):
    run,data = completed('f2')
    monkeypatch.setattr(t,'canonical_score_labels',lambda d,s,r:dict(zip(data[d,'dev'].ids,data[d,'dev'].labels.tolist())))
    def fail(*args,**kw):raise ValueError('source reload differs')
    monkeypatch.setattr(e,'source_reload',fail)
    monkeypatch.setattr(e,'load_encoders',lambda *a:pytest.fail('Target preparation before source verification'))
    out = e.OUTPUT/'f2/1234'
    with pytest.raises(ValueError,match='source reload differs'):
        e.evaluate(run,out,'f2',tmp_path,'cpu',logging.getLogger('fail'))
    assert not out.exists()


@pytest.mark.parametrize('bad',[False,True])
def test_target_tail_saved_and_nonfinite_fails(tmp_path,monkeypatch,bad):
    fold='f4';spec=t.FOLDS[fold];expected={'a':0,'b':1}
    monkeypatch.setattr(e,'read_dataset',lambda *a:[('a.wav',0),('b.wav',1)])
    monkeypatch.setattr(e,'canonical_score_labels',lambda *a,**kw:expected)
    rows=[{'index':i,'id':f"{spec['target']}/{spec['target_split']}/{name}",'label':label,
           'frames':160000,'rate':16000,'intervals':[(0,160000)],'view_rows':[0],'views':[None]}
          for i,(name,label) in enumerate(expected.items())]
    monkeypatch.setattr(e,'loader',lambda *a,**kw:iter(rows))
    arrays={'wavlm_layers':np.ones((2,13,768),np.float32),'clap_projected':np.ones((2,512),np.float32)}
    if bad:arrays['clap_projected'][0,0]=np.nan
    monkeypatch.setattr(e,'encode_chunk',lambda *a:arrays)
    (tmp_path/'evaluation_config.json').write_text('{}')
    chunks=e.extract_chunks(None,None,fold,tmp_path,tmp_path,{'code_sha256':{}},logging.getLogger('extract'))
    if bad:
        with pytest.raises(ValueError,match='Invalid target feature'):list(chunks)
        assert not (tmp_path/'target_features/complete.json').exists()
    else:
        assert list(chunks)[0][0]==['a','b']
        assert e.read(tmp_path/'target_features/complete.json')['records']==2
        assert 'three-expert' in e.read(tmp_path/'progress.json')['phase']


def test_queue_stops_before_next_fold_on_failure(tmp_path,monkeypatch):
    output=tmp_path/'evaluation';(output/'launch').mkdir(parents=True)
    (output/'launch/provenance.json').write_text('{"code_sha256":{}}')
    monkeypatch.setattr(e,'OUTPUT',output)
    monkeypatch.setattr(e,'verify_training',lambda *a:({},{}))
    original=tmp_path/'old/f1/1234';original.mkdir(parents=True)
    (original/'evaluation_complete.json').write_text('{}')
    monkeypatch.setattr(e,'ORIGINAL_F1',original)
    monkeypatch.setattr(e,'verify_evaluation',lambda *a:{'checkpoint_sha256':'same'})
    calls=[]
    def fail(training,out,fold,*args):
        calls.append(fold);raise ValueError('failed fold')
    monkeypatch.setattr(e,'evaluate',fail)
    with pytest.raises(ValueError,match='failed fold'):e.run_queue(tmp_path,'cpu',logging.getLogger('queue'))
    assert calls==['f2'] and (output/'f1/1234').resolve()==original
    assert not (output/'launch/complete.json').exists()
