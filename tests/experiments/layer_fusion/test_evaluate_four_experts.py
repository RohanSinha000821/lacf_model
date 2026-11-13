"""CPU-only end-to-end source gates and canonical four-fold score export."""
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from audio_deepfake_detection.protocol import checkpoint_digest
from scripts.experiments.layer_fusion import evaluate_four_experts as evaluator
from scripts.experiments.layer_fusion import train_four_experts as trainer
from test_four_experts import bank, prototypes


@pytest.fixture
def completed(tmp_path):
    def make(fold):
        sources = bank(fold)
        cache = tmp_path / 'cache'
        cache.mkdir(exist_ok=True)
        (cache/'config.json').write_text('{}')
        (cache/'execution_accelerated.json').write_text('{"pretrained":{}}')
        (cache/'complete.json').write_text(json.dumps({'splits':{f'{d}/dev':{} for d in trainer.sources_for(fold)}}))
        provenance = {'code_sha256': {evaluator.TRAINER:checkpoint_digest(evaluator.ROOT/evaluator.TRAINER)},
                      'cache':str(cache)}
        for name,key in [('config.json','cache_config_sha256'),('complete.json','cache_completion_sha256'),
                         ('execution_accelerated.json','cache_execution_sha256')]:
            provenance[key]=checkpoint_digest(cache/name)
        run=tmp_path/'training'/fold/'1234'
        trainer.fit(sources,prototypes(),run,provenance,'cpu',logging.getLogger('evaluation_fixture'),
                    fold=fold,max_epochs=1)
        return run,sources
    return make


@pytest.mark.parametrize('fold',tuple(trainer.FOLDS))
def test_canonical_all_fold_evaluation_preserves_training(completed,tmp_path,monkeypatch,fold):
    torch.set_num_threads(2)
    run,sources=completed(fold)
    hashes={p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    target_domain=trainer.FOLDS[fold]['target']
    target_split=trainer.FOLDS[fold]['target_split']
    target_ids={f'target_{i}':i%2 for i in range(8)}
    calls=[]
    def canonical(domain,split,root,**kw):
        calls.append((domain,split))
        if domain==target_domain:
            assert split==target_split
            assert (out/'source_validation.json').exists()
            if fold!='f1':assert (out/'source_encoder_check.json').exists()
            return target_ids
        assert split=='dev'
        ds=sources[domain,'dev']
        return dict(zip(ds.ids,ds.labels.tolist()))
    monkeypatch.setattr(evaluator,'canonical_score_labels',canonical)
    import scripts.evaluate_scores as shared
    monkeypatch.setattr(shared,'canonical_score_labels',canonical)
    monkeypatch.setattr(evaluator,'load_split',lambda cache,d,s,marker,data_root,logger=None:sources[d,s])
    ds=next(iter(sources.values()))
    arrays={'wavlm_layers':ds.wavlm.numpy(),'clap_projected':ds.clap.numpy()}
    features=tmp_path/'existing_features';features.mkdir();(features/'complete.json').write_text('{}')
    monkeypatch.setattr(evaluator,'TARGET_FEATURES',features)
    monkeypatch.setattr(evaluator,'target_chunks',lambda *args:iter([(list(target_ids),arrays)]))
    monkeypatch.setattr(evaluator,'load_encoders',lambda *args:(None,None))
    def encoder_check(encoders,collator,cache,domain,root,output,logger):
        assert domain in trainer.FOLDS[fold]['sources'] and domain!=target_domain
        assert not any(d==target_domain for d,s in calls)
        evaluator.save_json(output/'source_encoder_check.json',{'passed':True})
    monkeypatch.setattr(evaluator,'source_encoder_check',encoder_check)
    def extract(encoders,collator,fold,root,output,guards,logger):
        canonical(target_domain,target_split,root)
        f=output/'target_features';f.mkdir();(f/'complete.json').write_text('{}')
        yield list(target_ids),arrays
    monkeypatch.setattr(evaluator,'extract_chunks',extract)
    out=tmp_path/'eval'/fold/'1234'
    report=evaluator.evaluate(run,out,fold,tmp_path,'cpu',logging.getLogger('eval'),family=tmp_path/'training',bootstrap_resamples=5)
    assert report['fold']==fold and report['model_family']=='lacf_layer_moe4'
    assert (out/'evaluation_complete.json').is_file()
    assert json.loads((out/'source_validation.json').read_text())['passed']
    assert json.loads((out/'training_complete.json').read_text())['best_checkpoint_sha256']==checkpoint_digest(run/'best.pt')
    assert hashes=={p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):evaluator.evaluate(run,out,fold,tmp_path,'cpu',logging.getLogger('eval'))
    (run/'source_calibration.json').write_text('{}')
    with pytest.raises(ValueError):evaluator.verify_training(run,fold,tmp_path,tmp_path/'training')


def test_source_reload_failure_never_reads_target_or_creates_output(completed,tmp_path,monkeypatch):
    run,sources=completed('f1')
    monkeypatch.setattr(evaluator,'canonical_score_labels',lambda d,s,r:dict(zip(sources[d,'dev'].ids,sources[d,'dev'].labels.tolist())))
    def mismatch(*args,**kw):raise ValueError('source differs')
    monkeypatch.setattr(evaluator,'source_reload',mismatch)
    def forbidden(*args,**kw):raise AssertionError('Target accessed before source verification')
    monkeypatch.setattr(evaluator,'target_chunks',forbidden)
    output=tmp_path/'eval/f1/1234'
    with pytest.raises(ValueError,match='source differs'):
        evaluator.evaluate(run,output,'f1',tmp_path,'cpu',logging.getLogger('fail'))
    assert not output.exists()


def test_wait_failure_does_not_advance(tmp_path):
    launch=tmp_path/'launch_f2_f3_f4';launch.mkdir()
    (launch/'exit.json').write_text('{"verified_complete":false}')
    with pytest.raises(RuntimeError,match='exited without completion'):
        evaluator.wait_for_completion(tmp_path/'f2/1234','f2',tmp_path,logging.getLogger('wait'))


def test_f1_trainer_reconciliation_is_narrow(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluator,'ROOT',tmp_path)
    trainer_path=tmp_path/evaluator.TRAINER;trainer_path.parent.mkdir(parents=True);trainer_path.write_text('new')
    old=tmp_path/'old';old.write_text('old');old_hash=checkpoint_digest(old);new_hash=checkpoint_digest(trainer_path)
    family=tmp_path/'family';snapshot=family/'launch/f1_source_snapshot'/evaluator.TRAINER
    snapshot.parent.mkdir(parents=True);snapshot.write_text('old')
    extension=family/'launch_f2_f3_f4';extension.mkdir()
    (extension/'provenance.json').write_text(json.dumps({'trainer_extension':{'previous_sha256':old_hash,'current_sha256':new_hash}}))
    seen=[]
    monkeypatch.setattr(evaluator,'check_guards',lambda p:seen.append(p['code_sha256']))
    config={'fold':'f1','provenance':{'code_sha256':{evaluator.TRAINER:old_hash}}}
    guards,record=evaluator.training_guards(config,family)
    assert record['archived_sha256']==old_hash and guards['code_sha256'][evaluator.TRAINER]==new_hash
    assert config['provenance']['code_sha256'][evaluator.TRAINER]==old_hash
    snapshot.write_text('tampered')
    with pytest.raises(ValueError,match='archived trainer'):evaluator.training_guards(config,family)
    config['fold']='f2'
    with pytest.raises(ValueError,match='Unreviewed'):evaluator.training_guards(config,family)


@pytest.mark.parametrize('fold',('f1','f2'))
def test_reviewed_calibration_fix_preserves_historical_hash_chain(tmp_path,monkeypatch,fold):
    monkeypatch.setattr(evaluator,'ROOT',tmp_path)
    current=tmp_path/evaluator.TRAINER
    current.parent.mkdir(parents=True);current.write_text('fixed verifier')
    family=tmp_path/'family';folder=family/'launch/calibration_verification_fix'
    snapshot=folder/'snapshot'/evaluator.TRAINER
    snapshot.parent.mkdir(parents=True);snapshot.write_text('fold-aware trainer')
    prior=checkpoint_digest(snapshot);now=checkpoint_digest(current)
    recorded=prior
    if fold=='f1':
        first=family/'launch/f1_source_snapshot'/evaluator.TRAINER
        first.parent.mkdir(parents=True);first.write_text('original F1 trainer')
        recorded=checkpoint_digest(first)
        extension=family/'launch_f2_f3_f4';extension.mkdir()
        (extension/'provenance.json').write_text(json.dumps({'trainer_extension':{'previous_sha256':recorded,'current_sha256':prior}}))
    review={'scope':'source CSV ID ordering only; training and scorer unchanged',
            'previous_sha256':prior,'current_sha256':now}
    (folder/'record.json').write_text(json.dumps(review))
    config={'fold':fold,'provenance':{'code_sha256':{evaluator.TRAINER:recorded,'model.py':'unaltered model hash'}}}
    seen=[]
    monkeypatch.setattr(evaluator,'check_guards',lambda p:seen.append(p['code_sha256']))
    guards,record=evaluator.training_guards(config,family)
    assert guards['code_sha256'][evaluator.TRAINER]==now
    assert record['archived_sha256']==recorded and 'source CSV ID ordering' in record['reason']
    assert seen[-1]['model.py']=='unaltered model hash'
    assert config['provenance']['code_sha256'][evaluator.TRAINER]==recorded
    for field in ('scope','previous_sha256','current_sha256'):
        (folder/'record.json').write_text(json.dumps({**review,field:'unreviewed'}))
        with pytest.raises(ValueError,match='reconciliation differs'):
            evaluator.training_guards(config,family)
    (folder/'record.json').write_text(json.dumps(review))
    snapshot.write_text('tampered historical source')
    with pytest.raises(ValueError,match='reconciliation differs'):
        evaluator.training_guards(config,family)


@pytest.mark.parametrize('nonfinite',[False,True])
def test_new_target_extraction_keeps_canonical_tail_and_aborts_on_nan(tmp_path,monkeypatch,nonfinite):
    fold='f4';spec=trainer.FOLDS[fold]
    expected={'a':0,'b':1}
    monkeypatch.setattr(evaluator,'read_dataset',lambda *args:[('a.wav',0),('b.wav',1)])
    monkeypatch.setattr(evaluator,'canonical_score_labels',lambda *args,**kw:expected)
    prepared=[{'index':i,'id':f"{spec['target']}/{spec['target_split']}/{key}",'label':label,
               'frames':200000,'rate':16000,'intervals':[(20000,160000)],'view_rows':[0],'views':[None]}
              for i,(key,label) in enumerate(expected.items())]
    monkeypatch.setattr(evaluator,'loader',lambda *args,**kw:iter(prepared))
    arrays={'wavlm_layers':np.ones((2,13,768),np.float32),'clap_projected':np.ones((2,512),np.float32)}
    if nonfinite:arrays['clap_projected'][0,0]=np.nan
    monkeypatch.setattr(evaluator,'encode_chunk',lambda *args:arrays)
    (tmp_path/'evaluation_config.json').write_text('{}')
    iterator=evaluator.extract_chunks(None,None,fold,tmp_path,tmp_path,{'code_sha256':{}},logging.getLogger('extract'))
    if nonfinite:
        with pytest.raises(ValueError,match='Invalid target feature'):list(iterator)
        assert not (tmp_path/'target_features/complete.json').exists()
    else:
        chunks=list(iterator)
        assert len(chunks)==1 and chunks[0][0]==['a','b']
        marker=json.loads((tmp_path/'target_features/complete.json').read_text())
        assert marker['records']==2 and len(marker['chunks'])==1
        chunk=tmp_path/'target_features/chunk_00000'
        proof=json.loads((chunk/'complete.json').read_text())
        assert all(checkpoint_digest(chunk/name)==digest for name,digest in proof['files_sha256'].items())
