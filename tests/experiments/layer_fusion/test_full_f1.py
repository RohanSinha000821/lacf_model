"""Synthetic full-F1 C4 fitting; no encoders, audio or targets."""

import hashlib
import json
import logging

import numpy as np
import pytest
import torch

from audio_deepfake_detection.lacf.probes import FeatureSplit
from scripts.experiments.layer_fusion import train_full_f1 as runner
from scripts.experiments.layer_fusion import evaluate_lacf_layer_moe as evaluator
from audio_deepfake_detection.protocol import checkpoint_digest


def bank():
    torch.manual_seed(9)
    result = {}
    for domain in runner.SOURCES:
        for part in ('train', 'dev'):
            labels = torch.tensor([0, 0, 1, 1, 1, 1])
            count = len(labels); features = count*3 if part == 'train' else count
            views = torch.arange(features).reshape(count, 3) if part == 'train' else torch.arange(count)[:, None].expand(-1, 3)
            result[domain, part] = FeatureSplit([(f'{domain}/{part}/{i}', int(y)) for i,y in enumerate(labels)],
                [f'{domain}/{part}/{i}' for i in range(count)], labels, views,
                torch.randn(features, 13, 768), torch.randn(features, 512))
    return result


def prototypes():
    return torch.nn.functional.normalize(torch.randn(8, 512), dim=-1)


def test_real_c4_full_f1_reloads_calibrates_and_records_recipe(tmp_path):
    sources = bank(); output = tmp_path/'run'
    result = runner.fit(sources, prototypes(), output, {'code_sha256': {}}, 'cpu', logging.getLogger('full_c4'), max_epochs=2)
    assert result['selected_checkpoint_reloaded_and_source_verified'] and not result['target_access']
    assert set(result['source_scores_sha256']) == {f'source_dev_{d}.csv' for d in runner.SOURCES}
    config = json.loads((output/'config.json').read_text())
    assert config['seed_plan'] == [1234] and config['model']['condition'] == 'C4'
    assert config['model']['trainable_parameters'] == 967338
    assert config['model']['objective'] == 'detection BCE only'
    assert config['train_records'] == 18 and config['sources'] == ['asv2019','asv5','cfad']
    assert config['lr'] == .0003 and config['weight_decay'] == .0001 and config['batch_size'] == 132
    for filename,h in result['source_scores_sha256'].items():
        assert hashlib.sha256((output/filename).read_bytes()).hexdigest() == h
    assert set(json.loads((output/'routing.json').read_text())['domains']) == set(runner.SOURCES)
    assert not list(output.glob('*target*'))
    with pytest.raises(FileExistsError):
        runner.fit(sources, prototypes(), output, {'code_sha256': {}}, 'cpu', logging.getLogger('full_c4'))


def test_ties_keep_first_best_and_patience_stops(tmp_path, monkeypatch):
    sources = bank()
    def scores(*args, **kwargs):
        values = {d:sources[d,'dev'].labels.float()*2-1 for d in runner.SOURCES}
        metrics = {'by_domain':{d:{'eer':0.,'auroc':1.} for d in runner.SOURCES},'macro_eer':0.,'macro_auroc':1.}
        return values,metrics
    monkeypatch.setattr(runner,'score_sources',scores)
    result = runner.fit(sources,prototypes(),tmp_path/'run',{'code_sha256':{}},'cpu',logging.getLogger('ties'),max_epochs=8,patience=2)
    assert result['best_epoch']==1 and result['final_epoch']==3
    last = torch.load(tmp_path/'run/last.pt',weights_only=True)
    assert last['epoch']==3 and last['patience']==2 and 'optimizer_state_dict' in last


def test_wrong_source_bank_and_guard_change_stop_before_run(tmp_path):
    sources = bank(); sources['speechfake','dev'] = next(iter(sources.values()))
    with pytest.raises(ValueError,match='Only complete F1'):
        runner.fit(sources,prototypes(),tmp_path/'run',{'code_sha256':{}},'cpu',logging.getLogger('sources'))
    sources.pop(('speechfake','dev'))
    with pytest.raises(ValueError,match='Execution source changed'):
        runner.fit(sources,prototypes(),tmp_path/'run',{'code_sha256':{'README.md':'wrong'}},'cpu',logging.getLogger('guards'))
    assert not (tmp_path/'run').exists()


def test_failed_selected_reload_never_marks_complete(tmp_path, monkeypatch):
    sources = bank(); actual = runner.score_sources; calls = 0
    def changed(model, bank, device, logger=None):
        nonlocal calls
        calls += 1
        scores,metrics = actual(model,bank,device,logger)
        if calls == 2:
            scores[runner.SOURCES[0]] = scores[runner.SOURCES[0]].clone()
            scores[runner.SOURCES[0]][0] += 1
        return scores,metrics
    monkeypatch.setattr(runner,'score_sources',changed)
    with pytest.raises(ValueError,match='source reload differs'):
        runner.fit(sources,prototypes(),tmp_path/'run',{'code_sha256':{}},'cpu',logging.getLogger('reload'),max_epochs=1)
    assert not (tmp_path/'run/training_complete.json').exists()


@pytest.fixture
def completed_moe(tmp_path, monkeypatch):
    sources = bank()
    cache = tmp_path/'source_features'; cache.mkdir()
    for name, value in {'config.json': {}, 'execution_accelerated.json': {'pretrained': {'wavlm_revision': 'w', 'clap_revision': 'c'}},
                        'complete.json': {'splits': {f'{d}/dev': {} for d in runner.SOURCES}}}.items():
        (cache/name).write_text(json.dumps(value))
    provenance = {'code_sha256': {}, 'cache': str(cache),
        **{key: checkpoint_digest(cache/name) for key, name in (
            ('cache_config_sha256', 'config.json'), ('cache_completion_sha256', 'complete.json'),
            ('cache_execution_sha256', 'execution_accelerated.json'))}}
    run = tmp_path/'training/lacf_layer_moe/f1/1234'
    runner.fit(sources, prototypes(), run, provenance, 'cpu', logging.getLogger('moe_eval'), max_epochs=1)
    expected = {d: dict(zip(sources[d,'dev'].ids, sources[d,'dev'].labels.tolist())) for d in runner.SOURCES}
    target = {f'record_{i}': int(i >= 2) for i in range(6)}
    monkeypatch.setattr(evaluator, 'canonical_score_labels', lambda domain, part, root: target if domain=='speechfake' else expected[domain])
    monkeypatch.setattr('scripts.evaluate_scores.canonical_score_labels', lambda domain, part, root: target if domain=='speechfake' else expected[domain])
    monkeypatch.setattr(evaluator, 'load_split', lambda root,d,part,*args,**kwargs: sources[d,part])
    return run, cache, sources, target


@pytest.fixture
def target_cache(completed_moe, tmp_path):
    run, source_cache, sources, target = completed_moe
    directory = tmp_path/'prior_evaluation/target_features'; directory.mkdir(parents=True)
    config = {'provenance': {'cache_root': str(source_cache), 'cache_completion_sha256': checkpoint_digest(source_cache/'complete.json')},
              'frozen_encoder_revisions': json.loads((source_cache/'execution_accelerated.json').read_text())['pretrained'],
              'batching': {'wavlm':32,'clap':64,'head':132,'chunk':1024,'workers':8,'prefetch':2}}
    (directory.parent/'evaluation_config.json').write_text(json.dumps(config))
    rows = []
    for i,(name,label) in enumerate(target.items()):
        frames = 200000 if i%2 else 4000
        intervals,aliases = evaluator.crop_intervals(frames,16000,name,training=False)
        rows.append({'index':i,'id':f'speechfake/test/{name}','label':label,'frames':frames,'rate':16000,
                     'intervals':[list(x) for x in intervals],'view_rows':aliases})
    for index in range(2):
        chunk = directory/f'chunk_{index:05d}';chunk.mkdir()
        (chunk/'records.json').write_text(json.dumps(rows[index*3:(index+1)*3]))
        values = sources['asv2019','dev']
        np.save(chunk/'wavlm_layers.npy',values.wavlm[index*3:(index+1)*3].numpy())
        np.save(chunk/'clap_projected.npy',values.clap[index*3:(index+1)*3].numpy())
    def seal():
        chunks = []
        for index in range(2):
            chunk = directory/f'chunk_{index:05d}'
            (chunk/'complete.json').write_text(json.dumps({'records':3,'files_sha256':{
                name:checkpoint_digest(chunk/name) for name in ('records.json','wavlm_layers.npy','clap_projected.npy')}}))
            chunks.append({'path':chunk.name,'completion_sha256':checkpoint_digest(chunk/'complete.json')})
        (directory/'complete.json').write_text(json.dumps({'records':6,'chunks':chunks,
            'canonical_ids_labels_sha256':hashlib.sha256(json.dumps(list(target.items())).encode()).hexdigest(),
            'evaluation_config_sha256':checkpoint_digest(directory.parent/'evaluation_config.json')}))
    seal()
    return directory,seal


def test_named_moe_full_evaluation_reuses_cache_and_shared_metrics(completed_moe, target_cache, tmp_path):
    run,cache,sources,target = completed_moe;features,_ = target_cache
    before = {name: checkpoint_digest(run/name) for name in ('metrics.json','best.pt','source_calibration.json')}
    output = tmp_path/'evaluations/lacf_layer_moe/f1/1234'
    report = evaluator.evaluate(run,output,features,tmp_path,'cpu',logging.getLogger('eval'),bootstrap_resamples=0)
    assert report['model']=='lacf_layer_moe' and report['model_family']=='lacf_layer_moe'
    assert report['target']['n_bona_fide']==2 and report['target']['n_spoof']==4
    assert set(report['strict_raw_transfer']['target_operating_points'])=={'1%','5%','10%'}
    assert report['unlabeled_zscore_transfer']['transductive']
    assert json.loads((output/'evaluation_complete.json').read_text())['no_target_tuning']
    assert json.loads((output/'source_validation.json').read_text())['passed']
    assert all(checkpoint_digest(run/name)==sha for name,sha in before.items())
    with pytest.raises(FileExistsError):
        evaluator.evaluate(run,output,features,tmp_path,'cpu',logging.getLogger('eval'))


@pytest.mark.parametrize('artifact',['config.json','best.pt','source_calibration.json','source_dev_asv5.csv','training_complete.json'])
def test_changed_training_artifact_stops_before_target(completed_moe, tmp_path, monkeypatch, artifact):
    run,*_ = completed_moe
    path = run/artifact
    if artifact=='training_complete.json':path.unlink()
    elif artifact=='config.json':
        value=json.loads(path.read_text());value['seed']=1235;path.write_text(json.dumps(value))
    else:
        with path.open('ab') as stream:stream.write(b'changed')
    monkeypatch.setattr(evaluator,'target_chunks',lambda *args:pytest.fail('target must remain unopened'))
    with pytest.raises((ValueError,FileNotFoundError)):
        evaluator.evaluate(run,tmp_path/'eval/f1/1234',tmp_path/'unopened',tmp_path,'cpu',logging.getLogger('gate'))


def test_full_source_reload_mismatch_keeps_target_unopened(completed_moe, tmp_path, monkeypatch):
    run,*_ = completed_moe;original=evaluator.score_probe
    def mismatch(*args,**kwargs):
        scores,metrics=original(*args,**kwargs);scores=scores.clone();scores[0]+=1
        return scores,metrics
    monkeypatch.setattr(evaluator,'score_probe',mismatch)
    monkeypatch.setattr(evaluator,'target_chunks',lambda *args:pytest.fail('target must remain unopened'))
    with pytest.raises(ValueError,match='source selected-checkpoint reload differs'):
        evaluator.evaluate(run,tmp_path/'eval/f1/1234',tmp_path/'unopened',tmp_path,'cpu',logging.getLogger('gate'))


@pytest.mark.parametrize('change',['bytes','label','crop','revision'])
def test_reused_target_cache_rejects_changed_payload_and_semantics(completed_moe,target_cache,tmp_path,change):
    _,cache,_,_=completed_moe;features,seal=target_cache
    chunk=features/'chunk_00000'
    if change=='bytes':
        with (chunk/'wavlm_layers.npy').open('ab') as stream:stream.write(b'changed')
    elif change in ('label','crop'):
        rows=json.loads((chunk/'records.json').read_text())
        if change=='label':rows[0]['label']=1
        else:rows[0]['intervals'][0][0]+=1
        (chunk/'records.json').write_text(json.dumps(rows));seal()
    else:
        p=features.parent/'evaluation_config.json';value=json.loads(p.read_text())
        value['frozen_encoder_revisions']['wavlm_revision']='different';p.write_text(json.dumps(value));seal()
    with pytest.raises(ValueError):list(evaluator.target_chunks(features,cache,tmp_path))
