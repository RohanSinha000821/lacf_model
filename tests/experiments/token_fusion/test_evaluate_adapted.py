"""CPU-only adapted checkpoint/source gates and shared metric export."""
import hashlib
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import checkpoint_digest
from audio_deepfake_detection.token_fusion.adaptation_data import SelectedCropDataset
from scripts.experiments.token_fusion import evaluate_adapted as e


class TinyDetector(nn.Module):
    def __init__(self,configuration):
        super().__init__();self.linear=nn.Linear(1,1);self.spec=configuration
    def forward(self,batch):return {'logit':self.linear(batch['features']).squeeze(-1)}
    def configuration(self):return self.spec


class SyntheticLoader:
    def __init__(self,dataset,size):self.dataset,self.batch_size=dataset,size
    def __iter__(self):
        for start in range(0,len(self.dataset),self.batch_size):
            labels=torch.tensor([label for _,label in self.dataset.records[start:start+self.batch_size]])
            yield {'features':labels[:,None].float()*4,'labels':labels}


@pytest.fixture
def completed(tmp_path,monkeypatch):
    run=tmp_path/'training/f1/1234';run.mkdir(parents=True)
    output=tmp_path/'evaluation/f1/1234';launch=tmp_path/'launch';launch.mkdir()
    (launch/'provenance.json').write_text('{"code_sha256":{}}')
    monkeypatch.setattr(e,'LAUNCH',launch)
    cache=tmp_path/'cache';cache.mkdir();(cache/'config.json').write_text('{}')
    monkeypatch.setattr(e.trainer,'CACHE',cache)
    labels={f'id{i}':i%2 for i in range(5)}
    bank={d:SelectedCropDataset([(f'{d}/{key}.wav',label) for key,label in labels.items()],
          [f'{d}/dev/{key}' for key in labels],training=False) for d in e.SOURCES}
    membership={f'{d}/dev':{'records':5,'canonical_ids_labels_sha256':hashlib.sha256(json.dumps(list(labels.items())).encode()).hexdigest()} for d in e.SOURCES}
    (cache/'complete.json').write_text(json.dumps({'splits':membership}))
    model_config={'clap_final_stage':True,'rank':8,'alpha':16,'lora_lr':1e-4,'clap_lr':1e-5,
                  'fusion_lr':3e-4,'weight_decay':1e-4,'trainable_parameters':e.PARAMETERS,
                  'objective':'TF2 detection BCE only'}
    revisions={'wavlm_revision':'pinned_w','clap_revision':'pinned_c'}
    config={'mode':'train','condition':'wavlm_lora_clap','full_f1':True,'held_source':None,'fold':'f1','seed':1234,
        'training_sources':list(e.SOURCES),'development_sources':list(e.SOURCES),'model':model_config,
        'target_access':False,'provenance':{'code_sha256':{}},'source_partitions':membership,
        'cache_config_sha256':checkpoint_digest(cache/'config.json'),'pretrained':revisions,
        'physical_batch':2,'development_workers':0,'precision':{'autocast':'bfloat16'}}
    (run/'config.json').write_text(json.dumps(config))
    model=TinyDetector(model_config)
    with torch.no_grad():model.linear.weight.fill_(.25);model.linear.bias.zero_()
    source={}
    hashes={}
    for d in e.SOURCES:
        path=run/f'source_dev_{d}.csv'
        write_score_csv(path,[(key,d,label,float(label)) for key,label in labels.items()])
        source[d]=read_score_csv(path,d,expected_labels=labels);hashes[path.name]=checkpoint_digest(path)
    calibration=calibrate_source_scores(source)
    (run/'source_calibration.json').write_text(json.dumps(calibration))
    metrics={'source_dev':calibration['source_dev'],'best_epoch':2,'final_epoch':7}
    (run/'metrics.json').write_text(json.dumps(metrics))
    torch.save({'trainable_state':model.state_dict(),'epoch':2,'source_dev':metrics['source_dev'],
                'run_config_sha256':checkpoint_digest(run/'config.json')},run/'best_adaptation.pt')
    done={'fold':'f1','seed':1234,'full_f1':True,'held_source':None,'sources':list(e.SOURCES),
          'condition':'wavlm_lora_clap','selected_checkpoint_reloaded_and_source_verified':True,
          'target_access':False,'best_epoch':2,'final_epoch':7,'best_macro_source_dev_eer':calibration['source_dev']['macro_eer'],
          'config_sha256':checkpoint_digest(run/'config.json'),'checkpoint_sha256':checkpoint_digest(run/'best_adaptation.pt'),
          'metrics_sha256':checkpoint_digest(run/'metrics.json'),'calibration_sha256':checkpoint_digest(run/'source_calibration.json'),
          'source_scores_sha256':hashes}
    (run/'complete.json').write_text(json.dumps(done))
    (launch/'provenance.json').write_text(json.dumps({'code_sha256':{},
        'expected_config_sha256':done['config_sha256'],'expected_checkpoint_sha256':done['checkpoint_sha256']}))
    monkeypatch.setattr(e,'source_split',lambda root,d,part:bank[d])
    def canonical(d,part,root,**kw):
        if d=='speechfake':assert (output/'source_validation.json').is_file()
        return labels
    monkeypatch.setattr(e,'canonical_score_labels',canonical)
    monkeypatch.setattr(e,'read_dataset',lambda *a:[(f'target/{key}.wav',label) for key,label in labels.items()])
    monkeypatch.setattr(e,'make_audio_loader',lambda dataset,collator,*,batch_size,workers,seed:SyntheticLoader(dataset,batch_size))
    monkeypatch.setattr(e.trainer,'load_model',lambda config,seed,device:(TinyDetector(model_config).to(device),None,revisions))
    monkeypatch.setattr(e,'source_memory_check',lambda *a:{'passed':True,'test_only_mock':True})
    return run,output,config,bank


def test_full_source_binding_reconstruction_subset_parity_tail_and_metrics(completed,tmp_path):
    run,output,config,bank=completed
    before={p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    assert e.verify_training(run,tmp_path)[1]['best_epoch']==2
    model,collator=e.reconstruct(run,config,'cpu')
    assert torch.equal(model.linear.weight,torch.tensor([[.25]]))
    result=e.evaluate(run,output,tmp_path,'cpu',logging.getLogger('adapted_eval'),bootstrap_resamples=5)
    assert result['model_family']=='tf2_adapted' and result['target']['n_bona_fide']==3 and result['target']['n_spoof']==2
    validation=e.read(output/'source_validation.json')
    assert validation['full_source_reload']['records']=={d:5 for d in e.SOURCES}
    assert all(x['records']==3 and x['score_bit_exact'] for x in validation['fresh_evaluator_source_subset'])
    assert e.read(output/'evaluation_config.json')['checkpoint_format'].startswith('sparse trainable weights')
    assert not (output/'best.pt').exists() and not (output/'training_complete.json').exists()
    assert before=={p.name:checkpoint_digest(p) for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):e.evaluate(run,output,tmp_path,'cpu',logging.getLogger('retry'))


@pytest.mark.parametrize('file',['best_adaptation.pt','config.json','source_calibration.json','metrics.json','source_dev_asv5.csv'])
def test_changed_artifact_prevents_target_access(completed,tmp_path,file):
    run,output,config,bank=completed
    path=run/file;path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):e.evaluate(run,output,tmp_path,'cpu',logging.getLogger('changed'))
    assert not output.exists()


def test_missing_completion_never_opens_target(completed,tmp_path):
    run,output,config,bank=completed;(run/'complete.json').unlink()
    with pytest.raises(FileNotFoundError):e.evaluate(run,output,tmp_path,'cpu',logging.getLogger('incomplete'))
    assert not output.exists()


def test_source_score_mismatch_stops_before_target(completed,tmp_path,monkeypatch):
    run,output,config,bank=completed
    original=e.trainer.score_audio
    def changed(*args,**kw):
        scores,labels=original(*args,**kw);return scores+1,labels
    monkeypatch.setattr(e.trainer,'score_audio',changed)
    with pytest.raises(ValueError,match='source scores differ'):
        e.evaluate(run,output,tmp_path,'cpu',logging.getLogger('mismatch'))
    assert not output.exists()


def test_sparse_reconstruction_rejects_missing_weight(completed):
    run,output,config,bank=completed
    state=torch.load(run/'best_adaptation.pt',weights_only=True);state['trainable_state'].pop('linear.bias')
    torch.save(state,run/'best_adaptation.pt')
    with pytest.raises(RuntimeError):e.reconstruct(run,config,'cpu')


def test_failed_training_verification_stops_wait(tmp_path,monkeypatch):
    producer=tmp_path/'producer';producer.mkdir()
    (producer/'exit_status.json').write_text('{"full_f1_source_completion_verified":false,"status":"failed"}')
    monkeypatch.setattr(e,'PRODUCER_LAUNCH',producer)
    with pytest.raises(RuntimeError,match='completion failed'):e.wait_for_training(logging.getLogger('wait'))
