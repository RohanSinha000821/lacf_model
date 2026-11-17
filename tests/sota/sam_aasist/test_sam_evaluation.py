"""SAM evaluation gates, frozen source calibration, and artifact preservation."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from audio_deepfake_detection.metrics import calibrate_source_scores, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, dataset_root, utterance_id
from scripts.sam_aasist import evaluate as ev
from scripts.sam_aasist.train import source_identity, source_metrics


class Tiny(nn.Module):
    def forward(self, waves):
        return torch.stack((torch.zeros_like(waves[:, 0]), waves[:, 0]), dim=1)

    @staticmethod
    def spoof_score(logits):
        return logits[:, 1] - logits[:, 0]


@pytest.fixture
def completed(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    run = tmp_path / 'outputs' / 'sam_aasist' / 'f1' / '1234'
    run.mkdir(parents=True)
    records = {d: [(dataset_root(root, d) / 'z.wav', 0), (dataset_root(root, d) / 'a.wav', 1)]
               for d in FOLDS['f1']['sources']}
    scores = {d: (np.array([0, 1]), np.array([-2., 2.], dtype=np.float32)) for d in records}
    config = dict(
        fold='f1', seed=1234, run_name='sam_aasist', model_name='SAM-AASIST',
        model_config=copy.deepcopy(ev.MODEL_CONFIG), sam_upstream_commit=ev.SAM_UPSTREAM_COMMIT,
        sources=list(records), target='speechfake', target_access=False,
        parameter_count=297866, trainable_parameters=297866, precision='float32', batch_size=24,
        gradient_accumulation_steps=1, eval_batch_size=1, train_input='random64600_after16k_resampling_repeat_short',
        eval_input='first64600_repeat_short', rawboost_algorithm=0, freq_aug=False,
        optimizer='nonadaptive_SAM_Adam', rho=.05, learning_rate=1e-4, weight_decay=1e-4,
        minimum_learning_rate=5e-6, max_epochs=100, patience=5, class_weights=[.9,.1],
        scheduler='per_Adam_update_cosine', sampler='equal_domain_natural_classes_replacement_N_draws',
        selection='strict_equal_domain_macro_source_dev_eer_ties_earlier', cuda_matmul_tf32=False,
        cudnn_tf32=True, cudnn_deterministic=True, cudnn_benchmark=False,
        label_order=['bona_fide','spoof'], score='spoof_logit_minus_bona_fide_logit',
        source_identity=source_identity({d:{'dev':r} for d,r in records.items()}, root))
    checkpoint = dict(config=config, fold='f1', epoch=14, macro_dev_eer=0., model_state_dict={},
                      source_dev=source_metrics(scores),
                      source_dev_scores={d:torch.from_numpy(s) for d,(_,s) in scores.items()})
    torch.save(checkpoint, run/'best.pt')
    (run/'config.json').write_text(json.dumps(config))
    hashes = {}
    for d, rows in records.items():
        path=run/f'source_dev_{d}.csv'
        write_score_csv(path, ((utterance_id(p,d,root), d, y, float(s)) for (p,y),s in zip(rows,scores[d][1])))
        hashes[path.name]=checkpoint_digest(path)
    (run/'source_calibration.json').write_text(json.dumps(calibrate_source_scores(scores)))
    completion=dict(fold='f1',seed=1234,best_epoch=14,final_epoch=19,best_macro_source_dev_eer=0.,
                    selected_checkpoint_reloaded_and_source_verified=True,target_access=False,
                    best_checkpoint_sha256=checkpoint_digest(run/'best.pt'),source_scores_sha256=hashes,
                    source_calibration_sha256=checkpoint_digest(run/'source_calibration.json'))
    (run/'training_complete.json').write_text(json.dumps(completion))
    monkeypatch.setattr(ev,'check_guards',lambda c:None)
    monkeypatch.setattr(ev,'evaluation_guards',lambda:{'fixed':'sha'})
    def read(d, split, data_root):
        assert split=='dev' and d in records, 'Target opened before source validation'
        return records[d]
    monkeypatch.setattr(ev,'read_dataset',read)
    args=SimpleNamespace(fold='f1',seed=1234,source_data_root=root,target_data_root=root,
                         num_workers=0,prefetch_factor=4,cpu_threads=2,bootstrap_resamples=8)
    return run,config,checkpoint,completion,records,scores,args


@pytest.mark.parametrize('field,value',[
    ('model_name','AASIST'),('precision','bfloat16'),('eval_input','full_recording'),
    ('batch_size',12),('rho',.01),('rawboost_algorithm',4),('cuda_matmul_tf32',True),
    ('sources',['asv2019','asv5','speechfake']),('score','bona_fide_minus_spoof'),
    ('model_config',{'pool_ratios':[.5,.7,.5,.5]}),
])
def test_rejects_changed_recipe(completed,field,value):
    _,config,checkpoint,completion,*_=completed
    config[field]=value
    with pytest.raises(ValueError,match='configuration mismatch'):
        ev.verify_checkpoint(checkpoint,completion,config,'f1',1234)


def test_selected_epoch_and_completion_gate(completed):
    _,config,checkpoint,completion,*_=completed
    ev.verify_checkpoint(checkpoint,completion,config,'f1',1234)
    for key,value in [('best_epoch',13),('selected_checkpoint_reloaded_and_source_verified',False)]:
        changed={**completion,key:value}
        with pytest.raises(ValueError,match='source-reload completion'):
            ev.verify_checkpoint(checkpoint,changed,config,'f1',1234)
    with pytest.raises(ValueError,match='saved configuration'):
        ev.verify_checkpoint(checkpoint,completion,{**config,'extra':'changed'},'f1',1234)


def test_source_csv_hash_and_frozen_calibration_block_target(completed,monkeypatch):
    run,config,checkpoint,completion,records,scores,args=completed
    observed,_=ev.verified_sources(run,config,completion,checkpoint,args.source_data_root)
    assert np.array_equal(observed['asv2019'][1],[-2.,2.])  # not canonical sorted order
    monkeypatch.setattr(ev,'score_split',lambda *a,**k:pytest.fail('Target must remain unopened'))
    path=run/'source_dev_asv2019.csv';path.write_text(path.read_text().replace('-2.0','-1.0'))
    with pytest.raises(ValueError,match='source CSV changed'):
        ev.evaluate_run(args,run,torch.device('cpu'))
    assert not (run/'source_validation.json').exists()


def test_calibration_must_reproduce_without_tolerance(completed):
    run,config,checkpoint,completion,_,_,args=completed
    path=run/'source_calibration.json';cal=json.loads(path.read_text())
    cal['source_stats']['asv2019']['mean']+=1e-14
    path.write_text(json.dumps(cal));completion['source_calibration_sha256']=checkpoint_digest(path)
    with pytest.raises(ValueError,match='calibration does not reproduce exactly'):
        ev.verified_sources(run,config,completion,checkpoint,args.source_data_root)


def test_fresh_source_parity_mismatch_fails(completed,monkeypatch):
    _,_,_,_,records,scores,_=completed
    class Wrong:
        def __init__(self,rows,training): pass
        def __getitem__(self,i): return torch.tensor([-1.]),0
    monkeypatch.setattr(ev,'AASISTDataset',Wrong)
    with pytest.raises(ValueError,match='Fresh source score differs'):
        ev.source_parity(Tiny(),records,scores,torch.device('cpu'))


def test_cpu_flow_preserves_sources_and_binds_completion(completed,monkeypatch):
    run,_,_,_,_,_,args=completed
    before={p.name:p.read_bytes() for p in run.iterdir()}
    monkeypatch.setattr(ev,'SAMAASIST',Tiny)
    class Views:
        def __init__(self,rows,training):self.rows=rows;assert not training
        def __getitem__(self,i):
            y=self.rows[i][1];return torch.tensor([2. if y else -2.]),y
    monkeypatch.setattr(ev,'AASISTDataset',Views)
    def target(model,domain,split,root,path,device,*,manifest,**kwargs):
        assert json.loads((run/'source_validation.json').read_text())['status']=='passed'
        assert domain=='speechfake' and split=='test'
        write_score_csv(path,[('b','speechfake',0,-1.),('s','speechfake',1,1.)])
        manifest['score_sha256'][path.name]=checkpoint_digest(path)
        ev.save_score_manifest(run,manifest)
        return np.array([0,1]),np.array([-1.,1.])
    monkeypatch.setattr(ev,'score_split',target)
    ev.evaluate_run(args,run,torch.device('cpu'))
    marker=json.loads((run/'evaluation_complete.json').read_text())
    assert marker['metrics_sha256']==checkpoint_digest(run/'metrics.json')
    assert marker['target_scores_sha256']==checkpoint_digest(run/'target_scores.csv')
    assert marker['target_recordings']==2
    assert all((run/name).read_bytes()==data for name,data in before.items())
    with pytest.raises(FileExistsError,match='previous evaluation artifact'):
        ev.evaluate_run(args,run,torch.device('cpu'))


@pytest.mark.parametrize('artifact',['target_scores.csv.tmp','source_validation.json','metrics.json'])
def test_preserves_incomplete_attempt(completed,artifact):
    run,*rest=completed;args=rest[-1];path=run/artifact;path.write_text('preserve')
    with pytest.raises(FileExistsError):ev.evaluate_run(args,run,torch.device('cpu'))
    assert path.read_text()=='preserve'
