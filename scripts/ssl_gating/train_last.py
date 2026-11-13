"""Fresh last-block SSL F1; reuse approved partial training/scoring helpers."""

import gc
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, utterance_id
from audio_deepfake_detection.sota.ssl_gating import FullAudioDataset, make_train_loader, make_eval_loader
from audio_deepfake_detection.sota.ssl_gating_last import LAST_MODEL_CONFIG, RUN_FAMILY, TRAINABLE_PARAMETERS, LastSSLFeatureGating
from scripts.ssl_gating.train import LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, set_seed, write_json
from scripts.ssl_gating.train_partial import save_checkpoint, score_sources, source_metrics, train_epoch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'outputs'/RUN_FAMILY/'f1/1234'
LAUNCH = ROOT/'outputs'/RUN_FAMILY/'f1/launch'
DATA = Path('/mnt/drive/audio-deepfake-cache')


def check_guards():
    for path,digest in json.loads((LAUNCH/'source_guards.json').read_text()).items():
        if checkpoint_digest(ROOT/path)!=digest:
            raise RuntimeError('Execution/preserved source changed: '+path)


def main():
    if RUN.exists():raise FileExistsError('Existing run preserved; no resume or overwrite')
    check_guards()
    validation=json.loads((LAUNCH/'source_validation.json').read_text())
    if (not validation['passed'] or not validation['fresh_reload_source_score_bit_exact']
            or not validation['probe_weights_discarded'] or validation['trainable_parameters']!=TRAINABLE_PARAMETERS
            or validation['model_config']!=LAST_MODEL_CONFIG
            or validation['source_guards_sha256']!=checkpoint_digest(LAUNCH/'source_guards.json')):
        raise ValueError('Matching source validation required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required; no fallback')
    torch.cuda.set_per_process_memory_fraction(19*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    set_seed(1234)
    sources=FOLDS['f1']['sources']
    records={d:{s:read_dataset(d,s,DATA) for s in ('train','dev')} for d in sources}
    train=make_train_loader([FullAudioDataset(records[d]['train'],training=True) for d in sources],
                            batch_size=24,seed=1234,num_workers=8,prefetch_factor=2)
    dev={d:make_eval_loader(FullAudioDataset(records[d]['dev']),num_workers=4,prefetch_factor=2) for d in sources}
    model=LastSSLFeatureGating().cuda()
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count!=TRAINABLE_PARAMETERS:raise ValueError('Last-block trainable inventory differs')
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    config={'fold':'f1','seed':1234,'run_name':RUN_FAMILY,'model_name':'SSLFeatureGating','model_config':LAST_MODEL_CONFIG,
            'frontend_config':model.feature_extractor.config.to_dict(),'sources':list(sources),'target_domain':'speechfake','target_access':False,
            'data_root':str(DATA),'dataset_counts':{d:{s:len(v) for s,v in parts.items()} for d,parts in records.items()},
            'batch_size':24,'effective_batch':24,'accumulation_steps':1,'num_workers':8,'eval_workers':4,'prefetch_factor':2,
            'learning_rate':LEARNING_RATE,'weight_decay':WEIGHT_DECAY,'optimizer':'Adam','scheduler':'constant',
            'maximum_epochs':MAX_EPOCHS,'patience':PATIENCE,'train_max_seconds':20,'eval_batch_size':1,
            'precision':'BF16_autocast_FP32_pooling_losses_scores','loss':'weighted_CE_0.9_0.1_plus_CKA_coefficient1',
            'selection':'strict_equal_domain_macro_source_dev_eer_ties_earlier',
            'trainable_parameters':count,'total_parameters':sum(p.numel() for p in model.parameters()),
            'initialization':'fresh pinned official XLS-R plus fresh backend; source-check weights discarded',
            'source_sha256':json.loads((LAUNCH/'source_guards.json').read_text()),
            'source_validation_sha256':checkpoint_digest(LAUNCH/'source_validation.json'),
            'readiness_sha256':checkpoint_digest(LAUNCH/'readiness.json'),'gpu_memory_budget_gib':19,
            'snapshot_policy':'atomic15minute/end_training_phase snapshots; no automatic resume',
            'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'torch':str(torch.__version__),'transformers':__import__('transformers').__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()}
    RUN.mkdir(parents=True,exist_ok=False);write_json(RUN/'config.json',config)
    if (LAUNCH/'console.log').is_file():
        (RUN/'training.log').hardlink_to(LAUNCH/'console.log')
    print(f'START {RUN_FAMILY}/f1/1234 | block23 + backend | {count} trainable | batch24 BF16/FP32 | target unopened',flush=True)
    def snapshot(epoch,step,seen):
        save_checkpoint(RUN/'recovery.pt',{'config':config,'epoch':epoch,'step':step,'recordings_seen':seen,
            'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'kind':'intermediate_training_snapshot_not_completion','exact_mid_epoch_resume_supported':False})
        print(f'SNAPSHOT epoch{epoch} step{step}; not a completion marker',flush=True)
    best,best_epoch,stale,best_scores=math.inf,0,0,None
    started=time.monotonic()
    for epoch in range(1,MAX_EPOCHS+1):
        check_guards()
        losses=train_epoch(model,train,optimizer,epoch,snapshot,15*60)
        scored=score_sources(model,dev);metrics=source_metrics(scored)
        if metrics['macro_eer']<best:
            best,best_epoch,stale=metrics['macro_eer'],epoch,0
            best_scores={d:s.copy() for d,(_,s) in scored.items()};names=('last.pt','best.pt')
        else:stale+=1;names=('last.pt',)
        payload={'config':config,'fold':'f1','epoch':epoch,'model_state_dict':model.state_dict(),
                 'optimizer_state_dict':optimizer.state_dict(),'source_dev':metrics,'macro_dev_eer':metrics['macro_eer'],
                 'train_losses':losses,'best_macro_dev_eer':best,'bad_epochs':stale}
        for name in names:save_checkpoint(RUN/name,payload)
        with (RUN/'history.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch':epoch,'losses':losses,'source_dev':metrics,'best_epoch':best_epoch,'patience':stale})+'\n')
        print(f'EPOCH {epoch} macro_source_EER={100*metrics["macro_eer"]:.4f}% best={best_epoch} patience={stale}/{PATIENCE}',flush=True)
        if stale>=PATIENCE:break
    state=torch.load(RUN/'best.pt',map_location='cpu',weights_only=True)
    model.cpu();del model,optimizer,payload;gc.collect();torch.cuda.empty_cache()
    fresh=LastSSLFeatureGating.from_frontend_config(config['frontend_config']).cuda().eval()
    fresh.load_state_dict(state['model_state_dict'],strict=True)
    reloaded=score_sources(fresh,dev)
    if source_metrics(reloaded)!=state['source_dev'] or any(not (reloaded[d][1]==best_scores[d]).all() for d in sources):
        raise RuntimeError('Fresh selected-checkpoint full-source reload differs')
    hashes={}
    for domain,(labels,scores) in reloaded.items():
        path=RUN/f'source_dev_{domain}.csv'
        write_score_csv(path,((utterance_id(p,domain,DATA),domain,y,float(s)) for (p,y),s in zip(records[domain]['dev'],scores)))
        hashes[path.name]=checkpoint_digest(path)
    write_json(RUN/'source_calibration.json',calibrate_source_scores(reloaded))
    check_guards()
    write_json(RUN/'training_complete.json',{'fold':'f1','seed':1234,'best_epoch':best_epoch,'final_epoch':epoch,
        'best_macro_source_dev_eer':best,'best_checkpoint_sha256':checkpoint_digest(RUN/'best.pt'),
        'source_scores_sha256':hashes,'source_calibration_sha256':checkpoint_digest(RUN/'source_calibration.json'),
        'selected_checkpoint_reloaded_and_source_verified':True,'wall_seconds':time.monotonic()-started,'target_access':False})
    print('COMPLETE: full source reload/calibration passed; no target evaluation',flush=True)


if __name__=='__main__':main()
