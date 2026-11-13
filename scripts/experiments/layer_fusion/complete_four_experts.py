"""Finish authorized four-expert F3/F4 using unchanged training and cached evaluation."""

import gc
import itertools
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from audio_deepfake_detection.metrics import write_score_csv
from audio_deepfake_detection.protocol import FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, save_score_manifest
from scripts.evaluate_scores import evaluate_run
from scripts.experiments.layer_fusion import evaluate_four_experts as four
from scripts.experiments.layer_fusion.evaluate_four_experts_cached import target_chunks
from scripts.experiments.token_fusion.evaluate import score_features
from scripts.token_fusion.train import ROOT, check_guards, replace_json, save_json

read = four.read
DATA = Path('/mnt/drive/audio-deepfake-cache')
TRAIN_LAUNCH = four.FAMILY / 'launch_f3_f4'
LAUNCH = four.OUTPUT / 'launch_f3_f4'
MAIN_EVALUATION = ROOT / 'outputs/lacf_moe/evaluations'
F2_LAUNCH = four.OUTPUT / 'launch_cached_f2'


def wait_for_record(path, producer_launch, logger, message, *, timeout=48*3600):
    """Operational wait only; do not inspect target content or alter a producer."""
    started = time.monotonic()
    while not path.is_file():
        if (producer_launch/'exit.json').is_file():
            raise RuntimeError('Producer exited before required completion: '+str(path))
        record = read(producer_launch/'launch_verified.json')
        if not Path(f"/proc/{record['pid']}").exists():
            raise RuntimeError('Required producer process absent')
        if time.monotonic()-started > timeout:
            raise TimeoutError('Completion wait exceeded48h')
        replace_json(LAUNCH/'progress.json',{'phase':'waiting','reason':message})
        logger.info('WAIT %s',message)
        time.sleep(30)


def wait_for_gpu_headroom(logger, *, minimum_free_mib=4096, timeout=48*3600):
    """Use measured spare memory; source fitting need not await target inference."""
    started = time.monotonic()
    while True:
        free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                   '--format=csv,noheader,nounits'],text=True).strip())
        if free >= minimum_free_mib:
            save_json(LAUNCH/'gpu_headroom_release.json',{'gpu_free_mib':free,
                'minimum_free_mib':minimum_free_mib,'training_allocator_budget_gib':1,
                'other_jobs_untouched':True})
            logger.info('GPU HEADROOM PASSED: %dMiB free; start cached training without waiting for encoder evaluation',free)
            return free
        if time.monotonic()-started > timeout:
            raise TimeoutError('Insufficient GPU headroom within48h; active jobs preserved')
        replace_json(LAUNCH/'progress.json',{'phase':'waiting for GPU headroom',
                     'gpu_free_mib':free,'minimum_free_mib':minimum_free_mib})
        logger.info('WAIT GPU headroom: %dMiB free; require%dMiB; active jobs unchanged',free,minimum_free_mib)
        time.sleep(30)


def verify_preserved():
    for fold in ('f1','f2'):
        four.verify_training(four.FAMILY/fold/'1234',fold,DATA,four.FAMILY)


def evaluate(training, output, producer, data_root, device, logger, *, fold, launch, bootstrap_resamples=1000):
    if output.exists():
        raise FileExistsError('Existing evaluation preserved; no overwrite/retry')
    config,done,training_guards,reconciliation = four.verify_training(training,fold,data_root,four.FAMILY)
    model = four.load_detector(training,device)
    validation = four.source_reload(model,training,config,data_root,logger,device)
    execution = read(launch/'provenance.json')
    check_guards(execution)
    output.mkdir(parents=True,exist_ok=False)
    for name in ('best.pt','config.json','training_complete.json','source_calibration.json'):
        shutil.copy2(training/name,output/name)
    shutil.copy2(training/'metrics.json',output/'source_metrics.json')
    validation['trainer_provenance_reconciliation'] = reconciliation
    save_json(output/'source_validation.json',validation)
    save_json(output/'evaluation_config.json',{'model_name':four.NAME,'fold':fold,'seed':1234,
        'training_run':str(training.resolve()),'target_feature_producer':str(producer.resolve()),
        'encoder_execution':False,'head_batch':132,'precision':config['precision'],
        'provenance':execution,'source_calibration':'unchanged from this four-expert selected checkpoint',
        'bootstrap_resamples':bootstrap_resamples,'bootstrap_seed':2026,
        'exploratory_after_prior_target_exposure':True,'target_tuning':False})
    manifest = load_score_manifest(output,fold,1234,done['best_checkpoint_sha256'],create=True)
    for domain in FOLDS[fold]['sources']:
        name = f'source_dev_{domain}.csv'
        shutil.copy2(training/name,output/name)
        manifest['score_sha256'][name] = checkpoint_digest(output/name)
    cache = Path(config['provenance']['cache'])
    ids,values = [],[]
    for identifiers,arrays in target_chunks(producer,cache,fold,data_root):
        check_guards(execution)
        ids.extend(identifiers);values.extend(score_features(model,arrays,device).tolist())
        replace_json(output/'progress.json',{'phase':'four-expert scoring on reused target features','scored':len(ids)})
        logger.info('%s four-expert cached target scoring: %d recordings',fold.upper(),len(ids))
    spec = FOLDS[fold]
    expected = canonical_score_labels(spec['target'],spec['target_split'],data_root)
    if ids != list(expected) or len(values) != len(expected):
        raise ValueError('Incomplete/reordered target export')
    target = output/'target_scores.csv'
    write_score_csv(target,zip(ids,itertools.repeat(spec['target']),expected.values(),values))
    features = producer/'target_features'
    manifest.update(model_family='lacf_layer_moe4',training_run=str(training.resolve()),
        inference={'input':'centered common physical interval up to10s; native unpadded frozen features',
                   'precision':config['precision'],'head_batch':132,'score':'scalar spoof logit',
                   'encoder_execution':False},
        source_validation_sha256=checkpoint_digest(output/'source_validation.json'),
        target_features_sha256=checkpoint_digest(features/'complete.json'),target_features_path=str(features.resolve()),
        scorer_sha256=execution['code_sha256'],exploratory_after_prior_target_exposure=True)
    manifest['score_sha256'][target.name] = checkpoint_digest(target)
    save_score_manifest(output,manifest)
    del model,arrays;gc.collect()
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    replace_json(output/'progress.json',{'phase':'metrics and conditional bootstrap','scored':len(ids),'total':len(ids)})
    report = evaluate_run(output,fold,seed=1234,source_data_root=data_root,target_data_root=data_root,bootstrap_resamples=bootstrap_resamples)
    check_guards(execution);check_guards(training_guards)
    save_json(output/'evaluation_complete.json',{'model_name':four.NAME,'fold':fold,'seed':1234,
        'checkpoint_sha256':done['best_checkpoint_sha256'],'metrics_sha256':checkpoint_digest(output/'metrics.json'),
        'source_validation_sha256':manifest['source_validation_sha256'],'score_sha256':manifest['score_sha256'],
        'target_features_sha256':manifest['target_features_sha256'],'no_target_tuning':True,
        'source_selected_checkpoint_only':True,'exploratory_after_prior_target_exposure':True})
    replace_json(output/'progress.json',{'phase':'complete','scored':len(ids),'total':len(ids)})
    logger.info('COMPLETE four-expert %s | EER %.4f%% AUROC %.6f',fold.upper(),report['target']['eer']*100,report['target']['auroc'])
    return report

def main():
    if (LAUNCH/'started.json').exists() or any((four.FAMILY/f/'1234').exists() for f in ('f3','f4')):
        raise FileExistsError('Prior continuation/run preserved; no retry/overwrite/resume')
    provenance = read(TRAIN_LAUNCH/'provenance.json');check_guards(provenance)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logger = logging.getLogger('moe4_continuation')
    verify_preserved()
    save_json(LAUNCH/'started.json',{'phase':'wait for existing four-expert F2 evaluation before GPU training',
        'fresh_training_folds':['f3','f4'],'preserved_training_folds':['f1','f2'],
        'reuse_main_target_features':True,'automatic_retry':False})
    wait_for_record(F2_LAUNCH/'exit.json',F2_LAUNCH,logger,'existing four-expert F2 cached evaluation must finish first')
    if not read(F2_LAUNCH/'exit.json')['verified_complete']:
        raise RuntimeError('Existing F2 evaluation failed; continuation held, no automatic retry')
    free=wait_for_gpu_headroom(logger)
    verify_preserved();check_guards(provenance)
    current_free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if current_free < 4096:
        raise RuntimeError('GPU headroom fell below4GiB before training; active jobs preserved')
    free=current_free
    save_json(TRAIN_LAUNCH/'started.json',{'phase':'fresh F3 then F4 with existing trainer',
        'gpu_free_mib':free,'gpu_allocator_budget_gib':1,'preserved_f1_f2':True})
    replace_json(LAUNCH/'progress.json',{'phase':'fresh F3 then F4 training; see console and per-fold logs'})
    logger.info('START fresh four-expert F3 then F4; original trainer/model/recipe unchanged')
    result=subprocess.run([sys.executable,'-u','-m','scripts.experiments.layer_fusion.train_four_experts',
                           '--folds','f3','f4'],cwd=ROOT)
    try:
        if result.returncode:
            raise RuntimeError('Fresh training failed; no retry or target evaluation')
        for fold in FOLDS:
            four.verify_training(four.FAMILY/fold/'1234',fold,DATA,four.FAMILY)
    except Exception as error:
        save_json(TRAIN_LAUNCH/'exit.json',{'trainer_exit':result.returncode,'verified_complete':False,'error':str(error)})
        raise
    save_json(TRAIN_LAUNCH/'exit.json',{'trainer_exit':0,'verified_complete':True,
              'completion_sha256':{f:checkpoint_digest(four.FAMILY/f/'1234/training_complete.json') for f in ('f3','f4')}})
    logger.info('All four source fits verified; evaluate remaining folds using existing target caches')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(512*1024**2/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    for fold in ('f3','f4'):
        producer=MAIN_EVALUATION/fold/'1234'
        wait_for_record(producer/'evaluation_complete.json',MAIN_EVALUATION/'launch',logger,
                        fold.upper()+' main evaluation must finish its reusable target cache')
        check_guards(provenance)
        replace_json(LAUNCH/'progress.json',{'phase':'cached target evaluation','fold':fold})
        evaluate(four.FAMILY/fold/'1234',four.OUTPUT/fold/'1234',producer,DATA,'cuda',logger,fold=fold,launch=LAUNCH)
    replace_json(LAUNCH/'progress.json',{'phase':'all four folds complete'})
    save_json(LAUNCH/'complete.json',{'folds':list(FOLDS),'preserved_f1_f2':True,
        'evaluation_completion_sha256':{f:checkpoint_digest(four.OUTPUT/f/'1234/evaluation_complete.json') for f in FOLDS}})


if __name__=='__main__':main()
