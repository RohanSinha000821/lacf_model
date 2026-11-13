"""Evaluate completed four-expert F2 using the main queue's frozen target cache."""

import gc
import hashlib
import itertools
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.lacf.diagnostic_cache import crop_intervals
from audio_deepfake_detection.metrics import write_score_csv
from audio_deepfake_detection.protocol import FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, save_score_manifest
from scripts.evaluate_scores import evaluate_run
from scripts.experiments.layer_fusion import evaluate_four_experts as four
from scripts.experiments.token_fusion.evaluate import score_features
from scripts.token_fusion.train import ROOT, check_guards, replace_json, save_json

read = four.read
FOLD = 'f2'
DATA = Path('/mnt/drive/audio-deepfake-cache')
PRODUCER = ROOT / 'outputs/lacf_moe/evaluations/f2/1234'
PRODUCER_LAUNCH = ROOT / 'outputs/lacf_moe/evaluations/launch'
LAUNCH = four.OUTPUT / 'launch_cached_f2'


def wait_for_features(logger, *, timeout=48*3600):
    """Wait for a complete verified producer, without inspecting target content."""
    start = time.monotonic()
    while not (PRODUCER / 'evaluation_complete.json').is_file():
        if (PRODUCER_LAUNCH / 'exit.json').exists():
            raise RuntimeError('Producer exited without completed F2; no retry')
        record = read(PRODUCER_LAUNCH / 'launch_verified.json')
        if not Path(f"/proc/{record['pid']}").exists():
            raise RuntimeError('Producer process absent; target held')
        if time.monotonic()-start > timeout:
            raise TimeoutError('No completed target cache within48h')
        logger.info('WAIT: main three-expert F2 is still creating/verifying its target features; no encoder duplication')
        time.sleep(30)


def target_chunks(producer, source_cache, fold, data_root):
    """Check crop, revisions, canonical labels and every reused feature payload."""
    features = producer / 'target_features'
    marker, config = read(features / 'complete.json'), read(producer / 'evaluation_config.json')
    done, manifest = read(producer / 'evaluation_complete.json'), read(producer / 'score_manifest.json')
    digest = checkpoint_digest(features / 'complete.json')
    inference = {'input': 'centered common interval up to10s; native unpadded WavLM; CLAP repeatpad',
                 'head_batch':132,'wavlm_batch':32,'clap_batch':64,'chunk':1024,'workers':8,'prefetch':2,
                 'precision':'BF16 mixed; sensitive FP32','score':'scalar spoof logit'}
    spec = FOLDS[fold]
    if (done['fold'] != fold or done['seed'] != 1234 or done['target_features_sha256'] != digest
            or manifest['target_features_sha256'] != digest
            or marker['evaluation_config_sha256'] != checkpoint_digest(producer / 'evaluation_config.json')
            or config['model_name'] != 'LACF-MoE' or config['fold'] != fold or config['seed'] != 1234
            or config['target'] != spec['target']+'/'+spec['target_split']
            or Path(config['source_cache']).resolve() != source_cache.resolve()
            or config['frozen_encoder_revisions'] != read(source_cache / 'execution_accelerated.json')['pretrained']
            or config['inference'] != inference):
        raise ValueError('Reused feature extraction identity differs')
    producer_config, producer_done = read(producer / 'config.json'), read(producer / 'training_complete.json')
    if (checkpoint_digest(producer / 'config.json') != producer_done['config_sha256']
            or producer_config['provenance']['cache_completion_sha256'] != checkpoint_digest(source_cache / 'complete.json')):
        raise ValueError('Producer/source cache completion identity differs')
    expected = canonical_score_labels(spec['target'], spec['target_split'], data_root)
    canonical = list(expected.items())
    if (marker['records'] != len(canonical)
            or marker['canonical_ids_labels_sha256'] != hashlib.sha256(json.dumps(canonical).encode()).hexdigest()):
        raise ValueError('Reused canonical target membership differs')
    offset = 0
    for index, descriptor in enumerate(marker['chunks']):
        if descriptor['path'] != f'chunk_{index:05d}':
            raise ValueError('Nonsequential reused target cache')
        chunk = features / descriptor['path']
        if checkpoint_digest(chunk / 'complete.json') != descriptor['completion_sha256']:
            raise ValueError('Target shard completion changed')
        complete = read(chunk / 'complete.json')
        for name in ('records.json','wavlm_layers.npy','clap_projected.npy'):
            if checkpoint_digest(chunk / name) != complete['files_sha256'][name]:
                raise ValueError('Reused target payload changed')
        rows = read(chunk / 'records.json')
        if not rows or len(rows) != complete['records'] or offset+len(rows) > len(canonical):
            raise ValueError('Target shard count differs')
        for local,row in enumerate(rows):
            identifier,label = canonical[offset+local]
            intervals,aliases = crop_intervals(row['frames'],row['rate'],row['id'],training=False)
            if (row['index'] != offset+local or row['id'] != f"{spec['target']}/{spec['target_split']}/{identifier}"
                    or row['label'] != label or row['intervals'] != [list(x) for x in intervals] or row['view_rows'] != aliases):
                raise ValueError('Reused target crop/ID/label differs')
        arrays = {name:np.load(chunk/f'{name}.npy',allow_pickle=False) for name in ('wavlm_layers','clap_projected')}
        for name,shape in (('wavlm_layers',(len(rows),13,768)),('clap_projected',(len(rows),512))):
            if arrays[name].shape != shape or arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
                raise ValueError('Invalid reused target features')
        yield [name for name,_ in canonical[offset:offset+len(rows)]], arrays
        offset += len(rows)
    if offset != len(canonical):
        raise ValueError('Incomplete reused target cache')


def evaluate(training, output, producer, data_root, device, logger, *, bootstrap_resamples=1000):
    if output.exists():
        raise FileExistsError('Existing evaluation preserved; no overwrite/retry')
    config,done,training_guards,reconciliation = four.verify_training(training,FOLD,data_root,four.FAMILY)
    model = four.load_detector(training,device)
    validation = four.source_reload(model,training,config,data_root,logger,device)
    execution = read(LAUNCH/'provenance.json')
    check_guards(execution)
    output.mkdir(parents=True,exist_ok=False)
    for name in ('best.pt','config.json','training_complete.json','source_calibration.json'):
        shutil.copy2(training/name,output/name)
    shutil.copy2(training/'metrics.json',output/'source_metrics.json')
    validation['trainer_provenance_reconciliation'] = reconciliation
    save_json(output/'source_validation.json',validation)
    save_json(output/'evaluation_config.json',{'model_name':four.NAME,'fold':FOLD,'seed':1234,
        'training_run':str(training.resolve()),'target_feature_producer':str(producer.resolve()),
        'encoder_execution':False,'head_batch':132,'precision':config['precision'],
        'provenance':execution,'source_calibration':'unchanged from this four-expert selected checkpoint',
        'bootstrap_resamples':bootstrap_resamples,'bootstrap_seed':2026,
        'exploratory_after_prior_target_exposure':True,'target_tuning':False})
    manifest = load_score_manifest(output,FOLD,1234,done['best_checkpoint_sha256'],create=True)
    for domain in FOLDS[FOLD]['sources']:
        name = f'source_dev_{domain}.csv'
        shutil.copy2(training/name,output/name)
        manifest['score_sha256'][name] = checkpoint_digest(output/name)
    cache = Path(config['provenance']['cache'])
    ids,values = [],[]
    for identifiers,arrays in target_chunks(producer,cache,FOLD,data_root):
        check_guards(execution)
        ids.extend(identifiers);values.extend(score_features(model,arrays,device).tolist())
        replace_json(output/'progress.json',{'phase':'four-expert scoring on reused target features','scored':len(ids)})
        logger.info('F2 four-expert cached target scoring: %d recordings',len(ids))
    spec = FOLDS[FOLD]
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
    report = evaluate_run(output,FOLD,seed=1234,source_data_root=data_root,target_data_root=data_root,bootstrap_resamples=bootstrap_resamples)
    check_guards(execution);check_guards(training_guards)
    save_json(output/'evaluation_complete.json',{'model_name':four.NAME,'fold':FOLD,'seed':1234,
        'checkpoint_sha256':done['best_checkpoint_sha256'],'metrics_sha256':checkpoint_digest(output/'metrics.json'),
        'source_validation_sha256':manifest['source_validation_sha256'],'score_sha256':manifest['score_sha256'],
        'target_features_sha256':manifest['target_features_sha256'],'no_target_tuning':True,
        'source_selected_checkpoint_only':True,'exploratory_after_prior_target_exposure':True})
    replace_json(output/'progress.json',{'phase':'complete','scored':len(ids),'total':len(ids)})
    logger.info('COMPLETE four-expert F2 | EER %.4f%% AUROC %.6f',report['target']['eer']*100,report['target']['auroc'])
    return report


def main():
    if (LAUNCH/'started.json').exists() or (four.OUTPUT/'f2/1234').exists():
        raise FileExistsError('Prior launch/evaluation preserved; no automatic retry')
    provenance = read(LAUNCH/'provenance.json');check_guards(provenance)
    four.verify_training(four.FAMILY/'f2/1234',FOLD,DATA,four.FAMILY)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logger = logging.getLogger('moe4_cached_eval')
    save_json(LAUNCH/'started.json',{'phase':'waiting for verified main F2 target features','fold':'f2',
        'held_folds':{'f3':'training absent','f4':'training absent'},'no_new_training':True})
    wait_for_features(logger)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(512*1024**2/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    evaluate(four.FAMILY/'f2/1234',four.OUTPUT/'f2/1234',PRODUCER,DATA,'cuda',logger)


if __name__=='__main__':main()
