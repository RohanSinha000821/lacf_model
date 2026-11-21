"""Native P1 FP32 main three-expert MoE evaluation using verified target caches.

No encoders or training bank are loaded. Source parity precedes target access.
Preserve historical results, guarded trainers and the stopped SSL F3 queue.
"""
import gc
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from audio_deepfake_detection.metrics import read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, save_score_manifest
from scripts.evaluate_scores import evaluate_run
from scripts.lacf_moe import train_fp32 as training
from scripts.lacf_moe.evaluate import load_detector
from scripts.experiments.lacf_moe_ablations.evaluate import source_bank, verify_producer
from scripts.experiments.layer_fusion.evaluate_four_experts_cached import target_chunks as fold_chunks
from scripts.experiments.layer_fusion.evaluate_lacf_layer_moe import TARGET_FEATURES, target_chunks as f1_chunks
from scripts.token_fusion.train import check_guards, replace_json, save_json

ROOT, DATA, CACHE, FAMILY, NAME = training.ROOT, training.DATA, training.CACHE, training.FAMILY, training.NAME
OUTPUT = FAMILY / 'evaluations'
LAUNCH = OUTPUT / 'launch'
read = training.read


def verify_training(run, fold, data_root=DATA):
    labels = {d:canonical_score_labels(d, 'dev', data_root) for d in FOLDS[fold]['sources']}
    done = training.verify_completion(run, fold, expected_labels=labels)
    config = read(run / 'config.json')
    training.guards(config['provenance'])
    if (config['model_name'] != NAME or config['precision'] != training.PRECISION
            or Path(config['provenance']['cache']).resolve() != CACHE.resolve()):
        raise ValueError('FP32 main model/precision/cache identity differs')
    return config, done


def source_reload(model, run, config, fold, device, logger, retained):
    bank = source_bank(CACHE, fold, DATA, logger, retained)
    training.assert_fp32(model)
    scores, metrics = training.score_sources(model, bank, fold, device, logger)
    if metrics != read(run / 'metrics.json')['source_dev']:
        raise ValueError('Fresh FP32 source metrics differ; target held')
    checks = []
    for domain in FOLDS[fold]['sources']:
        data = bank[domain, 'dev']
        _, saved = read_score_csv(run / f'source_dev_{domain}.csv', domain,
            expected_labels=dict(zip(data.ids, data.labels.tolist())))
        order = sorted(range(len(data.ids)), key=data.ids.__getitem__)
        if not np.array_equal(scores[domain].numpy()[order], saved):
            raise ValueError('Fresh full FP32 source scores differ; target held')
        checks.append({'domain':domain, 'records':len(saved), 'score_bit_exact':True})
    return {'passed':True, 'full_source_score_bit_exact':True, 'checks':checks,
        'precision':training.PRECISION, 'encoders_loaded_or_executed':False, 'target_access':False,
        'calibration':'exact frozen FP32 source calibration verified before scoring'}


@torch.inference_mode()
def score_arrays(model, arrays, device, inputs='concat', *, batch_size=132, last_layer_only=False):
    if inputs != 'concat' or batch_size != 132 or last_layer_only:
        raise ValueError('Preserve full-layer three-expert MoE inputs and head batch132')
    training.assert_fp32(model)
    count = len(arrays['wavlm_layers'])
    if (arrays['wavlm_layers'].shape != (count,13,768) or arrays['clap_projected'].shape != (count,512)
            or count < 1 or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in arrays.values())):
        raise ValueError('Invalid FP32 cached features; no skipping or dtype fallback')
    values = []
    for start in range(0,count,batch_size):
        batch = {'wavlm':torch.from_numpy(arrays['wavlm_layers'][start:start+batch_size]).to(device),
                 'clap':torch.from_numpy(arrays['clap_projected'][start:start+batch_size]).to(device)}
        with torch.autocast(torch.device(device).type, enabled=False):
            scores = model(batch)['logit']
        if scores.dtype != torch.float32 or scores.shape != (len(batch['wavlm']),) or not torch.isfinite(scores).all():
            raise ValueError('Final scoring layer must compute finite FP32 scores')
        values.append(scores.cpu())
    return torch.cat(values).numpy()



def finish_evaluation(model, run, output, fold, name, family, config, done, validation, provenance,
                      device, logger, *, inputs='concat', batch_size=132, last_layer_only=False, bootstrap_resamples=1000):
    """Shared cached export for this fixed study and the original cached F4."""
    if output.exists():
        raise FileExistsError('Existing evaluation preserved; no overwrite/retry')
    if not validation.get('passed') or not validation.get('full_source_score_bit_exact'):
        raise ValueError('Full source verification required before target access')
    check_guards(provenance)
    validation.update(scorer_sha256=provenance['code_sha256'],script_sha256=checkpoint_digest(Path(__file__)))
    output.mkdir(parents=True, exist_ok=False)
    for filename in ('best.pt','config.json','training_complete.json','source_calibration.json'):
        shutil.copy2(run / filename, output / filename)
    shutil.copy2(run / 'metrics.json', output / 'source_metrics.json')
    save_json(output / 'source_validation.json', validation)
    producer, features = verify_producer(fold, provenance)
    manifest = load_score_manifest(output, fold, 1234, done['best_checkpoint_sha256'], create=True)
    for d in FOLDS[fold]['sources']:
        filename = f'source_dev_{d}.csv'; shutil.copy2(run / filename, output / filename)
        manifest['score_sha256'][filename] = checkpoint_digest(output / filename)
    inference = {'input':'centered common interval up to10s; native-unpadded frozen WavLM; CLAP repeatpad',
                 'head_batch':batch_size, 'precision':config['precision'], 'score':'scalar spoof logit',
                 'encoder_execution':False, 'feature_inputs':inputs, 'last_layer_only':last_layer_only}
    save_json(output / 'evaluation_config.json', {'model_name':name, 'fold':fold, 'seed':1234,
        'training_run':str(run.resolve()),'training_completion_sha256':checkpoint_digest(run / 'training_complete.json'),
        'target_feature_producer':str(producer.resolve()),'target_features':str(features.resolve()),
        'inference':inference,'provenance':provenance,'target_tuning':False,
        'exploratory_after_prior_target_exposure':True,'bootstrap_resamples':bootstrap_resamples,'bootstrap_seed':2026,
        'comparison_disclosure':config.get('comparison_disclosure','single-seed FP32 precision control; frozen extraction unchanged; historical target exposure')})
    chunks = f1_chunks(features, CACHE, DATA) if fold == 'f1' else fold_chunks(producer, CACHE, fold, DATA)
    ids, values = [], []
    for identifiers, arrays in chunks:
        check_guards(provenance)
        ids.extend(identifiers)
        values.extend(score_arrays(model, arrays, device, inputs, batch_size=batch_size,
                                   last_layer_only=last_layer_only).tolist())
        replace_json(output / 'progress.json',{'phase':'cached target scoring','scored':len(ids)})
        if len(ids) % 10240 == 0:
            logger.info('%s %s TARGET %d cached recordings',name,fold.upper(),len(ids))
    spec = FOLDS[fold]; expected = canonical_score_labels(spec['target'],spec['target_split'],DATA)
    if ids != list(expected) or len(values) != len(expected):
        raise ValueError('Incomplete/reordered target export')
    write_score_csv(output / 'target_scores.csv',zip(ids,itertools.repeat(spec['target']),expected.values(),values))
    manifest['score_sha256']['target_scores.csv'] = checkpoint_digest(output / 'target_scores.csv')
    manifest.update(model_family=family,inference=inference,training_run=str(run.resolve()),
        training_metrics_copy='source_metrics.json',source_validation_sha256=checkpoint_digest(output / 'source_validation.json'),
        target_features_sha256=checkpoint_digest(features / 'complete.json'),target_features_path=str(features.resolve()),
        scorer_sha256=provenance['code_sha256'],exploratory_after_prior_target_exposure=True)
    save_score_manifest(output,manifest)
    del model,arrays; gc.collect()
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    replace_json(output / 'progress.json',{'phase':'metrics and conditional bootstrap','scored':len(ids),'total':len(ids)})
    logger.info('%s %s METRICS | %d conditional bootstrap resamples',name,fold.upper(),bootstrap_resamples)
    report=evaluate_run(output,fold,seed=1234,source_data_root=DATA,target_data_root=DATA,bootstrap_resamples=bootstrap_resamples)
    check_guards(provenance)
    if read(output / 'source_calibration.json') != read(run / 'source_calibration.json'):
        raise ValueError('Original calibration changed')
    save_json(output / 'evaluation_complete.json',{'model_name':name,'fold':fold,'seed':1234,
        'checkpoint_sha256':done['best_checkpoint_sha256'],'metrics_sha256':checkpoint_digest(output / 'metrics.json'),
        'source_validation_sha256':manifest['source_validation_sha256'],'score_sha256':manifest['score_sha256'],
        'target_features_sha256':manifest['target_features_sha256'],
        'source_selected_checkpoint_only':True,'no_target_tuning':True,'exploratory_after_prior_target_exposure':True})
    replace_json(output / 'progress.json',{'phase':'complete','scored':len(ids),'total':len(ids)})
    logger.info('COMPLETE %s %s | EER %.4f%% AUROC %.6f',name,fold.upper(),100*report['target']['eer'],report['target']['auroc'])
    return report

def verify_evaluation(output, fold, training, data_root):
    """Bind reusable results to this selected checkpoint and canonical scores."""
    config, selected = verify_training(training, fold, data_root)
    done = read(output / 'evaluation_complete.json')
    manifest = read(output / 'score_manifest.json')
    if (done['fold'] != fold or done['seed'] != 1234
            or done['checkpoint_sha256'] != selected['best_checkpoint_sha256']
            or checkpoint_digest(output / 'best.pt') != selected['best_checkpoint_sha256']
            or checkpoint_digest(output / 'config.json') != checkpoint_digest(training / 'config.json')
            or checkpoint_digest(output / 'training_complete.json') != checkpoint_digest(training / 'training_complete.json')
            or manifest['checkpoint_sha256'] != selected['best_checkpoint_sha256']
            or manifest['training_completion_sha256'] != checkpoint_digest(training / 'training_complete.json')
            or done['metrics_sha256'] != checkpoint_digest(output / 'metrics.json')
            or done['source_validation_sha256'] != checkpoint_digest(output / 'source_validation.json')
            or done['score_sha256'] != manifest['score_sha256']
            or not done['no_target_tuning'] or not done['source_selected_checkpoint_only']):
        raise ValueError('Completed evaluation identity/bindings differ')
    validation = read(output / 'source_validation.json')
    if (not validation['passed'] or not all(x['score_bit_exact'] for x in validation['checks'])
            or {x['domain'] for x in validation['checks']} != set(FOLDS[fold]['sources'])):
        raise ValueError('Completed source validation differs')
    if checkpoint_digest(output / 'source_calibration.json') != selected['calibration_sha256']:
        raise ValueError('Evaluation source calibration changed')
    expected_files = {f'source_dev_{d}.csv' for d in FOLDS[fold]['sources']} | {'target_scores.csv'}
    if set(done['score_sha256']) != expected_files:
        raise ValueError('Completed source/target file set differs')
    for name, digest in done['score_sha256'].items():
        if checkpoint_digest(output / name) != digest:
            raise ValueError('Completed evaluation score changed')
        if name != 'target_scores.csv' and digest != selected['source_scores_sha256'][name]:
            raise ValueError('Evaluation source differs from selected training')
    spec = FOLDS[fold]
    read_score_csv(output / 'target_scores.csv', spec['target'],
                   expected_labels=canonical_score_labels(spec['target'], spec['target_split'], data_root))
    return done

def evaluate(run, output, fold, provenance, device, logger, retained, *, bootstrap_resamples=1000):
    if output.exists():
        raise FileExistsError('Existing evaluation preserved; no retry/overwrite')
    if (checkpoint_digest(run / 'training_complete.json') != provenance['training_completion_sha256'][fold]
            or checkpoint_digest(run / 'best.pt') != provenance['checkpoint_sha256'][fold]):
        raise ValueError('Launch-bound FP32 training changed')
    config, done = verify_training(run, fold)
    model = load_detector(run, device)
    validation = source_reload(model, run, config, fold, device, logger, retained)
    training.guards(config['provenance'])
    report = finish_evaluation(model, run, output, fold, NAME, 'lacf_moe_fp32', config, done,
        validation, provenance, device, logger, bootstrap_resamples=bootstrap_resamples)
    verify_evaluation(output, fold, run, DATA)
    return report


def verify_main_producer(provenance):
    launch = FAMILY / 'launch'
    for name, digest in provenance['training_producer_sha256'].items():
        if checkpoint_digest(launch / name) != digest:
            raise ValueError('Main FP32 producer evidence changed')
    status = read(launch / 'exit.json')
    complete = read(launch / 'complete.json')
    if (status.get('trainer_exit') != 0 or status.get('logger_exit') != 0 or status.get('verified_complete') is not True
            or complete.get('folds') != list(FOLDS) or complete.get('precision') != training.PRECISION
            or complete.get('completion_sha256') != provenance['training_completion_sha256']):
        raise ValueError('FP32 main producer was not fully verified successful')


def main():
    if (LAUNCH / 'started.json').exists() or any((OUTPUT / f / '1234').exists() for f in FOLDS):
        raise FileExistsError('Prior evaluation preserved; no retry/resume/overwrite')
    provenance = read(LAUNCH / 'provenance.json')
    check_guards(provenance)
    verify_main_producer(provenance)
    # CPU checkpoint/calibration gates for all four folds before new target reads.
    for fold in FOLDS:
        verify_training(FAMILY / fold / '1234', fold)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.nnpack.set_flags(False)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 2*1024**3:
        raise RuntimeError('Need2GiB CUDA free; preserve active jobs, no fallback')
    torch.cuda.set_per_process_memory_fraction(1024**3 / torch.cuda.get_device_properties(0).total_memory)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logger = logging.getLogger('lacf_moe_fp32_evaluation')
    save_json(LAUNCH / 'started.json', {'pid':os.getpid(), 'folds':list(FOLDS), 'precision':training.PRECISION,
        'encoders_loaded_or_executed':False, 'bootstrap_resamples':1000, 'bootstrap_seed':2026})
    retained, completed = {}, []
    for fold in FOLDS:
        check_guards(provenance)
        replace_json(LAUNCH / 'progress.json', {'fold':fold, 'completed_folds':len(completed), 'phase':'source reload verification'})
        report = evaluate(FAMILY / fold / '1234', OUTPUT / fold / '1234', fold, provenance, 'cuda', logger, retained)
        completed.append({'fold':fold, 'evaluation_completion_sha256':checkpoint_digest(OUTPUT / fold / '1234/evaluation_complete.json'),
            'target_eer':report['target']['eer'], 'target_auroc':report['target']['auroc']})
        replace_json(LAUNCH / 'queue_status.json', {'completed_folds':len(completed), 'folds':completed})
        gc.collect();torch.cuda.empty_cache()
    check_guards(provenance)
    save_json(LAUNCH / 'complete.json', {'model_name':NAME, 'folds':completed, 'precision':training.PRECISION,
        'all_four_complete':True, 'no_target_tuning':True, 'exploratory_after_prior_target_exposure':True})
    replace_json(LAUNCH / 'progress.json', {'phase':'all four FP32 evaluations complete', 'completed_folds':4})


if __name__ == '__main__':
    main()
