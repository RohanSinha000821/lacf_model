"""Evaluate frozen TF2 F2–F4 using completed, verified frozen target features."""

import gc
import itertools
import json
import logging
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from audio_deepfake_detection.lacf.probes import FeatureBatches, load_split, score_probe
from audio_deepfake_detection.metrics import read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, save_score_manifest,
)
from scripts.evaluate_scores import evaluate_run
from scripts.experiments.layer_fusion.evaluate_four_experts_cached import target_chunks
from scripts.token_fusion.evaluate import load_detector, score_features
from scripts.token_fusion import train_all_folds as training
from scripts.token_fusion.train import ROOT, check_guards, replace_json, save_json


NAME = 'Frozen WavLM–CLAP Transformer fusion'
FOLDS_TO_EVALUATE = ('f2', 'f3', 'f4')
OUTPUT = ROOT / 'outputs/experiments/token_fusion/evaluations/tf2_frozen'
LAUNCH = OUTPUT / 'launch'
PRODUCERS = ROOT / 'outputs/lacf_moe/evaluations'
DATA = Path('/mnt/drive/audio-deepfake-cache')


def read(path):
    return json.loads(path.read_text())


def verify_training(run, fold, data_root):
    """Check canonical source membership and unchanged calibration before targets."""
    if fold not in FOLDS_TO_EVALUATE:
        raise ValueError('Only the authorized frozen TF2 F2–F4 seed1234 runs')
    config = read(run / 'config.json')
    training.guards(config['provenance'])
    if (config['fold'] != fold or config['seed'] != 1234 or config['seed_plan'] != [1234]
            or config['excluded_target'] != FOLDS[fold]['target']
            or config['precision'] != 'BF16 mixed; sensitive FP32'
            or config['max_epochs'] != 30 or config['patience'] != 5):
        raise ValueError('Frozen TF2 fold/seed/precision/recipe changed')
    bank = {}
    for domain in FOLDS[fold]['sources']:
        labels = canonical_score_labels(domain, 'dev', data_root)
        bank[domain, 'dev'] = SimpleNamespace(ids=list(labels), labels=torch.tensor(list(labels.values())))
    done = training.verify_run(run, fold, bank)
    return config, done


def source_reload(model, run, config, data_root, device, logger):
    """Load one source-dev partition at a time; retain original head batch132."""
    cache = training.CACHE
    marker, checks = read(cache / 'complete.json'), []
    selected = read(run / 'metrics.json')['source_dev']['by_domain']
    for domain in FOLDS[config['fold']]['sources']:
        dataset = load_split(cache, domain, 'dev', marker['splits'][f'{domain}/dev'], data_root, logger=logger)
        scores, metrics = score_probe(model, FeatureBatches(
            [dataset], 'concat', device, training=False, seed=1234))
        _, exported = read_score_csv(run / f'source_dev_{domain}.csv', domain,
                                    expected_labels=dict(zip(dataset.ids, dataset.labels.tolist())))
        order = sorted(range(len(dataset.ids)), key=dataset.ids.__getitem__)
        if metrics != selected[domain] or not np.array_equal(scores.numpy()[order], exported):
            raise ValueError('Fresh full frozen TF2 source reload differs; target held')
        checks.append({'domain': domain, 'records': len(scores), 'score_bit_exact': True, **metrics})
        logger.info('SOURCE %s PASS | %d scores bit-exact | EER %.4f%%', domain, len(scores), metrics['eer'] * 100)
        del dataset, scores
        gc.collect()
    return {'passed': True, 'full_source_score_bit_exact': True, 'checks': checks,
            'calibration': 'original source thresholds and statistics unchanged',
            'scope': 'fresh selected checkpoint; all canonical centered source-development features',
            'encoders_loaded_or_executed': False, 'target_access': False}


def verify_producer(producer, fold):
    """A successful producer and its exact payload identities are mandatory."""
    done = read(producer / 'evaluation_complete.json')
    if done['fold'] != fold or done['seed'] != 1234 or done['source_selected_checkpoint_only'] is not True:
        raise ValueError('Target-cache producer completion differs')
    for name, key in (('metrics.json', 'metrics_sha256'), ('best.pt', 'checkpoint_sha256'),
                      ('source_validation.json', 'source_validation_sha256'),
                      ('target_features/complete.json', 'target_features_sha256')):
        if checkpoint_digest(producer / name) != done[key]:
            raise ValueError('Completed target-cache producer artifact changed: ' + name)
    if read(producer / 'source_validation.json')['passed'] is not True:
        raise ValueError('Target-cache producer source validation failed')
    for name, digest in done['score_sha256'].items():
        if checkpoint_digest(producer / name) != digest:
            raise ValueError('Target-cache producer score export changed')
    return checkpoint_digest(producer / 'evaluation_complete.json')


def evaluate(run, output, producer, fold, provenance, data_root, device, logger, *, bootstrap_resamples=1000):
    if output.exists():
        raise FileExistsError('Existing evaluation preserved; no overwrite/retry')
    check_guards(provenance)
    config, done = verify_training(run, fold, data_root)
    if (done['best_checkpoint_sha256'] != provenance['checkpoint_sha256'][fold]
            or checkpoint_digest(run / 'training_complete.json') != provenance['training_completion_sha256'][fold]):
        raise ValueError('Launch-bound training checkpoint/completion changed')
    model = load_detector(run, device)
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    validation = source_reload(model, run, config, data_root, device, logger)
    validation['scorer_sha256'] = provenance['code_sha256']
    validation['script_sha256'] = checkpoint_digest(Path(__file__))
    if torch.device(device).type == 'cuda':
        validation['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        validation['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    check_guards(provenance)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('best.pt', 'config.json', 'training_complete.json', 'source_calibration.json'):
        shutil.copy2(run / name, output / name)
    shutil.copy2(run / 'metrics.json', output / 'source_metrics.json')
    save_json(output / 'source_validation.json', validation)
    # Only now inspect any target producer, canonical target labels or features.
    producer_digest = verify_producer(producer, fold)
    if producer_digest != provenance['producer_completion_sha256'][fold]:
        raise ValueError('Launch-bound target-cache producer changed')
    save_json(output / 'evaluation_config.json', {
        'model_name': NAME, 'condition': 'TF2', 'fold': fold, 'seed': 1234,
        'training_run': str(run.resolve()), 'target_feature_producer': str(producer.resolve()),
        'producer_completion_sha256': producer_digest, 'head_batch': 132,
        'precision': config['precision'], 'encoder_execution': False,
        'source_calibration': 'unchanged source-selected model and calibration',
        'bootstrap_resamples': bootstrap_resamples, 'bootstrap_seed': 2026,
        'provenance': provenance, 'exploratory_after_prior_target_exposure': True, 'target_tuning': False})
    manifest = load_score_manifest(output, fold, 1234, done['best_checkpoint_sha256'], create=True)
    for domain in FOLDS[fold]['sources']:
        name = f'source_dev_{domain}.csv'
        shutil.copy2(run / name, output / name)
        manifest['score_sha256'][name] = checkpoint_digest(output / name)
    ids, values = [], []
    for identifiers, arrays in target_chunks(producer, training.CACHE, fold, data_root):
        check_guards(provenance)
        ids.extend(identifiers)
        values.extend(score_features(model, arrays, device).tolist())
        replace_json(output / 'progress.json', {'phase': 'frozen TF2 cached target scoring', 'scored': len(ids)})
        if len(ids) % (10 * 1024) == 0:
            logger.info('%s TARGET %d cached recordings scored', fold.upper(), len(ids))
    spec = FOLDS[fold]
    expected = canonical_score_labels(spec['target'], spec['target_split'], data_root)
    if ids != list(expected) or len(values) != len(expected):
        raise ValueError('Incomplete/reordered frozen TF2 target scores')
    write_score_csv(output / 'target_scores.csv', zip(ids, itertools.repeat(spec['target']), expected.values(), values))
    manifest['score_sha256']['target_scores.csv'] = checkpoint_digest(output / 'target_scores.csv')
    manifest.update(model_family='token_fusion', condition='TF2', training_run=str(run.resolve()),
        training_metrics_copy='source_metrics.json',
        completion_record_scope='original training snapshot; metrics_sha256 binds source_metrics.json',
        inference={'input': 'centered common physical interval up to10s; native unpadded frozen features',
                   'precision': config['precision'], 'head_batch': 132,
                   'score': 'scalar spoof logit', 'encoder_execution': False},
        source_validation_sha256=checkpoint_digest(output / 'source_validation.json'),
        target_features_sha256=checkpoint_digest(producer / 'target_features/complete.json'),
        target_features_path=str((producer / 'target_features').resolve()),
        producer_completion_sha256=producer_digest, scorer_sha256=provenance['code_sha256'],
        exploratory_after_prior_target_exposure=True)
    save_score_manifest(output, manifest)
    del model, arrays
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    replace_json(output / 'progress.json', {'phase': 'metrics and conditional bootstrap', 'scored': len(ids), 'total': len(ids)})
    logger.info('%s TARGET complete: %d recordings; shared metrics and1000 conditional bootstraps', fold.upper(), len(ids))
    report = evaluate_run(output, fold, seed=1234, source_data_root=data_root,
                          target_data_root=data_root, bootstrap_resamples=bootstrap_resamples)
    check_guards(provenance)
    training.guards(config['provenance'])
    save_json(output / 'evaluation_complete.json', {
        'model_name': NAME, 'fold': fold, 'seed': 1234, 'checkpoint_sha256': done['best_checkpoint_sha256'],
        'metrics_sha256': checkpoint_digest(output / 'metrics.json'),
        'source_validation_sha256': manifest['source_validation_sha256'], 'score_sha256': manifest['score_sha256'],
        'target_features_sha256': manifest['target_features_sha256'], 'producer_completion_sha256': producer_digest,
        'source_selected_checkpoint_only': True, 'no_target_tuning': True,
        'exploratory_after_prior_target_exposure': True})
    replace_json(output / 'progress.json', {'phase': 'complete', 'scored': len(ids), 'total': len(ids)})
    logger.info('COMPLETE %s frozen TF2 | EER %.4f%% AUROC %.6f', fold.upper(), report['target']['eer'] * 100, report['target']['auroc'])
    return report


def main():
    if (LAUNCH / 'started.json').exists() or any((OUTPUT / f / '1234').exists() for f in FOLDS_TO_EVALUATE):
        raise FileExistsError('Prior evaluation preserved; no automatic retry/resume/overwrite')
    provenance = read(LAUNCH / 'provenance.json')
    check_guards(provenance)
    producer = read(training.LAUNCH / 'exit.json')
    if producer.get('verified_complete') is not True or producer.get('trainer_exit') != 0:
        raise ValueError('Frozen TF2 training queue did not complete successfully')
    for fold in FOLDS_TO_EVALUATE:
        run = training.OUTPUT / fold / '1234'
        config, done = verify_training(run, fold, DATA)
        if (checkpoint_digest(run / 'training_complete.json') != provenance['training_completion_sha256'][fold]
                or done['best_checkpoint_sha256'] != provenance['checkpoint_sha256'][fold]):
            raise ValueError('Launch-bound frozen TF2 checkpoint/completion changed')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required; no precision fallback')
    free = int(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', '0'], text=True).strip())
    if free < 2048:
        raise RuntimeError('Insufficient2GiB GPU headroom; active jobs preserved')
    torch.cuda.set_per_process_memory_fraction(1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.nnpack.set_flags(False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('tf2_frozen_evaluation')
    save_json(LAUNCH / 'started.json', {'folds': list(FOLDS_TO_EVALUATE), 'seed': 1234,
                                      'encoder_execution': False, 'gpu_free_mib': free})
    for fold in FOLDS_TO_EVALUATE:
        replace_json(LAUNCH / 'progress.json', {'phase': 'source verification then cached target evaluation', 'fold': fold})
        evaluate(training.OUTPUT / fold / '1234', OUTPUT / fold / '1234',
                 PRODUCERS / fold / '1234', fold, provenance, DATA, 'cuda', logger)
    save_json(LAUNCH / 'complete.json', {'folds': list(FOLDS_TO_EVALUATE), 'seed': 1234,
              'evaluation_completion_sha256': {f: checkpoint_digest(OUTPUT / f / '1234/evaluation_complete.json')
                                             for f in FOLDS_TO_EVALUATE}})
    replace_json(LAUNCH / 'progress.json', {'phase': 'all three evaluations complete'})


if __name__ == '__main__':
    main()
