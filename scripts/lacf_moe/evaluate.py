"""Native P1 evaluation of the completed main three-expert LACF-MoE folds."""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.lacf_moe.model import LACFMoE
from audio_deepfake_detection.metrics import read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest,
    read_dataset, save_score_manifest,
)
from scripts.evaluate_scores import evaluate_run
# These existing helpers only check/reload source features and encoders;
# the detector below always instantiates the main THREE-expert model.
from scripts.experiments.layer_fusion.evaluate_four_experts import source_reload, source_encoder_check
from scripts.experiments.token_fusion.evaluate import load_encoders, score_features
from scripts.lacf.cache_diagnostics import encode_chunk, loader
from scripts.lacf_moe.train import DATA, FAMILY, NAME, ROOT, guards, read, verify_completion
from scripts.token_fusion.train import replace_json, save_json

OUTPUT = FAMILY / 'evaluations'
ORIGINAL_F1 = ROOT / 'outputs/experiments/layer_fusion/evaluations/lacf_layer_moe/f1/1234'
GPU_BUDGET_GIB = 7


def verify_training(run, fold, data_root):
    done = verify_completion(run, fold, data_root=data_root)
    config = read(run / 'config.json')
    if config['model_name'] not in (NAME, 'LACF Layer-MoE'):
        raise ValueError('Main three-expert model identity differs')
    guards(config['provenance'])
    return config, done


def load_detector(run, device):
    config = read(run / 'config.json')
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    model = LACFMoE(state['model_state_dict']['backend.prototypes'])
    model.load_state_dict(state['model_state_dict'], strict=True)
    if json.loads(json.dumps(model.configuration())) != config['model']:
        raise ValueError('Main three-expert architecture differs')
    return model.to(device).requires_grad_(False).eval()


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


def extract_chunks(encoders, collator, fold, data_root, output, execution_guards, logger):
    """First target read occurs here, after source gates. Retain reusable features."""
    spec = FOLDS[fold]
    domain, split = spec['target'], spec['target_split']
    records = read_dataset(domain, split, data_root)
    expected = canonical_score_labels(domain, split, data_root, records=records)
    canonical = list(expected.items())
    stream = iter(loader(records, [f'{domain}/{split}/{x}' for x in expected], collator, training=False, workers=8))
    features = output / 'target_features'
    features.mkdir(exist_ok=False)
    descriptors, offset, started = [], 0, time.monotonic()
    while prepared := list(itertools.islice(stream, 1024)):
        guards(execution_guards)
        if [r['index'] for r in prepared] != list(range(offset, offset+len(prepared))):
            raise ValueError('Target ordering differs')
        if any(r['id'] != f'{domain}/{split}/{canonical[offset+i][0]}' or r['label'] != canonical[offset+i][1]
               for i, r in enumerate(prepared)):
            raise ValueError('Canonical target identity differs')
        arrays = encode_chunk(encoders, prepared, 32, 64)
        for key, shape in (('wavlm_layers', (len(prepared), 13, 768)), ('clap_projected', (len(prepared), 512))):
            if arrays[key].shape != shape or arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
                raise ValueError('Invalid target feature; no recordings may be skipped')
        chunk = features / f'chunk_{len(descriptors):05d}'
        chunk.mkdir(exist_ok=False)
        for key, values in arrays.items():
            with (chunk / f'{key}.npy').open('xb') as stream_file:
                np.save(stream_file, values, allow_pickle=False)
        save_json(chunk / 'records.json', [{k:v for k,v in row.items() if k != 'views'} for row in prepared])
        save_json(chunk / 'complete.json', {'records': len(prepared), 'files_sha256': {
            name: checkpoint_digest(chunk / name) for name in ('records.json', 'wavlm_layers.npy', 'clap_projected.npy')}})
        descriptors.append({'path': chunk.name, 'completion_sha256': checkpoint_digest(chunk / 'complete.json')})
        identifiers = [x for x, _ in canonical[offset:offset+len(prepared)]]
        offset += len(prepared)
        seconds = time.monotonic()-started
        replace_json(output / 'progress.json', {'phase': 'target frozen features and three-expert LACF-MoE scoring',
                     'scored': offset, 'total': len(records), 'examples_per_second': offset/seconds,
                     'eta_hours': (len(records)-offset)/(offset/seconds)/3600})
        logger.info('%s TARGET %d/%d | %.1f audio/s | ETA %.2fh', fold.upper(), offset, len(records),
                    offset/seconds, (len(records)-offset)/(offset/seconds)/3600)
        yield identifiers, arrays
    if offset != len(expected):
        raise ValueError('Incomplete target extraction')
    save_json(features / 'complete.json', {'records': offset, 'chunks': descriptors,
              'canonical_ids_labels_sha256': hashlib.sha256(json.dumps(canonical).encode()).hexdigest(),
              'evaluation_config_sha256': checkpoint_digest(output / 'evaluation_config.json')})

def evaluate(training, output, fold, data_root, device, logger, *, bootstrap_resamples=1000):
    if output.exists():
        raise FileExistsError('Existing evaluation preserved; no overwrite/retry')
    if output.name != '1234' or output.parent.name != fold:
        raise ValueError('Canonical fold/1234 output required')
    config, done = verify_training(training, fold, data_root)
    model = load_detector(training, device)
    validation = source_reload(model, training, config, data_root, logger, device)
    guards(config['provenance'])
    output.mkdir(parents=True, exist_ok=False)
    for name in ('best.pt', 'config.json', 'training_complete.json', 'source_calibration.json'):
        shutil.copy2(training / name, output / name)
    shutil.copy2(training / 'metrics.json', output / 'source_metrics.json')
    save_json(output / 'source_validation.json', validation)
    manifest = load_score_manifest(output, fold, 1234, done['best_checkpoint_sha256'], create=True)
    for domain in FOLDS[fold]['sources']:
        name = f'source_dev_{domain}.csv'
        shutil.copy2(training / name, output / name)
        manifest['score_sha256'][name] = checkpoint_digest(output / name)
    execution = read(OUTPUT / 'launch/provenance.json')
    execution_guards = {'code_sha256': execution['code_sha256']}
    guards(execution_guards)
    cache = Path(config['provenance']['cache'])
    inference = {'input': 'centered common interval up to10s; native unpadded WavLM; CLAP repeatpad',
                 'head_batch': 132, 'wavlm_batch': 32, 'clap_batch': 64, 'chunk': 1024,
                 'workers': 8, 'prefetch': 2, 'precision': config['precision'], 'score': 'scalar spoof logit'}
    save_json(output / 'evaluation_config.json', {'model_name': NAME, 'fold': fold, 'seed': 1234,
        'target': FOLDS[fold]['target']+'/'+FOLDS[fold]['target_split'], 'inference': inference,
        'training_run': str(training.resolve()), 'training_completion_sha256': checkpoint_digest(training / 'training_complete.json'),
        'provenance': execution_guards, 'source_cache': str(cache.resolve()),
        'frozen_encoder_revisions': read(cache / 'execution_accelerated.json')['pretrained'],
        'exploratory_after_prior_target_exposure': True, 'target_tuning': False,
        'bootstrap_resamples': bootstrap_resamples, 'bootstrap_seed': 2026,
        'target_copy_status': 'source SHA256 verified; destination presence/sizes checked; destination full checksums owner-waived'})
    manifest.update(model_family='lacf_moe', inference=inference, training_run=str(training.resolve()),
                    source_validation_sha256=checkpoint_digest(output / 'source_validation.json'),
                    scorer_sha256=execution_guards['code_sha256'], exploratory_after_prior_target_exposure=True)
    encoders, collator = load_encoders(cache)
    source_domain = 'asv5' if 'asv5' in FOLDS[fold]['sources'] else 'speechfake'
    source_encoder_check(encoders, collator, cache, source_domain, data_root, output, logger)
    guards(execution_guards)
    features = output / 'target_features'
    ids, values = [], []
    with torch.inference_mode():
        for identifiers, arrays in extract_chunks(encoders, collator, fold, data_root, output, execution_guards, logger):
            ids.extend(identifiers)
            values.extend(score_features(model, arrays, device).tolist())
    spec = FOLDS[fold]
    expected = canonical_score_labels(spec['target'], spec['target_split'], data_root)
    if ids != list(expected) or len(values) != len(expected):
        raise ValueError('Incomplete/reordered target score export')
    target = output / 'target_scores.csv'
    write_score_csv(target, zip(ids, itertools.repeat(spec['target']), expected.values(), values))
    manifest['target_features_sha256'] = checkpoint_digest(features / 'complete.json')
    manifest['target_features_path'] = str(features.resolve())
    manifest['score_sha256'][target.name] = checkpoint_digest(target)
    save_score_manifest(output, manifest)
    del model, encoders, collator, arrays
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    replace_json(output / 'progress.json', {'phase': 'metrics and conditional bootstrap', 'scored': len(ids), 'total': len(ids)})
    logger.info('%s METRICS: raw and transductive; %d bootstrap resamples', fold.upper(), bootstrap_resamples)
    report = evaluate_run(output, fold, seed=1234, source_data_root=data_root,
                          target_data_root=data_root, bootstrap_resamples=bootstrap_resamples)
    guards(execution_guards)
    save_json(output / 'evaluation_complete.json', {'model_name': NAME, 'fold': fold, 'seed': 1234,
        'checkpoint_sha256': done['best_checkpoint_sha256'], 'metrics_sha256': checkpoint_digest(output / 'metrics.json'),
        'source_validation_sha256': manifest['source_validation_sha256'], 'score_sha256': manifest['score_sha256'],
        'target_features_sha256': manifest['target_features_sha256'], 'no_target_tuning': True,
        'source_selected_checkpoint_only': True, 'exploratory_after_prior_target_exposure': True})
    verify_evaluation(output, fold, training, data_root)
    replace_json(output / 'progress.json', {'phase': 'complete', 'scored': len(ids), 'total': len(ids)})
    logger.info('COMPLETE %s %s | target EER %.4f%% AUROC %.6f', NAME, fold.upper(), report['target']['eer']*100, report['target']['auroc'])
    return report


def run_queue(data_root, device, logger):
    launch = OUTPUT / 'launch'
    provenance = read(launch / 'provenance.json')
    guards(provenance)
    # Verify every trained source checkpoint before any new target is opened.
    for fold in FOLDS:
        verify_training(FAMILY / fold / '1234', fold, data_root)
    f1 = verify_evaluation(ORIGINAL_F1, 'f1', FAMILY / 'f1/1234', data_root)
    link = OUTPUT / 'f1/1234'
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(ORIGINAL_F1, link.parent), target_is_directory=True)
    save_json(launch / 'f1_reuse_verified.json', {'evaluation_complete_sha256': checkpoint_digest(ORIGINAL_F1 / 'evaluation_complete.json'),
              'checkpoint_sha256': f1['checkpoint_sha256'], 'original': str(ORIGINAL_F1), 'rerun': False})
    logger.info('F1 completed evaluation verified and reused; original artifacts unchanged')
    for fold in ('f2', 'f3', 'f4'):
        guards(provenance)
        evaluate(FAMILY / fold / '1234', OUTPUT / fold / '1234', fold, data_root, device, logger)
    save_json(OUTPUT / 'summary.json', {'model_name': NAME, 'seed': 1234, 'folds': {
        fold: {'evaluation': str(OUTPUT / fold / '1234'),
               'metrics_sha256': checkpoint_digest(OUTPUT / fold / '1234/metrics.json')}
        for fold in FOLDS}, 'exploratory_after_prior_target_exposure': True})
    save_json(launch / 'complete.json', {'folds': list(FOLDS), 'preserved_f1': True,
        'evaluation_completion_sha256': {f: checkpoint_digest(OUTPUT / f / '1234/evaluation_complete.json') for f in FOLDS}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA)
    args = parser.parse_args()
    launch = OUTPUT / 'launch'
    if (launch / 'started.json').exists() or any((OUTPUT / f / '1234').exists() for f in FOLDS):
        raise FileExistsError('Prior evaluation/launch preserved; no retry or overwrite')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(GPU_BUDGET_GIB*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    save_json(launch / 'started.json', {'phase': 'verify all source completions; reuse F1; then F2-F4', 'seed': 1234})
    run_queue(args.data_root, 'cuda', logging.getLogger('lacf_moe_evaluation'))


if __name__ == '__main__':
    main()
