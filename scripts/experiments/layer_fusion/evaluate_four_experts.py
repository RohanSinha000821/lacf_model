"""Native P1 four-expert LACF evaluation, gated by completed source verification."""

import argparse
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

from audio_deepfake_detection.experiments.layer_fusion.four_experts import FourExpertDetector
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_split, score_probe
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, read_dataset,
    save_score_manifest, verify_score_digest,
)
from scripts.evaluate_scores import evaluate_run
from scripts.experiments.layer_fusion.evaluate_lacf_layer_moe import TARGET_FEATURES, target_chunks
from scripts.experiments.layer_fusion.train_four_experts import FAMILY, verify_completion
from scripts.experiments.token_fusion.evaluate import load_encoders, score_features
from scripts.lacf.cache_diagnostics import encode_chunk, loader
from scripts.token_fusion.train import ROOT, check_guards, replace_json, save_json

NAME = 'LACF Layer-MoE (4 experts)'
OUTPUT = ROOT / 'outputs/experiments/layer_fusion/evaluations/lacf_layer_moe4'
TRAINER = 'scripts/experiments/layer_fusion/train_four_experts.py'


def read(path):
    return json.loads(path.read_text())


def training_guards(config, family):
    """Reconcile recorded trainer revisions; never relax model/scorer hashes."""
    provenance = config['provenance']
    current = dict(provenance['code_sha256'])
    reconciliation = None
    if checkpoint_digest(ROOT / TRAINER) != current[TRAINER]:
        archived = current[TRAINER]
        steps = []
        if config['fold'] == 'f1':
            snapshot = family / 'launch/f1_source_snapshot'
            extension = read(family / 'launch_f2_f3_f4/provenance.json')['trainer_extension']
            if (checkpoint_digest(snapshot / TRAINER) != archived
                    or extension['previous_sha256'] != archived):
                raise ValueError('F1 archived trainer/extension differs')
            archived = extension['current_sha256']
            steps.append('completed F1 trainer parameterized for F2-F4')
        actual = checkpoint_digest(ROOT / TRAINER)
        if archived != actual:
            folder = family / 'launch/calibration_verification_fix'
            if not (folder / 'record.json').is_file():
                raise ValueError('Unreviewed training source change')
            fix = read(folder / 'record.json')
            if (fix.get('scope') != 'source CSV ID ordering only; training and scorer unchanged'
                    or fix['previous_sha256'] != archived or fix['current_sha256'] != actual
                    or checkpoint_digest(folder / 'snapshot' / TRAINER) != archived):
                raise ValueError('Calibration-verifier trainer reconciliation differs')
            steps.append('source CSV ID ordering made identical to saved calibration; exact equality retained')
        reconciliation = {'path': TRAINER, 'archived_sha256': current[TRAINER],
                          'current_sha256': actual, 'reason': '; '.join(steps)}
        current[TRAINER] = actual
    guards = {**provenance, 'code_sha256': current}
    check_guards(guards)
    return guards, reconciliation


def verify_training(run, fold, data_root, family):
    # This entire function reads sources only.
    done = verify_completion(run, fold)
    config = read(run / 'config.json')
    if (config['model_name'] != NAME or config['fold'] != fold or config['seed'] != 1234
            or config['seed_plan'] != [1234] or config['target_access'] is not False):
        raise ValueError('Four-expert run identity differs')
    guards, reconciliation = training_guards(config, family)
    manifest = read(run / 'score_manifest.json')
    sources = FOLDS[fold]['sources']
    if (manifest['fold'] != fold or manifest['seed'] != 1234
            or manifest['checkpoint_sha256'] != done['best_checkpoint_sha256']
            or manifest['training_completion_sha256'] != checkpoint_digest(run / 'training_complete.json')
            or set(manifest['score_sha256']) != {f'source_dev_{d}.csv' for d in sources}):
        raise ValueError('Source manifest identity differs')
    scores = {}
    for domain in sources:
        path = run / f'source_dev_{domain}.csv'
        verify_score_digest(path, manifest)
        scores[domain] = read_score_csv(path, domain, expected_labels=canonical_score_labels(domain, 'dev', data_root))
    if calibrate_source_scores(scores) != read(run / 'source_calibration.json'):
        raise ValueError('Canonical frozen source calibration differs')
    return config, done, guards, reconciliation


def load_detector(run, device):
    config, done = read(run / 'config.json'), read(run / 'training_complete.json')
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if (state['config_sha256'] != checkpoint_digest(run / 'config.json')
            or state['epoch'] != done['best_epoch']
            or state['source_dev'] != read(run / 'metrics.json')['source_dev']):
        raise ValueError('Selected checkpoint/config/epoch differs')
    model = FourExpertDetector(state['model_state_dict']['backend.prototypes'])
    model.load_state_dict(state['model_state_dict'], strict=True)
    if json.loads(json.dumps(model.configuration())) != config['model']:
        raise ValueError('Four-expert architecture differs')
    return model.to(device).requires_grad_(False).eval()


def source_reload(model, run, config, data_root, logger, device):
    cache = Path(config['provenance']['cache'])
    for name, key in (('config.json', 'cache_config_sha256'), ('complete.json', 'cache_completion_sha256'),
                      ('execution_accelerated.json', 'cache_execution_sha256')):
        if checkpoint_digest(cache / name) != config['provenance'][key]:
            raise ValueError('Frozen source feature-cache identity differs')
    marker, checks = read(cache / 'complete.json'), []
    for domain in FOLDS[config['fold']]['sources']:
        dataset = load_split(cache, domain, 'dev', marker['splits'][f'{domain}/dev'], data_root, logger=logger)
        scores, metrics = score_probe(model, FeatureBatches([dataset], 'concat', device, training=False, seed=1234))
        order = sorted(range(len(dataset.ids)), key=dataset.ids.__getitem__)
        _, exported = read_score_csv(run / f'source_dev_{domain}.csv', domain,
                                    expected_labels=dict(zip(dataset.ids, dataset.labels.tolist())))
        if not np.array_equal(scores.numpy()[order], exported):
            raise ValueError('Fresh full-source reload differs; target held')
        checks.append({'domain': domain, 'records': len(scores), 'score_bit_exact': True, **metrics})
        logger.info('SOURCE RELOAD %s passed: %d scores bit-exact', domain, len(scores))
        del dataset, scores
        gc.collect()
    return {'passed': True, 'target_access': False, 'checks': checks,
            'scope': 'all canonical source development scores; fresh selected checkpoint; frozen calibration'}


def source_encoder_check(encoders, collator, cache, domain, data_root, output, logger):
    """Same batch32/64 extraction as preserved cache; source-only 10s stress."""
    directory = cache / domain / 'dev/chunk_00000'
    metadata = read(directory / 'records.json')
    candidates = [r for r in metadata if r['frames'] >= 10*r['rate']]
    if not candidates:
        raise ValueError('No source 10s record for the maximum-input check')
    selected = [candidates[i % len(candidates)] for i in range(32)]
    canonical = read_dataset(domain, 'dev', data_root)
    prepared = list(loader([canonical[r['index']] for r in selected],
                           [f'source-stress/{i}' for i in range(32)], collator, training=False, workers=4))
    if any(r['views'][0]['wavlm']['input_values'].shape[-1] != 160000 for r in prepared):
        raise ValueError('Source stress duration differs')
    torch.cuda.reset_peak_memory_stats()
    tick = time.monotonic()
    arrays = encode_chunk(encoders, prepared, 32, 64)
    drift = {}
    for key, values in arrays.items():
        original = np.load(directory / f'{key}.npy', mmap_mode='r', allow_pickle=False)[
            [r['view_rows'][0] for r in selected]].copy()
        a, b = values.reshape(32, -1).astype(np.float64), original.reshape(32, -1).astype(np.float64)
        cosine = (a*b).sum(1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
        if not np.isfinite(values).all() or not np.isfinite(cosine).all() or cosine.min() < .99:
            raise ValueError('Source encoder drift exceeds preserved cosine0.99 guard; target held')
        drift[key] = {'minimum_cosine': float(cosine.min()), 'bit_exact': bool(np.array_equal(a, b)),
                      'max_absolute_difference': float(np.max(np.abs(a-b)))}
    report = {'passed': True, 'target_access': False, 'domain': domain, 'records': 32,
              'seconds': time.monotonic()-tick, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'wavlm_batch': 32, 'clap_batch': 64, 'source_feature_drift': drift,
              'policy': 'same native-unpadded BF16 cache extraction; batching rounding disclosed'}
    save_json(output / 'source_encoder_check.json', report)
    logger.info('SOURCE ENCODERS passed: %.1fs, %.2f GiB peak', report['seconds'], report['peak_allocated_bytes']/1024**3)


def extract_chunks(encoders, collator, fold, data_root, output, guards, logger):
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
        check_guards(guards)
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
        replace_json(output / 'progress.json', {'phase': 'target frozen features and four-expert scoring',
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


def evaluate(training, output, fold, data_root, device, logger, *, family=FAMILY, bootstrap_resamples=1000):
    if output.exists():
        raise FileExistsError('Existing evaluation preserved; no overwrite/retry')
    if output.name != '1234' or output.parent.name != fold:
        raise ValueError('Canonical fold/1234 output required')
    config, done, guards, reconciliation = verify_training(training, fold, data_root, family)
    model = load_detector(training, device)
    validation = source_reload(model, training, config, data_root, logger, device)
    check_guards(guards)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('best.pt', 'config.json', 'training_complete.json', 'source_calibration.json'):
        shutil.copy2(training / name, output / name)
    shutil.copy2(training / 'metrics.json', output / 'source_metrics.json')
    validation['trainer_provenance_reconciliation'] = reconciliation
    save_json(output / 'source_validation.json', validation)
    manifest = load_score_manifest(output, fold, 1234, done['best_checkpoint_sha256'], create=True)
    for domain in FOLDS[fold]['sources']:
        name = f'source_dev_{domain}.csv'
        shutil.copy2(training / name, output / name)
        manifest['score_sha256'][name] = checkpoint_digest(output / name)
    # Guard the actual scorer/extractor as well as all guarded training dependencies.
    for path in (Path(__file__), ROOT / 'scripts/experiments/token_fusion/evaluate.py',
                 ROOT / 'scripts/experiments/layer_fusion/evaluate_lacf_layer_moe.py',
                 ROOT / 'scripts/lacf/cache_diagnostics.py', ROOT / 'scripts/evaluate_scores.py'):
        guards['code_sha256'][str(path.resolve().relative_to(ROOT))] = checkpoint_digest(path)
    cache = Path(config['provenance']['cache'])
    inference = {'input': 'centered common physical interval up to10s; native unpadded WavLM; CLAP repeatpad',
                 'head_batch': 132, 'wavlm_batch': 32, 'clap_batch': 64, 'chunk': 1024,
                 'precision': config['precision'], 'score': 'scalar spoof logit'}
    save_json(output / 'evaluation_config.json', {'model_name': NAME, 'fold': fold, 'seed': 1234,
        'target': FOLDS[fold]['target']+'/'+FOLDS[fold]['target_split'], 'inference': inference,
        'training_run': str(training.resolve()), 'training_completion_sha256': checkpoint_digest(training / 'training_complete.json'),
        'provenance': guards, 'frozen_encoder_revisions': read(cache / 'execution_accelerated.json')['pretrained'],
        'exploratory_after_prior_target_exposure': True, 'target_tuning': False,
        'bootstrap_resamples': bootstrap_resamples, 'bootstrap_seed': 2026,
        'target_copy_status': 'source SHA256 verified; destination presence/sizes checked; destination full checksums owner-waived'})
    manifest.update(model_family='lacf_layer_moe4', inference=inference, training_run=str(training.resolve()),
                    source_validation_sha256=checkpoint_digest(output / 'source_validation.json'),
                    scorer_sha256=guards['code_sha256'], exploratory_after_prior_target_exposure=True)
    encoders = collator = None
    if fold == 'f1':
        features = TARGET_FEATURES
        chunks = target_chunks(features, cache, data_root)
    else:
        encoders, collator = load_encoders(cache)
        # Choose a source only; never use the fold's held-out domain for this gate.
        domain = 'asv5' if 'asv5' in FOLDS[fold]['sources'] else 'speechfake'
        source_encoder_check(encoders, collator, cache, domain, data_root, output, logger)
        check_guards(guards)
        features = output / 'target_features'
        chunks = extract_chunks(encoders, collator, fold, data_root, output, guards, logger)
    ids, values = [], []
    with torch.inference_mode():
        for identifiers, arrays in chunks:
            check_guards(guards)
            ids.extend(identifiers)
            values.extend(score_features(model, arrays, device).tolist())
            if fold == 'f1' and len(ids) % 10240 == 0:
                logger.info('F1 reused target features: %d scored', len(ids))
    expected = canonical_score_labels(FOLDS[fold]['target'], FOLDS[fold]['target_split'], data_root)
    if ids != list(expected) or len(values) != len(expected):
        raise ValueError('Incomplete/reordered target score export')
    target = output / 'target_scores.csv'
    write_score_csv(target, zip(ids, itertools.repeat(FOLDS[fold]['target']), expected.values(), values))
    manifest['target_features_sha256'] = checkpoint_digest(features / 'complete.json')
    manifest['target_features_path'] = str(features.resolve())
    manifest['score_sha256'][target.name] = checkpoint_digest(target)
    save_score_manifest(output, manifest)
    del model, encoders, collator, arrays
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    logger.info('%s METRICS: raw and transductive; %d bootstrap resamples', fold.upper(), bootstrap_resamples)
    report = evaluate_run(output, fold, seed=1234, source_data_root=data_root,
                          target_data_root=data_root, bootstrap_resamples=bootstrap_resamples)
    check_guards(guards)
    save_json(output / 'evaluation_complete.json', {'model_name': NAME, 'fold': fold, 'seed': 1234,
        'checkpoint_sha256': done['best_checkpoint_sha256'], 'metrics_sha256': checkpoint_digest(output / 'metrics.json'),
        'source_validation_sha256': manifest['source_validation_sha256'], 'score_sha256': manifest['score_sha256'],
        'target_features_sha256': manifest['target_features_sha256'], 'no_target_tuning': True,
        'source_selected_checkpoint_only': True, 'exploratory_after_prior_target_exposure': True})
    logger.info('COMPLETE %s %s | target EER %.4f%% AUROC %.6f', NAME, fold.upper(), report['target']['eer']*100, report['target']['auroc'])
    return report


def wait_for_completion(run, fold, family, logger):
    started = time.monotonic()
    launch = family / 'launch_f2_f3_f4'
    while not (run / 'training_complete.json').exists():
        launch = family / 'launch_f2_f3_f4'
        if (launch / 'exit.json').exists() or (launch / 'complete.json').exists():
            raise RuntimeError(f'{fold} training queue exited without completion; evaluation held')
        record = read(launch / 'launch_verified.json')
        pid = record.get('pid')
        if pid is None or not Path(f'/proc/{pid}').exists():
            raise RuntimeError('Training process unavailable; evaluation held')
        if time.monotonic()-started > 48*3600:
            raise TimeoutError('Training completion wait exceeded48h; no evaluation launched')
        logger.info('WAIT %s: training incomplete; no target access', fold.upper())
        time.sleep(30)
    # Score-manifest serialization immediately follows training_complete.json.
    # Wait for the trainer's verified completion marker in its console / next-fold
    # transition, avoiding the short artifact publication race.
    while not (run / 'score_manifest.json').exists():
        if (launch / 'exit.json').exists():
            raise RuntimeError('Training queue exited without source manifest')
        if time.monotonic()-started > 48*3600:
            raise TimeoutError('Source manifest not published')
        time.sleep(2)
    verify_completion(run, fold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', nargs='+', choices=tuple(FOLDS), default=list(FOLDS))
    parser.add_argument('--wait-for-training', action='store_true')
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/drive/audio-deepfake-cache'))
    args = parser.parse_args()
    if args.folds != [f for f in FOLDS if f in args.folds]:
        parser.error('Unique canonical fold order required')
    if any((OUTPUT / fold / '1234').exists() for fold in args.folds):
        raise FileExistsError('Evaluation output exists; no automatic restart')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('moe4_evaluation')
    for fold in args.folds:
        run = FAMILY / fold / '1234'
        if not (run / 'training_complete.json').exists() and not args.wait_for_training:
            logger.info('HOLD %s: training incomplete', fold.upper())
            continue
        wait_for_completion(run, fold, FAMILY, logger)
        evaluate(run, OUTPUT / fold / '1234', fold, args.data_root, 'cuda', logger)


if __name__ == '__main__':
    main()
