"""A0 reuse, then fresh A1–A6 × F1–F4, seed1234; source data only."""

import gc
import itertools
import json
import logging
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from audio_deepfake_detection.experiments.lacf_moe_ablations.model import (
    ADDITIONAL, CORE, FEATURE_INPUTS, NAMES, PARAMETERS, build_model,
)
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_bank
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.metrics import (
    calibrate_source_scores, compute_auroc, compute_eer, read_score_csv, write_score_csv,
)
from audio_deepfake_detection.protocol import (
    FOLDS, checkpoint_digest, save_score_manifest, verify_completed_run, verify_source_macro,
)
from scripts.lacf.train_ablations import fixed_prototypes
from scripts.lacf.train_probes import save_json, seed_all
from scripts.lacf_moe.train import assert_bank, checkpoint, guards, verify_completion as verify_reference


ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / 'outputs/experiments/lacf_moe_ablations'
CACHE = ROOT / 'outputs/experiments/lacf_diagnostics/frozen_features'
DATA = Path('/mnt/drive/audio-deepfake-cache')
REFERENCE = ROOT / 'outputs/lacf_moe'
STUDY = 'lacf_moe_ablations'


def read(path):
    return json.loads(path.read_text())


def progress(launch, **status):
    path = launch / 'progress.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'unix_time': time.time(), 'target_access': False, **status}, indent=2) + '\n')
    temporary.replace(path)


def labels_for(bank, fold):
    return {d: dict(zip(bank[d, 'dev'].ids, bank[d, 'dev'].labels.tolist())) for d in FOLDS[fold]['sources']}


def validate_preflight(provenance, fold, variant, device):
    path = Path(provenance['source_validation_record'])
    if checkpoint_digest(path) != provenance['source_validation_sha256']:
        raise ValueError('Source preflight evidence changed')
    report = read(path)
    if (report.get('passed') is not True or report['fold'] != fold or report['variant'] != variant
            or report['device'] != torch.device(device).type
            or report['full_source_dev_reload_bit_exact'] is not True
            or report['probe_weights_used_for_final_training'] is not False
            or report['trainable_parameters'] != PARAMETERS[variant]
            or report['code_sha256'] != provenance['code_sha256']
            or report['cache_completion_sha256'] != provenance.get('cache_completion_sha256')
            or report['target_access'] is not False):
        raise ValueError('Source preflight evidence does not match this fit')


def summarize_routes(totals):
    result = {}
    for name, row in zip(('all', 'bonafide', 'spoof'), totals.cpu()):
        count = int(row[0]); result[name] = {'count': count}
        if count:
            mean = row[1:4] / count
            result[name].update(mean_weights=mean.tolist(),
                                std_weights=(row[4:7] / count - mean.square()).clamp_min(0).sqrt().tolist(),
                                mean_entropy_nats=float(row[7] / count),
                                top_expert_counts=row[8:11].long().tolist())
    return result


@torch.inference_mode()
def score_sources(model, bank, fold, variant, device, logger=None, *, collect_routing=False):
    """Same BF16 head batches and shared metrics; optional routes in the same pass."""
    model.eval(); scores, metrics, routing = {}, {}, {}
    for domain in FOLDS[fold]['sources']:
        data = bank[domain, 'dev']
        loader = FeatureBatches([data], FEATURE_INPUTS[variant], device, training=False, seed=1234)
        values = torch.empty(loader.count, dtype=torch.float32, device=device)
        totals, offset = {}, 0
        for batch in loader:
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                output = model(batch)
            size = len(batch['labels']); values[offset:offset + size] = output['logit'].float(); offset += size
            if collect_routing:
                for key in ('routing', 'fusion_routing'):
                    if key not in output:
                        continue
                    weights = output[key].double()
                    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1, keepdim=True)
                    packed = torch.cat((torch.ones_like(entropy), weights, weights.square(), entropy,
                                        torch.nn.functional.one_hot(weights.argmax(-1), 3).double()), -1)
                    masks = torch.stack((torch.ones_like(batch['labels']), batch['labels'] == 0,
                                         batch['labels'] == 1)).double()
                    totals.setdefault(key, torch.zeros(3, 11, dtype=torch.float64, device=device))
                    totals[key] += masks @ packed
        if offset != loader.count or not torch.isfinite(values).all():
            raise ValueError('Incomplete or nonfinite source scores')
        scores[domain] = values.cpu()
        metrics[domain] = {'eer': compute_eer(data.labels.numpy(), scores[domain].numpy()),
                           'auroc': compute_auroc(data.labels.numpy(), scores[domain].numpy())}
        routing[domain] = {key: summarize_routes(value) for key, value in totals.items()}
        if logger:
            logger.info('%s %s DEV %s | EER %.4f%% AUROC %.6f', variant, fold.upper(), domain,
                        metrics[domain]['eer'] * 100, metrics[domain]['auroc'])
    return scores, {'by_domain': metrics, 'macro_eer': statistics.mean(m['eer'] for m in metrics.values()),
                    'macro_auroc': statistics.mean(m['auroc'] for m in metrics.values())}, routing


def verify_completion(run, fold, variant, expected_labels):
    done, digest = verify_completed_run(run, fold, 1234)
    config = read(run / 'config.json')
    if (done['study'] != STUDY or done['variant'] != variant or config['variant'] != variant
            or config['seed_plan'] != [1234] or config['sources'] != list(FOLDS[fold]['sources'])
            or config['target_access'] is not False or done['target_access'] is not False
            or config['model']['trainable_parameters'] != PARAMETERS[variant]
            or config['model']['objective'] != 'detection BCE only'
            or done['best_checkpoint_sha256'] != digest
            or done['selected_checkpoint_reloaded_and_source_verified'] is not True):
        raise ValueError('Ablation completion identity differs')
    for filename, key in (('config.json', 'config_sha256'), ('metrics.json', 'metrics_sha256'),
                          ('routing.json', 'routing_sha256'), ('source_calibration.json', 'calibration_sha256')):
        if checkpoint_digest(run / filename) != done[key]:
            raise ValueError('Completion artifact changed: ' + filename)
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if (state['epoch'] != done['best_epoch'] or state['config_sha256'] != done['config_sha256']
            or state['source_dev'] != read(run / 'metrics.json')['source_dev']):
        raise ValueError('Selected checkpoint/config/source result differs')
    inputs = {}
    for domain in FOLDS[fold]['sources']:
        filename = f'source_dev_{domain}.csv'
        if checkpoint_digest(run / filename) != done['source_scores_sha256'][filename]:
            raise ValueError('Source export changed')
        inputs[domain] = read_score_csv(run / filename, domain, expected_labels=expected_labels[domain])
    calibration = calibrate_source_scores(inputs)
    if calibration != read(run / 'source_calibration.json'):
        raise ValueError('Canonical source calibration differs')
    verify_source_macro(calibration['source_dev']['macro_eer'], done['best_macro_source_dev_eer'])
    manifest = read(run / 'score_manifest.json')
    if (manifest['study'] != STUDY or manifest['variant'] != variant or manifest['fold'] != fold
            or manifest['seed'] != 1234 or manifest['sources'] != list(FOLDS[fold]['sources'])
            or manifest['target_access'] is not False or manifest['checkpoint_sha256'] != digest
            or manifest['training_completion_sha256'] != checkpoint_digest(run / 'training_complete.json')
            or manifest['score_sha256'] != done['source_scores_sha256']):
        raise ValueError('Source provenance manifest differs')
    return done


def fit(bank, prototypes, output, fold, variant, provenance, device, logger, *, max_epochs=30, patience=5,
        launch=None):
    if variant not in CORE[1:]:
        raise ValueError('Only core A1–A6 may be trained; A0 is reused and additional controls stay held')
    assert_bank(bank, fold); guards(provenance); validate_preflight(provenance, fold, variant, device)
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / 'training.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s')); logger.addHandler(handler)
    try:
        seed_all(1234); model = build_model(variant, prototypes).to(device); optimizer = make_optimizer(model)
        sources = FOLDS[fold]['sources']
        train = FeatureBatches([bank[d, 'train'] for d in sources], FEATURE_INPUTS[variant], device,
                               training=True, seed=1234)
        config = {'study': STUDY, 'variant': variant, 'model_name': f'LACF-MoE {variant} {NAMES[variant]}',
                  'fold': fold, 'seed': 1234, 'seed_plan': [1234], 'sources': list(sources),
                  'excluded_target': FOLDS[fold]['target'], 'target_access': False, 'model': model.configuration(),
                  'batch_size': 132, 'accumulation': 1, 'optimizer': 'AdamW', 'lr': 3e-4,
                  'weight_decay': 1e-4, 'scheduler': 'constant', 'max_epochs': max_epochs, 'patience': patience,
                  'precision': 'BF16 mixed; sensitive FP32; reference BF16-rounded logits',
                  'sampling': 'equal domain/class mass; replacement; recording count per epoch',
                  'training_views': 'center plus two fixed random up-to10s views; identical short views aliased',
                  'development_views': 'centered up-to10s; native-unpadded cached encoders; head batch132',
                  'selection': 'strict equal-domain macro source-dev EER; ties earlier',
                  'resume': False, 'feature_workers': 0, 'train_records': train.count,
                  'dev_records': {d: len(bank[d, 'dev'].labels) for d in sources},
                  'scope': 'fixed exploratory full-fold controls after prior target exposure; no target tuning',
                  'provenance': provenance}
        save_json(output / 'config.json', config); digest = checkpoint_digest(output / 'config.json')
        best, best_epoch, stale, selected_scores = float('inf'), 0, 0, None
        logger.info('START %s %s | %d records | %d trainable | cached encoders; BCE only',
                    variant, fold.upper(), train.count, PARAMETERS[variant])
        with (output / 'history.jsonl').open('x') as history:
            for epoch in range(1, max_epochs + 1):
                guards(provenance); tick = time.monotonic()
                if launch:
                    progress(launch, phase='training', variant=variant, fold=fold, epoch=epoch,
                             best_epoch=best_epoch, completed_fits=read(launch / 'queue_status.json')['completed_fits'])
                losses, updates = train_epoch(model, train, optimizer, device, accumulation_steps=1,
                                              logger=logger, phase=f'{variant} {fold.upper()} epoch{epoch}')
                scores, metrics, _ = score_sources(model, bank, fold, variant, device, logger)
                if metrics['macro_eer'] < best:
                    best, best_epoch, stale, selected_scores = metrics['macro_eer'], epoch, 0, scores
                    checkpoint(output / 'best.pt', {'model_state_dict': model.state_dict(), 'epoch': epoch,
                                                   'source_dev': metrics, 'config_sha256': digest})
                else:
                    stale += 1
                checkpoint(output / 'last.pt', {'model_state_dict': model.state_dict(), 'epoch': epoch,
                           'optimizer_state_dict': optimizer.state_dict(), 'source_dev': metrics,
                           'best_epoch': best_epoch, 'patience': stale, 'config_sha256': digest})
                row = {'epoch': epoch, 'losses': losses, 'updates': updates, 'source_dev': metrics,
                       'best_epoch': best_epoch, 'patience': stale, 'seconds': time.monotonic() - tick}
                history.write(json.dumps(row, allow_nan=False) + '\n'); history.flush()
                logger.info('%s %s EPOCH %d | source EER %.4f%% AUROC %.6f | best%d patience%d/%d | %.1fs',
                            variant, fold.upper(), epoch, metrics['macro_eer'] * 100, metrics['macro_auroc'],
                            best_epoch, stale, patience, row['seconds'])
                if stale >= patience:
                    break
        state = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        del model, optimizer; gc.collect()
        model = build_model(variant, prototypes).to(device); model.load_state_dict(state['model_state_dict'], strict=True)
        scores, metrics, routes = score_sources(model, bank, fold, variant, device, logger, collect_routing=True)
        if (state['config_sha256'] != digest or checkpoint_digest(output / 'config.json') != digest
                or metrics != state['source_dev'] or any(not torch.equal(scores[d], selected_scores[d]) for d in sources)):
            raise ValueError('Fresh selected-checkpoint source reload differs')
        inputs, hashes = {}, {}; labels = labels_for(bank, fold)
        for domain in sources:
            data = bank[domain, 'dev']; path = output / f'source_dev_{domain}.csv'
            write_score_csv(path, zip(data.ids, itertools.repeat(domain), data.labels.tolist(), scores[domain].tolist()))
            inputs[domain] = read_score_csv(path, domain, expected_labels=labels[domain])
            order = sorted(range(len(data.ids)), key=data.ids.__getitem__)
            if not np.array_equal(inputs[domain][1], scores[domain].numpy()[order]):
                raise ValueError('Canonical source export differs')
            hashes[path.name] = checkpoint_digest(path)
        calibration = calibrate_source_scores(inputs); verify_source_macro(calibration['source_dev']['macro_eer'], best)
        save_json(output / 'source_calibration.json', calibration)
        save_json(output / 'routing.json', {'scope': 'source development only; descriptive, not selection',
                  'domains': routes, 'argmax_ties': 'first expert; inspect soft weights and entropy too'})
        save_json(output / 'metrics.json', {'study': STUDY, 'variant': variant, 'source_dev': metrics,
                  'best_epoch': best_epoch, 'final_epoch': epoch, 'target_access': False})
        done = {'study': STUDY, 'variant': variant, 'fold': fold, 'seed': 1234, 'sources': list(sources),
                'best_epoch': best_epoch, 'final_epoch': epoch, 'best_macro_source_dev_eer': best,
                'selected_checkpoint_reloaded_and_source_verified': True,
                'best_checkpoint_sha256': checkpoint_digest(output / 'best.pt'), 'config_sha256': digest,
                'metrics_sha256': checkpoint_digest(output / 'metrics.json'), 'source_scores_sha256': hashes,
                'calibration_sha256': checkpoint_digest(output / 'source_calibration.json'),
                'routing_sha256': checkpoint_digest(output / 'routing.json'), 'target_access': False}
        guards(provenance); save_json(output / 'training_complete.json', done)
        save_score_manifest(output, {'study': STUDY, 'variant': variant, 'fold': fold, 'seed': 1234,
                            'sources': list(sources), 'checkpoint_sha256': done['best_checkpoint_sha256'],
                            'training_completion_sha256': checkpoint_digest(output / 'training_complete.json'),
                            'score_sha256': hashes, 'target_access': False})
        verify_completion(output, fold, variant, labels)
        logger.info('COMPLETE %s %s | selected epoch%d source EER %.4f%% | reload/export/calibration passed',
                    variant, fold.upper(), best_epoch, best * 100)
        return done
    finally:
        logger.removeHandler(handler); handler.close()


def preflight(bank, prototypes, fold, variant, provenance, device, logger):
    assert_bank(bank, fold); guards(provenance); seed_all(1234)
    model = build_model(variant, prototypes).to(device); optimizer = make_optimizer(model)
    train_sets = [bank[d, 'train'] for d in FOLDS[fold]['sources']]
    for count in (8, 16):
        loader = FeatureBatches(train_sets, FEATURE_INPUTS[variant], device, training=True, seed=1234, max_batches=count)
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        losses, updates = train_epoch(model, loader, optimizer, device, accumulation_steps=1)
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.monotonic() - tick
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    expected, metrics, _ = score_sources(model, bank, fold, variant, device)
    fresh = build_model(variant, prototypes).to(device); fresh.load_state_dict(state, strict=True)
    tick = time.monotonic(); actual, other, _ = score_sources(fresh, bank, fold, variant, device)
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()
    dev_seconds = time.monotonic() - tick
    if metrics != other or any(not torch.equal(expected[d], actual[d]) for d in expected):
        raise ValueError('GPU source preflight reload differs')
    report = {'passed': True, 'variant': variant, 'fold': fold, 'losses': losses, 'measured_updates': updates,
              'device': torch.device(device).type, 'trainable_parameters': PARAMETERS[variant],
              'code_sha256': provenance['code_sha256'],
              'cache_completion_sha256': provenance.get('cache_completion_sha256'),
              'warmup_updates': 8, 'train_examples_per_second': loader.count / seconds,
              'full_source_dev_reload_bit_exact': True, 'source_dev_seconds': dev_seconds,
              'probe_weights_used_for_final_training': False, 'target_access': False,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.device(device).type == 'cuda' else None,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.device(device).type == 'cuda' else None}
    report['rough_epoch_seconds'] = sum(len(d.labels) for d in train_sets) / report['train_examples_per_second'] + dev_seconds
    logger.info('PREFLIGHT %s %s passed | %.1f audio/s | estimated epoch %.1fs (short shared-GPU measurement)',
                variant, fold.upper(), report['train_examples_per_second'], report['rough_epoch_seconds'])
    del model, optimizer, fresh; gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return report


def reuse_reference(bank, prototypes, provenance, device, launch, logger):
    for fold in FOLDS:
        original = REFERENCE / fold / '1234'
        guards(provenance)
        for filename, digest in provenance['reference_sha256'][fold].items():
            if checkpoint_digest(original / filename) != digest:
                raise ValueError('Launch-bound reference changed')
        verify_reference(original, fold, expected_labels=labels_for(bank, fold))
        state = torch.load(original / 'best.pt', map_location='cpu', weights_only=True)
        if not torch.equal(state['model_state_dict']['backend.prototypes'], prototypes):
            raise ValueError('Reference prototype bank differs')
        model = build_model('A0', prototypes).to(device); model.load_state_dict(state['model_state_dict'], strict=True)
        scores, metrics, _ = score_sources(model, bank, fold, 'A0', device, logger)
        if metrics != state['source_dev']:
            raise ValueError('Reference source metrics differ')
        for domain in FOLDS[fold]['sources']:
            data = bank[domain, 'dev']
            _, saved = read_score_csv(original / f'source_dev_{domain}.csv', domain,
                                      expected_labels=labels_for(bank, fold)[domain])
            order = sorted(range(len(data.ids)), key=data.ids.__getitem__)
            if not np.array_equal(scores[domain].numpy()[order], saved):
                raise ValueError('Reference full source score reload differs')
        link = FAMILY / 'A0_reference' / fold / '1234'; link.parent.mkdir(parents=True, exist_ok=False)
        link.symlink_to(os.path.relpath(original, link.parent), target_is_directory=True)
        save_json(launch / f'reference_reuse_{fold}.json', {'fold': fold, 'variant': 'A0',
                  'full_source_reload_bit_exact': True, 'reference_artifact_sha256': provenance['reference_sha256'][fold],
                  'training_rerun': False, 'target_access': False})
        del model, state, scores; gc.collect(); torch.cuda.empty_cache()


def run_queue(bank, prototypes, provenance, device, launch, logger):
    completed = []
    for variant in CORE[1:]:
        for fold in FOLDS:
            if torch.device(device).type == 'cuda':
                while torch.cuda.mem_get_info()[0] < 2 * 1024**3:
                    guards(provenance)
                    progress(launch, phase='waiting for GPU headroom', variant=variant, fold=fold,
                             completed_fits=len(completed))
                    logger.info('WAIT %s %s | need2GiB free, preserving other jobs', variant, fold.upper())
                    time.sleep(10)
            subset = {(d, s): bank[d, s] for d in FOLDS[fold]['sources'] for s in ('train', 'dev')}
            progress(launch, phase='source preflight', variant=variant, fold=fold, completed_fits=len(completed))
            proof = launch / f'source_validation_{variant}_{fold}.json'
            save_json(proof, preflight(subset, prototypes, fold, variant, provenance, device, logger))
            current = {**provenance, 'source_validation_record': str(proof),
                       'source_validation_sha256': checkpoint_digest(proof)}
            run = FAMILY / f'{variant}_{NAMES[variant]}' / fold / '1234'
            fit(subset, prototypes, run, fold, variant, current, device, logger, launch=launch)
            verify_completion(run, fold, variant, labels_for(subset, fold))
            completed.append({'variant': variant, 'fold': fold,
                              'completion_sha256': checkpoint_digest(run / 'training_complete.json')})
            progress(launch, phase='advancing after verified completion', variant=variant, fold=fold,
                     completed_fits=len(completed))
            status = launch / 'queue_status.json'; temporary = status.with_suffix('.json.tmp')
            temporary.write_text(json.dumps({'completed_fits': len(completed), 'runs': completed}, indent=2) + '\n')
            temporary.replace(status)
            gc.collect()
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
    return completed


def main():
    launch = FAMILY / 'launch'
    if (launch / 'started.json').exists() or any((FAMILY / f'{v}_{NAMES[v]}').exists() for v in CORE):
        raise FileExistsError('Previous runs/launch preserved; no automatic retry, resume or overwrite')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger(STUDY)
    provenance = read(launch / 'provenance.json'); guards(provenance)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required; no fallback')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.nnpack.set_flags(False)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(1024**3 / total)
    # RAM required for the verified CPU feature bank plus reserve; wait without
    # stopping any other job. Fresh GPU headroom is checked for every fit below.
    save_json(launch / 'started.json', {'pid': os.getpid(), 'unix_time': time.time(), 'target_access': False})
    while True:
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
        free, _ = torch.cuda.mem_get_info()
        if available >= 75 * 1024**3 and free >= 2 * 1024**3:
            break
        progress(launch, phase='waiting for resources', available_ram_bytes=available, free_gpu_bytes=free)
        logger.info('WAIT | RAM %.1fGiB (need75) GPU free %.1fGiB (need2)', available / 1024**3, free / 1024**3)
        time.sleep(10)
    progress(launch, phase='verifying and loading source cache', completed_fits=0)
    bank = load_bank(CACHE, DATA, logger=logger)
    for data in bank.values():
        if not torch.isfinite(data.wavlm).all() or not torch.isfinite(data.clap).all():
            raise ValueError('Nonfinite cached source features')
    save_json(launch / 'cache_validation.json', {'all_canonical_memberships_and_payload_sha256_verified': True,
              'all_features_finite': True, 'target_access': False,
              'splits': {f'{d}/{s}': len(value.labels) for (d, s), value in bank.items()}})
    prototypes, evidence = fixed_prototypes(); provenance['prototypes'] = evidence
    save_json(launch / 'queue_config.json', {'study': STUDY, 'seed_plan': [1234], 'folds': list(FOLDS),
              'core': {v: {'name': NAMES[v], 'parameters': PARAMETERS[v]} for v in CORE},
              'additional_not_queued': ADDITIONAL, 'expected_fresh_fits': 24, 'provenance': provenance,
              'target_access': False, 'target_evaluation_queued': False})
    progress(launch, phase='full source reload of preserved A0 references', completed_fits=0)
    reuse_reference(bank, prototypes, provenance, 'cuda', launch, logger)
    save_json(launch / 'queue_status.json', {'completed_fits': 0, 'runs': []})
    completed = run_queue(bank, prototypes, provenance, 'cuda', launch, logger)
    guards(provenance)
    save_json(launch / 'complete.json', {'study': STUDY, 'completed_fits': len(completed), 'runs': completed,
              'reference_runs_reused': 4, 'target_access': False})
    progress(launch, phase='all source training complete', completed_fits=len(completed))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if (FAMILY / 'launch').is_dir():
            progress(FAMILY / 'launch', phase='FAILED; stopped without retry')
        logging.exception('Ablation queue stopped; all evidence preserved')
        raise
