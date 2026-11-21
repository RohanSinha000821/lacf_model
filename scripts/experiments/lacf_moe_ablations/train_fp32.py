"""FP32 A1--A6, B1--B6, C1--C3 after the verified main FP32 F1--F4 queue.

Same models, cached features and seed1234 recipe; separate historical-safe outputs.
No new architecture, seeds, inner-source fits or target evaluation are queued.
"""
import gc
import hashlib
import io
import itertools
import json
import logging
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from audio_deepfake_detection.experiments.lacf_moe_ablations import model as core_models
from audio_deepfake_detection.experiments.lacf_moe_ablations import additional_model as additional_models
from audio_deepfake_detection.experiments.lacf_moe_ablations import loss_model as loss_models
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_bank
from audio_deepfake_detection.lacf.training import make_optimizer
from audio_deepfake_detection.metrics import calibrate_source_scores, compute_auroc, compute_eer, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, save_score_manifest, verify_completed_run, verify_source_macro
from scripts.lacf.train_ablations import fixed_prototypes
from scripts.lacf.train_probes import save_json, seed_all
from scripts.lacf_moe.train import assert_bank, checkpoint, guards
from scripts.lacf_moe.train_fp32 import PRECISION, assert_fp32, train_epoch, verify_completion as verify_reference

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / 'outputs/experiments/lacf_moe_ablations/fp32'
CACHE = ROOT / 'outputs/experiments/lacf_diagnostics/frozen_features'
DATA = Path('/mnt/drive/audio-deepfake-cache')
REFERENCE = ROOT / 'outputs/lacf_moe/fp32'
STUDY = 'lacf_moe_ablations'
NAMES = {**core_models.NAMES, **additional_models.NAMES, **loss_models.NAMES}
PARAMETERS = {**core_models.PARAMETERS, **additional_models.PARAMETERS, **loss_models.PARAMETERS}
FEATURE_INPUTS = {**core_models.FEATURE_INPUTS, **additional_models.FEATURE_INPUTS, **loss_models.FEATURE_INPUTS}
CORE = tuple(NAMES)  # A0 is reused; the fifteen fixed controls are fresh fits.
OBJECTIVES = {v:loss_models.OBJECTIVES.get(v, 'detection BCE only') for v in CORE}
EXPECTED_FITS = 60


def build_model(variant, prototypes):
    if variant in loss_models.VARIANTS:
        return loss_models.build_model(variant, prototypes)
    if variant in additional_models.VARIANTS:
        return additional_models.build_model(variant, prototypes)
    return core_models.build_model(variant, prototypes)


def configuration_for(model):
    config = model.configuration()
    config['score_precision'] = PRECISION
    return config


def dependency_ready(provenance):
    """Only a successful, fully verified producer exit unlocks the dependent queue."""
    launch = REFERENCE / 'launch'
    for name, digest in provenance['dependency_inputs_sha256'].items():
        if checkpoint_digest(launch / name) != digest:
            raise ValueError('Main FP32 producer launch changed: ' + name)
    exit_path = launch / 'exit.json'
    if not exit_path.exists():
        return False
    status = read(exit_path)
    if status.get('trainer_exit') != 0 or status.get('logger_exit') != 0 or status.get('verified_complete') is not True:
        raise ValueError('Main FP32 queue failed/stopped; dependent ablations will not start')
    done = read(launch / 'complete.json')
    if done.get('folds') != list(FOLDS) or done.get('precision') != PRECISION or done.get('target_access') is not False:
        raise ValueError('Main FP32 producer completion identity differs')
    if set(done.get('completion_sha256', {})) != set(FOLDS):
        raise ValueError('Main FP32 producer did not complete all four folds')
    for fold, digest in done['completion_sha256'].items():
        if checkpoint_digest(REFERENCE / fold / '1234/training_complete.json') != digest:
            raise ValueError('Main FP32 fold completion changed')
    return True


def wait_for_reference(provenance, launch, logger):
    missing_checks = 0
    while True:
        guards(provenance)
        if dependency_ready(provenance):
            return
        pid = provenance['dependency_pid']
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
            alive = b'scripts.lacf_moe.train_fp32' in command
        except FileNotFoundError:
            alive = False
        missing_checks = 0 if alive else missing_checks + 1
        if missing_checks >= 3:
            raise RuntimeError('Main FP32 producer disappeared without verified exit; no retry')
        progress(launch, phase='waiting for main FP32 F1--F4 verified completion', completed_fits=0)
        logger.info('WAIT | main LACF-MoE FP32 queue must finish successfully; GPU and cache not loaded')
        time.sleep(30)



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
    if (report.get('precision') != PRECISION or report.get('linear_output_dtypes') != ['torch.float32']
            or report.get('serialized_source_reload_bit_exact') is not True
            or report.get('passed') is not True or report['fold'] != fold or report['variant'] != variant
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
    """FP32 model and final scores; optional routing in the same source-only pass."""
    assert_fp32(model)
    model.eval(); scores, metrics, routing = {}, {}, {}
    for domain in FOLDS[fold]['sources']:
        data = bank[domain, 'dev']
        loader = FeatureBatches([data], FEATURE_INPUTS[variant], device, training=False, seed=1234)
        values = torch.empty(loader.count, dtype=torch.float32, device=device)
        totals, offset = {}, 0
        for batch in loader:
            with torch.autocast(torch.device(device).type, enabled=False):
                output = model(batch)
            if output['logit'].dtype != torch.float32:
                raise TypeError('Scoring layer did not execute in FP32')
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
            or config['model']['objective'] != OBJECTIVES[variant]
            or config.get('precision') != PRECISION
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
    prototypes = next((value for key, value in state['model_state_dict'].items()
                       if key.endswith('prototypes')), torch.eye(8, 512))
    with torch.random.fork_rng(devices=[]):
        expected = configuration_for(build_model(variant, prototypes))
    if json.loads(json.dumps(expected)) != config['model']:
        raise ValueError('Ablation architecture/loss/precision configuration differs')
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
        raise ValueError('Only the predefined fifteen FP32 A/B/C controls may be trained; A0 is reused')
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
                  'excluded_target': FOLDS[fold]['target'], 'target_access': False, 'model': configuration_for(model),
                  'batch_size': 132, 'accumulation': 1, 'optimizer': 'AdamW', 'lr': 3e-4,
                  'weight_decay': 1e-4, 'scheduler': 'constant', 'max_epochs': max_epochs, 'patience': patience,
                  'precision': PRECISION,
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
        logger.info('START %s %s | %d records | %d trainable | FP32 | %s',
                    variant, fold.upper(), train.count, PARAMETERS[variant], OBJECTIVES[variant])
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
    linear_dtypes = set()
    hooks = [m.register_forward_hook(lambda m, args, out: linear_dtypes.add(str(out.dtype)))
             for m in model.modules() if isinstance(m, torch.nn.Linear)]
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
    serialized = io.BytesIO()
    torch.save(state, serialized); serialized.seek(0)
    fresh = build_model(variant, prototypes).to(device)
    fresh.load_state_dict(torch.load(serialized, map_location='cpu', weights_only=True), strict=True)
    for hook in hooks: hook.remove()
    if linear_dtypes != {'torch.float32'}:
        raise TypeError('Source model layers did not execute in FP32')
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
              'probe_weights_used_for_final_training': False, 'target_access': False, 'precision': PRECISION,
              'linear_output_dtypes': sorted(linear_dtypes), 'serialized_source_reload_bit_exact': True,
              'serialized_probe_sha256': hashlib.sha256(serialized.getvalue()).hexdigest(),
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
        raise FileExistsError('Prior FP32 ablation launch/runs preserved; no retry/resume/overwrite')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('lacf_moe_ablations_fp32')
    provenance = read(launch / 'provenance.json')
    guards(provenance)
    save_json(launch / 'waiting.json', {'pid':os.getpid(), 'target_access':False, 'expected_fits':EXPECTED_FITS})
    # CPU-only dependency wait: do not initialize CUDA or duplicate the 63GiB bank.
    wait_for_reference(provenance, launch, logger)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.nnpack.set_flags(False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no precision/device fallback')
    torch.cuda.set_per_process_memory_fraction(1024**3 / torch.cuda.get_device_properties(0).total_memory)
    while True:
        guards(provenance)
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
        free, _ = torch.cuda.mem_get_info()
        if available >= 75 * 1024**3 and free >= 2 * 1024**3:
            break
        progress(launch, phase='waiting for RAM/GPU after main completion', completed_fits=0)
        logger.info('WAIT | need75GiB RAM and2GiB GPU free; other jobs preserved')
        time.sleep(30)
    save_json(launch / 'started.json', {'pid':os.getpid(), 'target_access':False, 'precision':PRECISION})
    progress(launch, phase='loading verified source cache once for all sixty fits', completed_fits=0)
    bank = load_bank(CACHE, DATA, logger=logger)
    for ds in bank.values():
        if not torch.isfinite(ds.wavlm).all() or not torch.isfinite(ds.clap).all():
            raise ValueError('Nonfinite source cache')
    prototypes, evidence = fixed_prototypes()
    if evidence != provenance['prototypes']:
        raise ValueError('Original prototype bank changed')
    provenance['reference_sha256'] = {fold:{p.name:checkpoint_digest(p) for p in (REFERENCE / fold / '1234').iterdir()
        if p.is_file() and p.name != 'training.log' and p.suffix in ('.pt', '.json', '.csv')}
        for fold in FOLDS}
    save_json(launch / 'reference_inputs.json', provenance['reference_sha256'])
    save_json(launch / 'queue_config.json', {'study':STUDY, 'precision':PRECISION, 'variants':NAMES,
        'folds':list(FOLDS), 'seed_plan':[1234], 'expected_fresh_fits':EXPECTED_FITS,
        'objectives':OBJECTIVES, 'reference_ids':['A0', 'C0'], 'reference_fits_reused':4,
        'scope':'existing full-fold ablations only; not three-seed inner-source replication or CLAP repair',
        'target_access':False, 'target_evaluation_queued':False, 'provenance':provenance})
    reuse_reference(bank, prototypes, provenance, 'cuda', launch, logger)
    save_json(launch / 'queue_status.json', {'completed_fits':0, 'runs':[]})
    completed = run_queue(bank, prototypes, provenance, 'cuda', launch, logger)
    if len(completed) != EXPECTED_FITS:
        raise ValueError('Incomplete ablation queue')
    guards(provenance)
    save_json(launch / 'complete.json', {'study':STUDY, 'precision':PRECISION, 'completed_fits':len(completed),
        'runs':completed, 'reference_runs_reused':4, 'target_access':False})
    progress(launch, phase='all sixty FP32 source fits verified complete', completed_fits=len(completed))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if (FAMILY / 'launch').is_dir():
            progress(FAMILY / 'launch', phase='FAILED; stopped without retry')
        logging.exception('FP32 ablations stopped; preserve evidence, no automatic retry')
        raise
