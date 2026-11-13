"""Frozen TF2: preserve F1 and fit fresh F2–F4 from source-only cached features."""

from contextlib import contextmanager
import gc
import json
import logging
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from audio_deepfake_detection.lacf.probes import FeatureBatches, load_bank
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from audio_deepfake_detection.metrics import read_score_csv
from audio_deepfake_detection.token_fusion.model import TokenFusion
from scripts.lacf.train_probes import save_json, seed_all
from scripts.token_fusion import train as original


ROOT = original.ROOT
OUTPUT = ROOT / 'outputs/experiments/token_fusion/tf2_frozen'
LAUNCH = OUTPUT / 'launch'
CACHE = ROOT / 'outputs/experiments/lacf_diagnostics/frozen_features'
DATA = Path('/mnt/drive/audio-deepfake-cache')
FRESH_FOLDS = ('f2', 'f3', 'f4')


def read(path):
    return json.loads(path.read_text())


@contextmanager
def fold_context(fold):
    """Bind the unchanged F1 training functions inside this process only.

    This sequential entry point reuses their exact fitting/export mathematics;
    no source file or state in any other running process is modified.
    """
    if fold not in FOLDS:
        raise ValueError('Expected a canonical F1–F4 fold')
    names = ('FOLD', 'SOURCES', 'SCOPE', 'SEEDS')
    previous = {name: getattr(original, name) for name in names}
    original.FOLD = fold
    original.SOURCES = tuple(FOLDS[fold]['sources'])
    original.SCOPE = (f'full {fold.upper()} source train/dev; selected in-domain development, '
                      'not inner-held-domain validation')
    original.SEEDS = (1234,)
    try:
        yield original
    finally:
        for name, value in previous.items():
            setattr(original, name, value)


def source_bank(bank, fold):
    subset = {(d, s): bank[d, s] for d in FOLDS[fold]['sources'] for s in ('train', 'dev')}
    with fold_context(fold) as runner:
        runner.assert_bank(subset)
    return subset


def guards(provenance):
    original.check_guards(provenance)
    for name, digest in provenance['cache_sha256'].items():
        if checkpoint_digest(CACHE / name) != digest:
            raise ValueError('Frozen source-cache identity changed: ' + name)
    for name, digest in provenance['preserved_f1_sha256'].items():
        if checkpoint_digest(OUTPUT / 'f1/1234' / name) != digest:
            raise ValueError('Preserved F1 artifact changed: ' + name)


def verify_run(run, fold, bank):
    with fold_context(fold) as runner:
        done = runner.verify_fit(run, bank)
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if (state['config'] != read(run / 'config.json') or state['epoch'] != done['best_epoch']
            or state['source_dev'] != read(run / 'metrics.json')['source_dev']):
        raise ValueError('Selected TF2 checkpoint/config/source metrics differ')
    if (state['config']['model'] != TokenFusion('TF2').configuration()
            or state['config']['batch_size'] != 132 or state['config']['accumulation'] != 1
            or state['config']['optimizer'] != 'AdamW' or state['config']['lr'] != 3e-4
            or state['config']['weight_decay'] != 1e-4 or state['config']['scheduler'] != 'constant'
            or state['config']['target_access'] is not False):
        raise ValueError('Frozen TF2 architecture/recipe differs')
    return done


def reload_sources(run, bank, fold, device, logger):
    done = verify_run(run, fold, bank)
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    model = TokenFusion('TF2').to(device)
    model.load_state_dict(state['model_state_dict'], strict=True)
    with fold_context(fold) as runner:
        scores, metrics = runner.score_sources(model, bank, device, logger)
    if metrics != state['source_dev']:
        raise ValueError('Fresh full TF2 source metrics differ')
    for domain in FOLDS[fold]['sources']:
        data = bank[domain, 'dev']
        _, saved = read_score_csv(run / f'source_dev_{domain}.csv', domain,
                                 expected_labels=dict(zip(data.ids, data.labels.tolist())))
        order = sorted(range(len(data.ids)), key=data.ids.__getitem__)
        if not np.array_equal(scores[domain].numpy()[order], saved):
            raise ValueError('Fresh full TF2 source scores differ')
    del model, state, scores
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return done


def preflight(bank, fold, device, logger):
    """Discard all timing updates; compare two full source-dev reloads."""
    seed_all(1234)
    model = TokenFusion('TF2').to(device)
    optimizer = make_optimizer(model)
    loader = FeatureBatches([bank[d, 'train'] for d in FOLDS[fold]['sources']],
                           'concat', device, training=True, seed=1234, max_batches=32)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    losses, updates = train_epoch(model, loader, optimizer, device, accumulation_steps=1)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    with fold_context(fold) as runner:
        expected, metrics = runner.score_sources(model, bank, device, logger)
        del model, optimizer
        gc.collect()
        fresh = TokenFusion('TF2').to(device)
        fresh.load_state_dict(state, strict=True)
        started = time.monotonic()
        actual, other = runner.score_sources(fresh, bank, device, logger)
        torch.cuda.synchronize()
        dev_seconds = time.monotonic() - started
    if metrics != other or any(not torch.equal(expected[d], actual[d]) for d in expected):
        raise ValueError('Frozen TF2 GPU preflight source reload differs')
    count = loader.count
    epoch_records = sum(len(bank[d, 'train'].labels) for d in expected)
    report = {'passed': True, 'fold': fold, 'timed_updates': updates, 'losses': losses,
              'train_examples_per_second': count / seconds, 'source_dev_seconds': dev_seconds,
              'rough_epoch_seconds': epoch_records * seconds / count + dev_seconds,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'full_source_reload_bit_exact': True, 'pilot_weights_used_for_final_fit': False,
              'target_access': False}
    del fresh, state, expected, actual
    gc.collect()
    torch.cuda.empty_cache()
    logger.info('PREFLIGHT %s PASS | estimated epoch %.1f min | %.2f GiB peak allocated',
                fold.upper(), report['rough_epoch_seconds'] / 60,
                report['peak_allocated_bytes'] / 1024**3)
    return report


def run_folds(bank, provenance, device, logger, *, output=OUTPUT, launch=LAUNCH):
    for fold in FRESH_FOLDS:
        guards(provenance)
        wait_for_resources(provenance, logger, cached=True)
        subset = source_bank(bank, fold)
        validation = launch / f'source_validation_{fold}.json'
        save_json(validation, preflight(subset, fold, device, logger))
        current = {**provenance, 'source_validation_record': str(validation),
                   'source_validation_sha256': checkpoint_digest(validation)}
        # fit_run seeds a newly constructed TF2 after every discarded preflight.
        with fold_context(fold) as runner:
            runner.fit_run(subset, output / fold / '1234', 'TF2', 1234, None,
                           current, device, logger)
        verify_run(output / fold / '1234', fold, subset)
        guards(provenance)
        logger.info('%s COMPLETE; source reload/export/calibration verified; advance', fold.upper())
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()


def wait_for_resources(provenance, logger, *, cached=False, timeout=48 * 3600):
    # Feature arrays alone occupy 62.6 GiB; leave room for indices, readers and
    # other users' jobs. Never kill a cache producer or compete until safe.
    manifest = read(CACHE / 'complete.json')
    required_ram = 12 * 1024**3
    if not cached:
        required_ram += sum(s['unique_rows'] * (13 * 768 + 512) * 4
                            for s in manifest['splits'].values())
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        guards(provenance)
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines()}
        free_gpu = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', '0'],
            text=True).strip())
        report = {'phase': 'resource_check', 'available_ram_gib': memory['MemAvailable'] / 1024**3,
                  'required_ram_gib': required_ram / 1024**3, 'free_gpu_mib': free_gpu,
                  'target_access': False}
        original.replace_json(LAUNCH / 'resource_status.json', report)
        if memory['MemAvailable'] >= required_ram and free_gpu >= 5120:
            logger.info('RESOURCE GATE PASS | %.1f GiB RAM available | %d MiB GPU free',
                        report['available_ram_gib'], free_gpu)
            return
        logger.info('WAIT resources | RAM %.1f / required %.1f GiB | GPU %d / required 5120 MiB',
                    report['available_ram_gib'], report['required_ram_gib'], free_gpu)
        time.sleep(30)
    raise TimeoutError('Insufficient live memory for frozen TF2; no other job interrupted')


def main():
    if (LAUNCH / 'started.json').exists() or any((OUTPUT / f / '1234').exists() for f in FRESH_FOLDS):
        raise FileExistsError('Prior frozen TF2 launch/run preserved; no resume or retry')
    provenance = read(LAUNCH / 'provenance.json')
    guards(provenance)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('tf2_frozen')
    save_json(LAUNCH / 'started.json', {'phase': 'resource_check', 'fresh_folds': list(FRESH_FOLDS),
                                      'preserved_f1': True, 'seed': 1234, 'target_access': False})
    wait_for_resources(provenance, logger)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required; no precision fallback')
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.nnpack.set_flags(False)
    # Match original seed_all() and archived numerical behavior.
    seed_all(1234)
    logger.info('CACHE verify/load once | frozen WavLM 13 layers + CLAP | no audio encoder loaded')
    bank = load_bank(CACHE, DATA, logger=logger)
    if any(not torch.isfinite(s.wavlm).all() or not torch.isfinite(s.clap).all() for s in bank.values()):
        raise ValueError('Nonfinite frozen source features')
    save_json(LAUNCH / 'cache_validation.json', {'canonical_ids_labels_and_payload_hashes_verified': True,
              'source_partitions': [f'{d}/{s}' for d, s in bank], 'finite_features': True,
              'cache_completion_sha256': checkpoint_digest(CACHE / 'complete.json'), 'target_access': False})
    wait_for_resources(provenance, logger, cached=True)
    reload_sources(OUTPUT / 'f1/1234', source_bank(bank, 'f1'), 'f1', 'cuda', logger)
    save_json(LAUNCH / 'f1_reuse_verified.json', {'full_source_score_bit_exact': True,
              'selected_checkpoint_reloaded': True, 'retrained': False, 'target_access': False})
    run_folds(bank, provenance, 'cuda', logger)
    records = [verify_run(OUTPUT / fold / '1234', fold, source_bank(bank, fold)) for fold in FOLDS]
    save_json(OUTPUT / 'source_summary.json', {'model_name': 'Frozen WavLM–CLAP Transformer fusion',
              'condition': 'TF2', 'seed': 1234, 'folds': records, 'target_access': False,
              'scope': 'Exploratory full-fold source selection after prior target exposure; no inner selection claim'})
    guards(provenance)
    save_json(LAUNCH / 'complete.json', {'folds': list(FOLDS), 'preserved_f1': True, 'target_access': False,
              'source_summary_sha256': checkpoint_digest(OUTPUT / 'source_summary.json'),
              'completion_sha256': {f: checkpoint_digest(OUTPUT / f / '1234/training_complete.json') for f in FOLDS}})
    logger.info('COMPLETE all four source folds; stop without target evaluation')


if __name__ == '__main__':
    main()
