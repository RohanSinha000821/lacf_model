"""Fresh SSL-last F2, preserving the completed/guarded F1 trainer unchanged."""

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, utterance_id
from audio_deepfake_detection.sota.ssl_gating import FullAudioDataset, collate_audio, make_eval_loader, make_train_loader
from audio_deepfake_detection.sota.ssl_gating_last import (
    LAST_MODEL_CONFIG, RUN_FAMILY, TRAINABLE_PARAMETERS, LastSSLFeatureGating,
)
from audio_deepfake_detection.sota.ssl_gating_partial import detection_losses, partial_scores
from scripts.ssl_gating.train import LEARNING_RATE, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, set_seed, write_json
from scripts.ssl_gating.train_partial import save_checkpoint, score_sources, source_metrics, train_epoch


ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/mnt/drive/audio-deepfake-cache')


def check_guards(launch):
    for name, digest in json.loads((launch / 'source_guards.json').read_text()).items():
        if checkpoint_digest(ROOT / name) != digest:
            raise RuntimeError('Execution/preserved source changed: ' + name)


def validate_source(launch, records):
    """Discarded worst-case20s updates, real source batches and full129s reload."""
    check_guards(launch); set_seed(1234)
    ready = json.loads((launch / 'readiness.json').read_text())
    if ready['passed'] is not True or ready['fold'] != 'f2' or set(ready['sources']) != set(FOLDS['f2']['sources']):
        raise ValueError('F2 source readiness required')
    model = LastSSLFeatureGating().cuda().train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != TRAINABLE_PARAMETERS:
        raise ValueError('Last-block parameter inventory changed')
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    data = FullAudioDataset(ready['long_train_records'], training=True)
    # Long examples dominate the batch; include genuine source audio as well.
    waves, labels = collate_audio([data[i % len(data)] for i in range(24)])
    if waves.shape != (24, 320000):
        raise ValueError('Expected an actual maximum20s padded source batch')
    drop = model.feature_extractor.config.layerdrop
    seed = next(s for s in range(10000)
                if (torch.rand(24, generator=torch.Generator().manual_seed(s)) >= drop).all())
    active = set()
    hooks = [layer.register_forward_hook(lambda _, args, out, i=i: active.add(i))
             for i, layer in enumerate(model.feature_extractor.encoder.layers)]
    steps = []; torch.cuda.reset_peak_memory_stats()
    for index in range(4):
        torch.manual_seed(seed); active.clear(); tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        loss, ce, cka = detection_losses(model, waves.cuda(), labels.cuda())
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite source stress loss')
        loss.backward()
        if (len(active) != 24 or any(p.grad is not None for p in model.parameters() if not p.requires_grad)
                or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)):
            raise RuntimeError('Source stress gradient/freeze/layer execution check failed')
        optimizer.step(); torch.cuda.synchronize()
        steps.append({'seconds': time.monotonic() - tick, 'loss': float(loss.detach()), 'executed_layers': len(active)})
        print('SOURCE STRESS', index + 1, steps[-1], flush=True)
    for hook in hooks: hook.remove()
    iterator = iter(make_train_loader([FullAudioDataset(records[d]['train'], training=True) for d in records],
                                     batch_size=24, seed=1234, num_workers=8, prefetch_factor=2))
    tick = time.monotonic()
    try:
        for _ in range(3):
            waves, labels = next(iterator); optimizer.zero_grad(set_to_none=True)
            loss, ce, cka = detection_losses(model, waves.cuda(non_blocking=True), labels.cuda(non_blocking=True))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite actual source loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
                raise RuntimeError('Nonfinite actual source gradients')
            optimizer.step()
        torch.cuda.synchronize(); measured = time.monotonic() - tick
    finally:
        iterator._shutdown_workers()
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    del optimizer, parameters, iterator, loss, ce, cka, waves, labels; gc.collect(); torch.cuda.empty_cache()
    model.eval(); wave, _ = FullAudioDataset([ready['long_dev_record']])[0]
    with torch.inference_mode(): expected = partial_scores(model, wave[None].cuda()).cpu()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    frontend = model.feature_extractor.config.to_dict()
    model.cpu(); del model; gc.collect(); torch.cuda.empty_cache()
    fresh = LastSSLFeatureGating.from_frontend_config(frontend).cuda().eval()
    fresh.load_state_dict(state, strict=True)
    with torch.inference_mode(): actual = partial_scores(fresh, wave[None].cuda()).cpu()
    if not torch.isfinite(actual).all() or not torch.equal(expected, actual):
        raise RuntimeError('Fresh full-source-recording source reload differs')
    check_guards(launch)
    result = {'passed': True, 'fold': 'f2', 'target_access': False, 'model_config': LAST_MODEL_CONFIG,
              'probe_weights_discarded': True, 'trainable_parameters': TRAINABLE_PARAMETERS,
              'stress_steps': steps, 'peak_allocated_bytes': peak, 'peak_reserved_bytes': reserved,
              'source_timing': {'examples': 72, 'seconds': measured, 'includes_worker_startup': True},
              'fresh_reload_source_score_bit_exact': True, 'full_source_recording_seconds': wave.numel() / 16000,
              'source_guards_sha256': checkpoint_digest(launch / 'source_guards.json'),
              'readiness_sha256': checkpoint_digest(launch / 'readiness.json')}
    del fresh, state; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', choices=('f2',), required=True)
    args = parser.parse_args(); fold = args.fold
    base = ROOT / 'outputs' / RUN_FAMILY / fold; run = base / '1234'; launch = base / 'launch'
    if run.exists() or (launch / 'started.json').exists():
        raise FileExistsError('Existing run/launch preserved; no resume or automatic retry')
    check_guards(launch)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required; no fallback')
    free, total = torch.cuda.mem_get_info()
    if free < 21 * 1024**3: raise RuntimeError('Need21GiB live free; preserve other jobs')
    torch.cuda.set_per_process_memory_fraction(19 * 1024**3 / total)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    write_json(launch / 'started.json', {'fold': fold, 'phase': 'source-only GPU preflight', 'target_access': False})
    sources = FOLDS[fold]['sources']
    records = {d: {s: read_dataset(d, s, DATA) for s in ('train', 'dev')} for d in sources}
    ready = json.loads((launch / 'readiness.json').read_text())
    if any(len(records[d][s]) != ready['sources'][d][s]['records'] for d in sources for s in ('train', 'dev')):
        raise ValueError('Readiness source count changed')
    proof = validate_source(launch, records); write_json(launch / 'source_validation.json', proof)
    # Reset every final model/sampler RNG after discarding the preflight weights.
    set_seed(1234)
    train = make_train_loader([FullAudioDataset(records[d]['train'], training=True) for d in sources],
                              batch_size=24, seed=1234, num_workers=8, prefetch_factor=2)
    dev = {d: make_eval_loader(FullAudioDataset(records[d]['dev']), num_workers=4, prefetch_factor=2) for d in sources}
    model = LastSSLFeatureGating().cuda(); count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != TRAINABLE_PARAMETERS: raise ValueError('Last-block parameter inventory changed')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    config = {'fold': fold, 'seed': 1234, 'run_name': RUN_FAMILY, 'model_name': 'SSLFeatureGating',
              'model_config': LAST_MODEL_CONFIG, 'frontend_config': model.feature_extractor.config.to_dict(),
              'sources': list(sources), 'target_domain': FOLDS[fold]['target'], 'target_access': False,
              'data_root': str(DATA), 'dataset_counts': {d: {s: len(v) for s, v in parts.items()} for d, parts in records.items()},
              'batch_size': 24, 'effective_batch': 24, 'accumulation_steps': 1, 'num_workers': 8, 'eval_workers': 4,
              'prefetch_factor': 2, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'optimizer': 'Adam',
              'scheduler': 'constant', 'maximum_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'train_max_seconds': 20,
              'eval_batch_size': 1, 'precision': 'BF16_autocast_FP32_pooling_losses_scores',
              'loss': 'weighted_CE_0.9_0.1_plus_CKA_coefficient1',
              'selection': 'strict_equal_domain_macro_source_dev_eer_ties_earlier', 'trainable_parameters': count,
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'initialization': 'fresh pinned official XLS-R plus fresh backend; source-check weights discarded',
              'source_sha256': json.loads((launch / 'source_guards.json').read_text()),
              'source_validation_sha256': checkpoint_digest(launch / 'source_validation.json'),
              'readiness_sha256': checkpoint_digest(launch / 'readiness.json'), 'gpu_memory_budget_gib': 19,
              'snapshot_policy': 'atomic15minute/end_training_phase snapshots; no automatic resume',
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'torch': str(torch.__version__), 'transformers': __import__('transformers').__version__,
              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name()}
    run.mkdir(parents=True, exist_ok=False); write_json(run / 'config.json', config)
    (run / 'training.log').hardlink_to(launch / 'console.log')
    print(f'START {RUN_FAMILY}/{fold}/1234 | block23 + backend | {count} trainable | batch24 BF16/FP32 | target unopened', flush=True)
    def snapshot(epoch, step, seen):
        save_checkpoint(run / 'recovery.pt', {'config': config, 'epoch': epoch, 'step': step, 'recordings_seen': seen,
            'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'kind': 'intermediate_training_snapshot_not_completion', 'exact_mid_epoch_resume_supported': False})
        print(f'SNAPSHOT epoch{epoch} step{step}; not a completion marker', flush=True)
    best, best_epoch, stale, best_scores = math.inf, 0, 0, None
    started = time.monotonic()
    for epoch in range(1, MAX_EPOCHS + 1):
        check_guards(launch)
        losses = train_epoch(model, train, optimizer, epoch, snapshot, 15 * 60)
        scored = score_sources(model, dev); metrics = source_metrics(scored)
        if metrics['macro_eer'] < best:
            best, best_epoch, stale = metrics['macro_eer'], epoch, 0
            best_scores = {d: s.copy() for d, (_, s) in scored.items()}; names = ('last.pt', 'best.pt')
        else: stale += 1; names = ('last.pt',)
        payload = {'config': config, 'fold': fold, 'epoch': epoch, 'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(), 'source_dev': metrics, 'macro_dev_eer': metrics['macro_eer'],
                   'train_losses': losses, 'best_macro_dev_eer': best, 'bad_epochs': stale}
        for name in names: save_checkpoint(run / name, payload)
        with (run / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch, 'losses': losses, 'source_dev': metrics,
                                     'best_epoch': best_epoch, 'patience': stale}) + '\n')
        print(f'EPOCH {epoch} macro_source_EER={100*metrics["macro_eer"]:.4f}% best={best_epoch} patience={stale}/{PATIENCE}', flush=True)
        if stale >= PATIENCE: break
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    model.cpu(); del model, optimizer, payload; gc.collect(); torch.cuda.empty_cache()
    fresh = LastSSLFeatureGating.from_frontend_config(config['frontend_config']).cuda().eval()
    fresh.load_state_dict(state['model_state_dict'], strict=True); reloaded = score_sources(fresh, dev)
    if source_metrics(reloaded) != state['source_dev'] or any(not (reloaded[d][1] == best_scores[d]).all() for d in sources):
        raise RuntimeError('Fresh selected-checkpoint full-source reload differs')
    hashes = {}
    for domain, (_, scores) in reloaded.items():
        path = run / f'source_dev_{domain}.csv'
        write_score_csv(path, ((utterance_id(p, domain, DATA), domain, y, float(s))
                              for (p, y), s in zip(records[domain]['dev'], scores)))
        hashes[path.name] = checkpoint_digest(path)
    write_json(run / 'source_calibration.json', calibrate_source_scores(reloaded)); check_guards(launch)
    write_json(run / 'training_complete.json', {'fold': fold, 'seed': 1234, 'best_epoch': best_epoch, 'final_epoch': epoch,
        'best_macro_source_dev_eer': best, 'best_checkpoint_sha256': checkpoint_digest(run / 'best.pt'),
        'source_scores_sha256': hashes, 'source_calibration_sha256': checkpoint_digest(run / 'source_calibration.json'),
        'selected_checkpoint_reloaded_and_source_verified': True, 'wall_seconds': time.monotonic() - started, 'target_access': False})
    print('COMPLETE: full source reload/calibration passed; no target evaluation', flush=True)


if __name__ == '__main__': main()
