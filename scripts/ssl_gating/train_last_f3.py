"""SSL-last F3 with replayable draws/crops and explicit, checkpointed recovery."""

import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import time

import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, utterance_id
from audio_deepfake_detection.sota.ssl_gating import FullAudioDataset, collate_audio, make_eval_loader
from audio_deepfake_detection.sota.ssl_gating_last import (
    LAST_MODEL_CONFIG, RUN_FAMILY, TRAINABLE_PARAMETERS, LastSSLFeatureGating,
)
from audio_deepfake_detection.sota.ssl_gating_partial import detection_losses, partial_scores
from audio_deepfake_detection.sota.ssl_gating_resume import (
    EpochPlan, append_epoch, capture_rng, durable_save, restore_rng,
)
from scripts.ssl_gating.train import LEARNING_RATE, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, set_seed, write_json
from scripts.ssl_gating.train_last_fold import check_guards
from scripts.ssl_gating.train_partial import score_sources, source_metrics


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/ssl_gating_last/f3'
RUN, LAUNCH = BASE / '1234', BASE / 'launch'
DATA = Path('/mnt/drive/audio-deepfake-cache')
SOURCES = FOLDS['f3']['sources']


def optimizer_for(model):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def update(model, optimizer, waves, labels):
    optimizer.zero_grad(set_to_none=True)
    loss, ce, cka = detection_losses(model, waves, labels)
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; stop without fallback')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('Nonfinite gradient; update refused')
    optimizer.step()
    return torch.stack((loss.detach(), ce.detach(), cka.detach())).double()


def validate_source(ready):
    """Same batch/precision, smaller allocator; compare an actual recovered update."""
    check_guards(LAUNCH); set_seed(1234)
    model = LastSSLFeatureGating().cuda().train(); optimizer = optimizer_for(model)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != TRAINABLE_PARAMETERS:
        raise ValueError('Trainable scope changed')
    dataset = FullAudioDataset(ready['long_train_records'], training=True)
    waves, labels = collate_audio([dataset[i % len(dataset)] for i in range(24)])
    if waves.shape != (24, 320000): raise ValueError('Stress must include full20s × batch24')
    waves, labels = waves.cuda(), labels.cuda()
    drop = model.feature_extractor.config.layerdrop
    seed = next(s for s in range(10000) if (torch.rand(24, generator=torch.Generator().manual_seed(s)) >= drop).all())
    active = set(); hooks = [layer.register_forward_hook(lambda _, args, out, i=i: active.add(i))
                            for i, layer in enumerate(model.feature_extractor.encoder.layers)]
    torch.cuda.reset_peak_memory_stats(); steps = []
    for index in range(4):
        torch.manual_seed(seed); active.clear(); tick = time.monotonic()
        losses = update(model, optimizer, waves, labels); torch.cuda.synchronize()
        if len(active) != 24 or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('Stress did not execute/freeze the prescribed layers')
        steps.append({'seconds': time.monotonic() - tick, 'losses': losses.cpu().tolist(), 'executed_layers': len(active)})
        print('STRESS', index + 1, steps[-1], flush=True)
    for hook in hooks: hook.remove()
    proof_path = LAUNCH / 'recovery_probe.pt'
    durable_save(proof_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                             'rng': capture_rng(), 'frontend': model.feature_extractor.config.to_dict()})
    # Continue once, then reconstruct from the durable state and repeat that update.
    expected_loss = update(model, optimizer, waves, labels).cpu()
    expected_rng = capture_rng()
    expected = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    expected_buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    model.cpu(); del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    saved = torch.load(proof_path, map_location='cpu', weights_only=True)
    model = LastSSLFeatureGating.from_frontend_config(saved['frontend']).cuda().train()
    model.load_state_dict(saved['model'], strict=True); optimizer = optimizer_for(model)
    optimizer.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
    actual_loss = update(model, optimizer, waves, labels).cpu()
    actual_rng = capture_rng()
    differences = {n: float((p.detach().cpu() - expected[n]).abs().max())
                   for n, p in model.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(), expected[n])}
    write_json(LAUNCH / 'recovery_probe_comparison.json', {
        'expected_losses': expected_loss.tolist(), 'actual_losses': actual_loss.tolist(),
        'mismatched_parameter_count': len(differences), 'maximum_parameter_absolute_error': max(differences.values(), default=0),
        'parameter_absolute_errors': differences,
        'CPU_rng_equal_after_update': torch.equal(expected_rng['torch'], actual_rng['torch']),
        'CUDA_rng_equal_after_update': all(torch.equal(a, b) for a, b in zip(expected_rng['cuda'], actual_rng['cuda']))})
    if (not torch.equal(actual_loss, expected_loss)
            or any(not torch.equal(p.detach().cpu(), expected[n]) for n, p in model.named_parameters() if p.requires_grad)
            or any(not torch.equal(b.detach().cpu(), expected_buffers[n]) for n, b in model.named_buffers())):
        raise RuntimeError('Actual source optimizer/RNG recovery differs')
    del waves, labels; optimizer.zero_grad(set_to_none=True); model.eval()
    wave, _ = FullAudioDataset([ready['long_dev_record']])[0]
    expected_score = partial_scores(model, wave[None].cuda()).cpu()
    state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    frontend = model.feature_extractor.config.to_dict()
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    model.cpu(); del model, optimizer, saved; gc.collect(); torch.cuda.empty_cache()
    fresh = LastSSLFeatureGating.from_frontend_config(frontend).cuda().eval(); fresh.load_state_dict(state, strict=True)
    if not torch.equal(expected_score, partial_scores(fresh, wave[None].cuda()).cpu()):
        raise RuntimeError('Full native source score reload differs')
    result = {'passed': True, 'fold': 'f3', 'model_config': LAST_MODEL_CONFIG, 'trainable_parameters': TRAINABLE_PARAMETERS,
        'steps': steps, 'actual_source_update_recovery_bit_exact': True, 'full_source_score_reload_bit_exact': True,
        'full_source_seconds': wave.numel() / 16000, 'peak_allocated_bytes': peak, 'peak_reserved_bytes': reserved,
        'allocator_cap_gib': 13, 'allocator': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        'probe_weights_discarded': True, 'target_access': False,
        'source_guards_sha256': checkpoint_digest(LAUNCH / 'source_guards.json'),
        'readiness_sha256': checkpoint_digest(LAUNCH / 'readiness.json'), 'probe_sha256': checkpoint_digest(proof_path)}
    del fresh, state; gc.collect(); torch.cuda.empty_cache(); check_guards(LAUNCH)
    return result


def snapshot(model, optimizer, config, plan, epoch, stage, totals, best, stale):
    durable_save(RUN / 'recovery.pt', {'format_version': 1, 'config': config,
        'config_sha256': checkpoint_digest(RUN / 'config.json'), 'epoch': epoch, 'stage': stage,
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': capture_rng(),
        'plan': plan.state_dict(), 'loss_totals': totals.cpu(), 'best': best, 'patience': stale})


def verify_recovery(state, config):
    if (state['format_version'] != 1 or state['config_sha256'] != checkpoint_digest(RUN / 'config.json')
            or json.loads(json.dumps(state['config'])) != config or config['fold'] != 'f3' or config['seed'] != 1234
            or config['model_config'] != LAST_MODEL_CONFIG or config['trainable_parameters'] != TRAINABLE_PARAMETERS
            or state['stage'] not in ('train', 'source_dev', 'epoch_complete') or not 1 <= state['epoch'] <= 10):
        raise ValueError('Recovery checkpoint/config/recipe/stage mismatch')
    if (state['best'] is None and state['patience'] != 0) or (state['best'] is not None and not 1 <= state['best']['epoch'] <= state['epoch']):
        raise ValueError('Recovery selection state mismatch')
    if (not 0 <= state['patience'] <= PATIENCE
            or config['sources'] != list(SOURCES) or config['target_access'] is not False
            or config['batch_size'] != 24 or config['precision'] != 'BF16_autocast_FP32_pooling_losses_scores'
            or config['learning_rate'] != LEARNING_RATE or config['weight_decay'] != WEIGHT_DECAY
            or (state['stage'] != 'train' and state['plan']['offset'] != len(state['plan']['order']))
            or (state['stage'] == 'epoch_complete' and state['best'] is None)):
        raise ValueError('Recovery recipe or phase progress mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Explicitly recover this incomplete F3 run; never automatic')
    args = parser.parse_args()
    if args.validate_only and args.resume: parser.error('Choose validation or recovery')
    if (RUN / 'training_complete.json').exists(): raise FileExistsError('Completed run preserved')
    check_guards(LAUNCH)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native BF16 CUDA required')
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') != 'max_split_size_mb:128':
        raise RuntimeError('Use the source-validated standard allocator split limit')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Use the validated deterministic cuBLAS workspace setting')
    free, total = torch.cuda.mem_get_info()
    if free < 15 * 1024**3: raise RuntimeError('Need15GiB live free; preserve F2')
    torch.cuda.set_per_process_memory_fraction(13 * 1024**3 / total)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; set_seed(1234)
    torch.use_deterministic_algorithms(True)
    ready = json.loads((LAUNCH / 'readiness.json').read_text())
    if ready['passed'] is not True or ready['fold'] != 'f3' or set(ready['sources']) != set(SOURCES):
        raise ValueError('Canonical F3 source readiness required')
    if args.validate_only:
        if (LAUNCH / 'source_validation.json').exists() or RUN.exists(): raise FileExistsError('Prior validation/run preserved')
        proof = validate_source(ready); write_json(LAUNCH / 'source_validation.json', proof)
        print('SOURCE VALIDATION PASSED; all probe weights discarded', flush=True); return
    proof = json.loads((LAUNCH / 'source_validation.json').read_text())
    if (not proof['passed'] or not proof['actual_source_update_recovery_bit_exact']
            or not proof['full_source_score_reload_bit_exact'] or not proof['probe_weights_discarded']
            or proof['source_guards_sha256'] != checkpoint_digest(LAUNCH / 'source_guards.json')
            or proof['readiness_sha256'] != checkpoint_digest(LAUNCH / 'readiness.json')):
        raise ValueError('Matching GPU validation required')
    if args.resume:
        if not RUN.is_dir(): raise FileNotFoundError('No F3 run to resume')
    else:
        RUN.mkdir(parents=True, exist_ok=False)
    with (RUN / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = {d: {s: read_dataset(d, s, DATA) for s in ('train', 'dev')} for d in SOURCES}
        if any(len(records[d][s]) != ready['sources'][d][s]['records'] for d in SOURCES for s in ('train', 'dev')):
            raise ValueError('Source count differs from readiness')
        plan = EpochPlan([FullAudioDataset(records[d]['train'], training=True) for d in SOURCES])
        dev = {d: make_eval_loader(FullAudioDataset(records[d]['dev']), num_workers=4, prefetch_factor=2) for d in SOURCES}
        for i, loader in enumerate(dev.values()): loader.generator = torch.Generator().manual_seed(1334 + i)
        set_seed(1234)
        if args.resume:
            config = json.loads((RUN / 'config.json').read_text())
            saved = torch.load(RUN / 'recovery.pt', map_location='cpu', weights_only=True); verify_recovery(saved, config)
            if config['source_sha256'] != json.loads((LAUNCH / 'source_guards.json').read_text()):
                raise ValueError('Recovery runtime changed')
            # Preserve an interrupted temporary write; the durable previous state remains authoritative.
            for temporary in list(RUN.glob('*.tmp')):
                temporary.rename(RUN / f'interrupted_{time.time_ns()}_{temporary.name}')
            model = LastSSLFeatureGating.from_frontend_config(config['frontend_config']).cuda()
            model.load_state_dict(saved['model'], strict=True); optimizer = optimizer_for(model)
            optimizer.load_state_dict(saved['optimizer']); plan.load_state_dict(saved['plan'])
            epoch, stage, totals, best, stale = saved['epoch'], saved['stage'], saved['loss_totals'].cuda(), saved['best'], saved['patience']
            restore_rng(saved['rng']); del saved
            with (RUN / 'resume_events.jsonl').open('a') as stream:
                stream.write(json.dumps({'unix_time': time.time(), 'epoch': epoch, 'stage': stage,
                    'next_draw': plan.offset, 'checkpoint_sha256': checkpoint_digest(RUN / 'recovery.pt')}) + '\n')
        else:
            model = LastSSLFeatureGating().cuda(); optimizer = optimizer_for(model)
            config = {'fold': 'f3', 'seed': 1234, 'run_name': RUN_FAMILY, 'model_name': 'SSLFeatureGating',
                'model_config': LAST_MODEL_CONFIG, 'frontend_config': model.feature_extractor.config.to_dict(),
                'sources': list(SOURCES), 'target_domain': FOLDS['f3']['target'], 'target_access': False,
                'dataset_counts': {d: {s: len(r) for s, r in parts.items()} for d, parts in records.items()},
                'batch_size': 24, 'effective_batch': 24, 'accumulation_steps': 1, 'num_workers': 8, 'eval_workers': 4,
                'prefetch_factor': 2, 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
                'optimizer': 'Adam', 'scheduler': 'constant', 'maximum_epochs': MAX_EPOCHS, 'patience': PATIENCE,
                'precision': 'BF16_autocast_FP32_pooling_losses_scores', 'train_max_seconds': 20, 'eval_batch_size': 1,
                'loss': 'weighted_CE_0.9_0.1_plus_CKA_coefficient1',
                'selection': 'strict_equal_domain_macro_source_dev_eer_ties_earlier',
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'source_sha256': json.loads((LAUNCH / 'source_guards.json').read_text()),
                'source_validation_sha256': checkpoint_digest(LAUNCH / 'source_validation.json'),
                'readiness_sha256': checkpoint_digest(LAUNCH / 'readiness.json'), 'gpu_memory_budget_gib': 13,
                'allocator': os.environ['PYTORCH_CUDA_ALLOC_CONF'],
                'deterministic_algorithms': True, 'cublas_workspace': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                'recovery': 'every15min/end-train/end-epoch; explicit --resume; replay remaining draws/crops/optimizer/model RNG',
                'crop_rng': 'one seed per sampled draw, independent of worker; same uniform20s crop law, different RNG assignment from F1/F2',
                'initialization': 'fresh pinned official XLS-R and fresh backend; probe weights discarded',
                'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'transformers': __import__('transformers').__version__, 'data_root': str(DATA)}
            if config['trainable_parameters'] != TRAINABLE_PARAMETERS: raise ValueError('Trainable scope changed')
            write_json(RUN / 'config.json', config)
            epoch, stage, totals, best, stale = 1, 'train', torch.zeros(3, dtype=torch.float64, device='cuda'), None, 0
            plan.start_epoch(); snapshot(model, optimizer, config, plan, epoch, stage, totals, best, stale)
        print(f'START SSL-last F3 epoch{epoch} stage={stage} next_draw={plan.offset} | batch24 BF16/FP32 | resumable', flush=True)
        while True:
            check_guards(LAUNCH)
            if stage == 'epoch_complete':
                if stale >= PATIENCE or epoch >= MAX_EPOCHS: break
                epoch += 1; plan.start_epoch(); totals.zero_(); stage = 'train'
                snapshot(model, optimizer, config, plan, epoch, stage, totals, best, stale)
            if stage == 'train':
                model.train(); tick = last_log = last_save = time.monotonic(); start_offset = plan.offset
                for waves, labels in plan.loader():
                    values = update(model, optimizer, waves.cuda(non_blocking=True), labels.cuda(non_blocking=True))
                    plan.offset += len(labels); totals += values * len(labels); now = time.monotonic()
                    if now - last_log >= 30 or plan.offset == len(plan.data):
                        rate = (plan.offset - start_offset) / max(now - tick, 1e-6)
                        print(f'epoch={epoch} recordings={plan.offset}/{len(plan.data)} loss={float(totals[0]/plan.offset):.6f} '
                              f'audio/s={rate:.2f} ETAmin={(len(plan.data)-plan.offset)/max(rate,1e-6)/60:.1f}', flush=True)
                        last_log = now
                    if now - last_save >= 900:
                        snapshot(model, optimizer, config, plan, epoch, 'train', totals, best, stale); last_save = time.monotonic()
                        print('RECOVERY SNAPSHOT saved after optimizer update', flush=True)
                stage = 'source_dev'; snapshot(model, optimizer, config, plan, epoch, stage, totals, best, stale)
            scored = score_sources(model, dev); metrics = source_metrics(scored)
            if best is None or metrics['macro_eer'] < best['eer']:
                best = {'epoch': epoch, 'eer': metrics['macro_eer'], 'metrics': metrics,
                        'model': {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                        'scores': {d: torch.from_numpy(s.copy()) for d, (_, s) in scored.items()}}
                stale = 0
                durable_save(RUN / 'best.pt', {'config': config, 'fold': 'f3', 'epoch': epoch,
                    'model_state_dict': best['model'], 'source_dev': metrics, 'macro_dev_eer': best['eer']})
            else: stale += 1
            row = {'epoch': epoch, 'losses': dict(zip(('total', 'ce', 'cka'), (totals/plan.offset).tolist())),
                   'source_dev': metrics, 'best_epoch': best['epoch'], 'patience': stale}
            append_epoch(RUN / 'history.jsonl', row); stage = 'epoch_complete'
            snapshot(model, optimizer, config, plan, epoch, stage, totals, best, stale)
            print(f'EPOCH {epoch} source_EER={100*metrics["macro_eer"]:.4f}% best={best["epoch"]} patience={stale}/{PATIENCE}', flush=True)
            if stale >= PATIENCE or epoch >= MAX_EPOCHS: break
        model.cpu(); del model, optimizer; gc.collect(); torch.cuda.empty_cache()
        fresh = LastSSLFeatureGating.from_frontend_config(config['frontend_config']).cuda().eval()
        fresh.load_state_dict(best['model'], strict=True); reloaded = score_sources(fresh, dev)
        if source_metrics(reloaded) != best['metrics'] or any(not torch.equal(torch.from_numpy(reloaded[d][1]), best['scores'][d]) for d in SOURCES):
            raise RuntimeError('Selected full-source reload differs')
        selected = torch.load(RUN / 'best.pt', map_location='cpu', weights_only=True)
        if (selected['epoch'] != best['epoch'] or selected['source_dev'] != best['metrics']
                or any(not torch.equal(v, selected['model_state_dict'][n]) for n, v in best['model'].items())):
            raise ValueError('Selected checkpoint differs from durable recovery selection state')
        del selected
        hashes = {}
        for d, (_, scores) in reloaded.items():
            path = RUN / f'source_dev_{d}.csv'
            if path.exists():
                _, old = read_score_csv(path, d)
                if not (old == scores).all(): raise ValueError('Existing source export differs')
            else:
                write_score_csv(path, ((utterance_id(p, d, DATA), d, y, float(s)) for (p, y), s in zip(records[d]['dev'], scores)))
            hashes[path.name] = checkpoint_digest(path)
        calibration = calibrate_source_scores(reloaded); path = RUN / 'source_calibration.json'
        if path.exists():
            if json.loads(path.read_text()) != calibration: raise ValueError('Existing calibration differs')
        else: write_json(path, calibration)
        check_guards(LAUNCH)
        write_json(RUN / 'training_complete.json', {'fold': 'f3', 'seed': 1234, 'best_epoch': best['epoch'],
            'final_epoch': epoch, 'best_macro_source_dev_eer': best['eer'], 'best_checkpoint_sha256': checkpoint_digest(RUN / 'best.pt'),
            'source_scores_sha256': hashes, 'source_calibration_sha256': checkpoint_digest(path),
            'selected_checkpoint_reloaded_and_source_verified': True, 'target_access': False})
        print('COMPLETE: selected full source reload/export/calibration verified; target unopened', flush=True)


if __name__ == '__main__': main()
