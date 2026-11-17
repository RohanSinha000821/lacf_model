"""Discarded source-only F4 memory/reload check. Never starts final training."""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset
from audio_deepfake_detection.sota.wavlm_wa import WavLMDataset, collate_wavlm_batch
from audio_deepfake_detection.sota.wavlm_wa_last import LastBlockWavLMWA, PRETRAINED_PATH, PRETRAINED_REVISION
from scripts.wavlm.train import set_seed
from scripts.wavlm.last_block_support import ROOT

SOURCES = ('asv5', 'cfad', 'speechfake')
COUNTS = {'asv5': (182357, 140950), 'cfad': (38400, 14400), 'speechfake': (699608, 117463)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/drive/audio-deepfake-cache'))
    args = parser.parse_args()
    output = ROOT / 'outputs/wavlm_bs96/f4/preparation'
    output.mkdir(parents=True, exist_ok=True)
    record = output / 'source_validation.json'
    if record.exists() or (output / 'started.json').exists():
        raise FileExistsError('Previous preparation preserved; no automatic retry')
    if (ROOT / 'outputs/wavlm_bs96/f4/1234').exists():
        raise FileExistsError('Existing F4 run preserved')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native BF16 CUDA required')
    free, total = torch.cuda.mem_get_info(); budget = 14
    if free < (budget + 2) * 1024**3:
        raise RuntimeError('Need16GiB live GPU free for bounded source preparation; preserve jobs')
    torch.cuda.set_per_process_memory_fraction(budget * 1024**3 / total)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    set_seed(1234)
    assert FOLDS['f4']['sources'] == SOURCES
    records = {}; counts = {}
    for domain in SOURCES:
        for split, expected in zip(('train', 'dev'), COUNTS[domain]):
            rows = read_dataset(domain, split, args.data_root)
            if len(rows) != expected:
                raise ValueError('Canonical F4 source count changed: ' + domain + '/' + split)
            records[domain, split] = rows
        counts[domain] = dict(zip(('train', 'dev'), COUNTS[domain]))
    # Reuse established long SOURCE examples, not any F4 target example.
    ready = json.loads((ROOT / 'outputs/ssl_gating_last/f2/launch/readiness.json').read_text())
    train_rows = [records['asv5', 'train'][0], records['cfad', 'train'][0],
                  records['speechfake', 'train'][0], ready['long_train_records'][0]]
    if tuple(ready['long_train_records'][0]) not in records['speechfake', 'train']:
        raise ValueError('Long training stress example is not an F4 source-train member')
    long_row = tuple(ready['long_dev_record'])
    if long_row not in records['speechfake', 'dev']:
        raise ValueError('Long validation example is not an F4 source-dev member')
    started = {'phase': 'discarded source-only memory and reload checks', 'target_access': False,
               'training_started': False, 'gpu_allocator_budget_gib': budget}
    (output / 'started.json').write_text(json.dumps(started, indent=2) + '\n')
    model = LastBlockWavLMWA().cuda().train()
    inventory = model.adaptation_configuration()
    optimizer = torch.optim.Adam([
        {'params': [p for p in model.wavlm.parameters() if p.requires_grad], 'lr': 2e-5},
        {'params': [model.layer_weights, *model.classifier.parameters()], 'lr': 5e-3}])
    dataset = WavLMDataset(train_rows, training=True)
    waves, mask, labels = collate_wavlm_batch([dataset[i % len(dataset)] for i in range(96)])
    waves, mask, labels = waves.cuda(), mask.cuda(), labels.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([9., 1.], device='cuda'))
    original_layerdrop = model.wavlm.config.layerdrop
    model.wavlm.config.layerdrop = 0  # Discarded stress only: execute all12 blocks.
    active = set(); prefix_gradient = []
    hooks = [layer.register_forward_hook(lambda _, inputs, output, i=i: active.add(i))
             for i, layer in enumerate(model.wavlm.encoder.layers)]
    hooks.append(model.wavlm.encoder.layers[0].register_forward_hook(
        lambda _, inputs, output: prefix_gradient.append(output[0].requires_grad)))
    torch.cuda.reset_peak_memory_stats(); steps = []
    for _ in range(3):
        active.clear(); optimizer.zero_grad(set_to_none=True); tick = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(waves, mask); loss = criterion(logits, labels)
        loss.backward()
        if (len(active) != 12 or any(prefix_gradient)
                or any(p.grad is not None for p in model.parameters() if not p.requires_grad)
                or not all(p.grad is not None and torch.isfinite(p.grad).all()
                           for p in model.parameters() if p.requires_grad)):
            raise ValueError('Final-block freeze/gradient/source stress differs')
        optimizer.step(); torch.cuda.synchronize()
        steps.append({'seconds': time.monotonic() - tick, 'loss': float(loss.detach())})
    for hook in hooks: hook.remove()
    train_peak = torch.cuda.max_memory_allocated(); train_reserved = torch.cuda.max_memory_reserved()
    model.wavlm.config.layerdrop = original_layerdrop
    del optimizer, criterion, logits, loss, waves, mask, labels
    gc.collect(); torch.cuda.empty_cache(); model.eval()
    full_wave, full_mask, _ = WavLMDataset([long_row], training=False)[0]
    wave = full_wave[None].cuda(); mask = full_mask[None].cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model.spoof_score(model(wave, mask)).float().cpu()
    dev_peak = torch.cuda.max_memory_allocated(); dev_reserved = torch.cuda.max_memory_reserved()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.cpu(); del model; gc.collect(); torch.cuda.empty_cache()
    fresh = LastBlockWavLMWA().cuda().eval(); fresh.load_state_dict(state, strict=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual = fresh.spoof_score(fresh(wave, mask)).float().cpu()
    if not torch.equal(expected, actual) or not torch.isfinite(actual).all():
        raise ValueError('Full-length source checkpoint reload differs')
    code_paths = ['scripts/wavlm/train.py', 'scripts/wavlm/evaluate.py', 'scripts/wavlm/prepare_f4_last.py',
        'scripts/wavlm/last_block_support.py', 'src/audio_deepfake_detection/sota/wavlm_wa.py',
        'src/audio_deepfake_detection/sota/wavlm_wa_last.py', 'src/audio_deepfake_detection/data.py']
    report = dict(passed=True, fold='f4', seed=1234, batch_size=96, training_started=False,
        target_access=False, data_root=str(args.data_root.resolve()), source_counts=counts,
        pretrained_revision=PRETRAINED_REVISION, adaptation=inventory,
        pretrained_sha256={p.name: checkpoint_digest(p) for p in PRETRAINED_PATH.iterdir()
                           if p.name in ('pytorch_model.bin', 'model.safetensors', 'config.json')},
        code_sha256={p: checkpoint_digest(ROOT / p) for p in code_paths},
        gpu_allocator_budget_gib=budget, initial_free_gpu_bytes=free,
        train_shape=[96, 64000], train_peak_allocated_bytes=train_peak,
        train_peak_reserved_bytes=train_reserved, source_stress_steps=steps,
        stress_layerdrop_disabled_only_for_discarded_probe=True,
        frozen_prefix_no_gradient_graph=True, all12_blocks_executed=True,
        long_source_seconds=full_wave.numel()/16000,
        long_source_peak_allocated_bytes=dev_peak, long_source_peak_reserved_bytes=dev_reserved,
        fresh_long_source_reload_bit_exact=True, probe_weights_discarded=True,
        representative_process_budget_gib=math.ceil(max(train_reserved, dev_reserved)/1024**3+1.5),
        limitation='Measured batch96/4s and one long source recording, not an exhaustive duration/memory audit')
    record.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print('PREPARATION COMPLETE; final F4 training NOT started', flush=True)


if __name__ == '__main__':
    main()
