"""Read-only preparation gate and selected-source finalization for WavLM F4."""
import json
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.metrics import calibrate_source_scores, compute_eer, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, save_score_manifest
from audio_deepfake_detection.sota.wavlm_wa_last import LastBlockWavLMWA, PRETRAINED_PATH, PRETRAINED_REVISION

ROOT = Path(__file__).resolve().parents[2]


def verify_preparation(path, data_root):
    report = json.loads(Path(path).read_text())
    if (report.get('passed') is not True or report.get('training_started') is not False
            or report.get('fold') != 'f4' or report.get('seed') != 1234 or report.get('batch_size') != 96
            or report.get('pretrained_revision') != PRETRAINED_REVISION
            or Path(report['data_root']).resolve() != Path(data_root).resolve()
            or report.get('probe_weights_discarded') is not True
            or report.get('fresh_long_source_reload_bit_exact') is not True):
        raise ValueError('Matching source-only last-block F4 preparation required')
    for name, digest in report['code_sha256'].items():
        if checkpoint_digest(ROOT / name) != digest:
            raise ValueError('Prepared F4 implementation changed: ' + name)
    for name, digest in report['pretrained_sha256'].items():
        if checkpoint_digest(PRETRAINED_PATH / name) != digest:
            raise ValueError('Prepared pretrained weights changed: ' + name)
    return report


def configure_runtime(report):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native BF16 CUDA required')
    free, total = torch.cuda.mem_get_info()
    budget = report['gpu_allocator_budget_gib']
    if free < (budget + 2) * 1024**3:
        raise RuntimeError('Insufficient live GPU headroom; preserve other jobs')
    torch.cuda.set_per_process_memory_fraction(budget * 1024**3 / total)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False


def source_scores(model, loader, records, domain, data_root, device):
    from scripts.wavlm.evaluate import score_rows
    rows = list(score_rows(model, loader, records, domain, data_root, device))
    if len(rows) != len(records):
        raise ValueError('Incomplete source scoring')
    return compute_eer([r[2] for r in rows], [r[3] for r in rows]), {r[0]: r[3] for r in rows}


def finalize_sources(output, dev_loaders, dev_records, data_root, device, expected, selected_scores):
    """Reload selected full model and export every source score, without target access."""
    from scripts.wavlm.evaluate import score_rows
    state = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model = LastBlockWavLMWA().to(device)
    model.load_state_dict(state['model_state_dict'], strict=True)
    inputs, hashes, by_domain = {}, {}, {}
    for domain in FOLDS['f4']['sources']:
        path = output / f'source_dev_{domain}.csv'
        write_score_csv(path, score_rows(model, dev_loaders[domain], dev_records[domain], domain, data_root, device))
        labels = canonical_score_labels(domain, 'dev', data_root, records=dev_records[domain])
        inputs[domain] = read_score_csv(path, domain, expected_labels=labels)
        ordered = [selected_scores[domain][key] for key in sorted(labels)]
        if not np.array_equal(inputs[domain][1], ordered):
            raise ValueError('Fresh selected-checkpoint source scores differ bit-for-bit')
        hashes[path.name] = checkpoint_digest(path)
    calibration = calibrate_source_scores(inputs)
    by_domain = calibration['source_dev']['eer_by_domain']
    if by_domain != state['dev_eers'] or sum(by_domain.values()) / len(by_domain) != expected:
        raise ValueError('Selected F4 full-source reload differs; no completion emitted')
    calibration_path = output / 'source_calibration.json'
    calibration_path.write_text(json.dumps(calibration, indent=2) + '\n')
    return {'selected_checkpoint_reloaded_and_source_verified': True,
        'full_source_scores_bit_exact': True,
        'best_checkpoint_sha256': checkpoint_digest(output / 'best.pt'),
        'source_scores_sha256': hashes, 'source_calibration_sha256': checkpoint_digest(calibration_path),
        'adaptation': state['config']['adaptation'], 'target_access': False}


def register_source_exports(output, completion):
    if (output / 'score_manifest.json').exists():
        raise FileExistsError('Prior source manifest preserved')
    if completion.get('full_source_scores_bit_exact') is not True:
        raise ValueError('Verified source scores required before manifest creation')
    for name, digest in completion['source_scores_sha256'].items():
        if checkpoint_digest(output / name) != digest:
            raise ValueError('Fresh source export changed before registration')
    fold = FOLDS['f4']
    manifest = {'model': output.parent.parent.name, 'fold': 'f4', 'seed': 1234,
        'checkpoint_sha256': completion['best_checkpoint_sha256'],
        'training_completion_sha256': checkpoint_digest(output / 'training_complete.json'),
        'splits': {**{f'source_dev_{d}.csv': {'dataset': d, 'split': 'dev'} for d in fold['sources']},
                   'target_scores.csv': {'dataset': fold['target'], 'split': fold['target_split']}},
        'score_sha256': completion['source_scores_sha256'], 'adaptation': completion['adaptation'],
        'full_source_scores_bit_exact': True}
    save_score_manifest(output, manifest)
    load_score_manifest(output, 'f4', 1234, completion['best_checkpoint_sha256'])
