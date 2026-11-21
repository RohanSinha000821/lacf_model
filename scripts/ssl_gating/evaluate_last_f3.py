"""Native F3 evaluation of the completed SSL-last model; source gates first."""

import fcntl
import json
import copy
import os
import subprocess
import time
from pathlib import Path

import torch

from audio_deepfake_detection.metrics import evaluate_transfer, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, read_dataset, save_score_manifest,
    verify_completed_run, verify_source_macro,
)
from audio_deepfake_detection.sota.ssl_gating import FullAudioDataset, make_eval_loader
from audio_deepfake_detection.sota.ssl_gating_last import (
    LAST_MODEL_CONFIG, RUN_FAMILY, TRAINABLE_PARAMETERS, LastSSLFeatureGating,
)
from scripts.ssl_gating.evaluate_partial import (
    score_rows, verified_sources, verify_run_config as _strict_verify_run_config, verify_source_parity,
)
from scripts.ssl_gating.train import set_seed, write_json


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs' / RUN_FAMILY / 'f3/1234'
LAUNCH = ROOT / 'outputs' / RUN_FAMILY / 'f3/evaluation'
DATA = Path('/mnt/drive/audio-deepfake-cache')
PRODUCER_EXIT = 'resume_exit_1794950472206927626.json'


def check_guards():
    for name, digest in json.loads((LAUNCH / 'source_guards.json').read_text()).items():
        if checkpoint_digest(ROOT / name) != digest:
            raise RuntimeError('Evaluation/preserved source changed: ' + name)


def verify_run_config(config, stored_config, current_guards):
    """Accept only the previously verified Transformers absent/null serialization key."""
    normalized = copy.deepcopy(config)
    frontend = normalized.get('frontend_config', {})
    expected = stored_config.get('frontend_config', {})
    if ('attn_implementation' not in expected and 'attn_implementation' in frontend
            and frontend['attn_implementation'] is None):
        del frontend['attn_implementation']
    _strict_verify_run_config(normalized, stored_config, current_guards)


def wait_for_headroom():
    """CPU-only queue; preserve running trainers and open no target while waiting."""
    while True:
        check_guards()
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip().splitlines()
        if len(lines) != 1:
            raise RuntimeError('Expected the prepared single GPU')
        free_mib = int(lines[0])
        # Leave space for this process's CUDA context before the actual 10-GiB gate.
        if free_mib >= 11 * 1024:
            return
        write_json(LAUNCH / 'queue_status.json', {'status': 'waiting_for_gpu_memory',
            'free_gpu_mib': free_mib, 'required_free_gpu_mib': 11 * 1024,
            'target_access': False, 'unix_time': time.time()})
        print(f'WAITING: {free_mib} MiB GPU free; need 11264 MiB before CUDA initialization; other jobs preserved', flush=True)
        time.sleep(30)


def verify_producer(producer):
    if (producer.get('verified_complete') is not True
            or producer.get('trainer_exit') != 0 or producer.get('logger_exit') != 0
            or producer.get('target_access') is not False):
        raise ValueError('F3 training producer has not completed successfully')


def verify_checkpoint(checkpoint, completion):
    c = checkpoint['config']; fold = FOLDS['f3']
    if (checkpoint['fold'] != 'f3' or c['fold'] != 'f3' or c['seed'] != 1234
            or c['run_name'] != RUN_FAMILY or c['model_name'] != 'SSLFeatureGating'
            or c['model_config'] != LAST_MODEL_CONFIG
            or c['precision'] != 'BF16_autocast_FP32_pooling_losses_scores'
            or c['trainable_parameters'] != TRAINABLE_PARAMETERS or c['total_parameters'] != 318083586
            or tuple(c['sources']) != fold['sources'] or c['target_domain'] != fold['target']
            or c['target_access'] is not False or completion.get('target_access') is not False
            or c['train_max_seconds'] != 20 or c['eval_batch_size'] != 1 or c['batch_size'] != 24
            or c['learning_rate'] != 3e-6 or c['weight_decay'] != 1e-4 or c['optimizer'] != 'Adam'
            or c['scheduler'] != 'constant' or c['maximum_epochs'] != 10 or c['patience'] != 3
            or c.get('torch') != str(torch.__version__) or c.get('cuda') != torch.version.cuda
            or c.get('transformers') != __import__('transformers').__version__
            or c.get('deterministic_algorithms') is not True
            or c.get('cublas_workspace') != ':4096:8'
            or c.get('allocator') != 'max_split_size_mb:128'
            or c['loss'] != 'weighted_CE_0.9_0.1_plus_CKA_coefficient1'
            or c['selection'] != 'strict_equal_domain_macro_source_dev_eer_ties_earlier'
            or checkpoint['epoch'] != completion['best_epoch']
            or completion.get('selected_checkpoint_reloaded_and_source_verified') is not True):
        raise ValueError('SSL-last checkpoint identity, recipe or source reload differs')
    verify_source_macro(checkpoint['macro_dev_eer'], completion['best_macro_source_dev_eer'])


def main():
    with (LAUNCH / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('metrics.json', 'metrics.json.tmp', 'target_scores.csv', 'target_scores.csv.tmp',
                     'score_manifest.json', 'source_validation.json', 'evaluation_complete.json'):
            if (RUN / name).exists(): raise FileExistsError('Prior evaluation preserved: ' + name)
        if (LAUNCH / 'started.json').exists(): raise FileExistsError('Prior launch preserved; no automatic retry')
        check_guards()
        bound = json.loads((LAUNCH / 'input_sha256.json').read_text())
        for name, digest in bound.items():
            if checkpoint_digest(RUN / name) != digest: raise ValueError('Launch-bound training artifact changed')
        producer = json.loads((RUN.parent / 'launch' / PRODUCER_EXIT).read_text())
        verify_producer(producer)
        completion, digest = verify_completed_run(RUN, 'f3', 1234)
        if digest != completion['best_checkpoint_sha256']: raise ValueError('Selected checkpoint changed')
        state = torch.load(RUN / 'best.pt', map_location='cpu', weights_only=True)
        verify_checkpoint(state, completion); config = state['config']
        current = {name: checkpoint_digest(ROOT / name) for name in config['source_sha256']}
        verify_run_config(config, json.loads((RUN / 'config.json').read_text()), current)
        for name, key in (('readiness.json', 'readiness_sha256'), ('source_validation.json', 'source_validation_sha256')):
            if checkpoint_digest(RUN.parent / 'launch' / name) != config[key]:
                raise ValueError('Training source preparation evidence changed')
        sources, records = verified_sources(RUN, completion, config, DATA)
        print('SOURCE GATE: checkpoint/config/canonical exports/frozen calibration passed', flush=True)
        wait_for_headroom()
        # Recheck launch-bound inputs after any resource wait.
        for name, expected in bound.items():
            if checkpoint_digest(RUN / name) != expected:
                raise ValueError('Launch-bound artifact changed while waiting')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError('Native CUDA BF16 required; no fallback')
        free, total = torch.cuda.mem_get_info()
        if free < 10 * 1024**3: raise RuntimeError('Need10GiB live free; preserve other jobs')
        torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / total)
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != config['cublas_workspace']:
            raise RuntimeError('CUBLAS workspace differs from F3 training')
        if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') != config['allocator']:
            raise RuntimeError('Allocator differs from F3 training')
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; set_seed(1234)
        write_json(LAUNCH / 'queue_status.json', {'status': 'source_gpu_verification', 'target_access': False})
        write_json(LAUNCH / 'started.json', {'phase': 'fresh source parity', 'target_access': False})
        model = LastSSLFeatureGating.from_frontend_config(copy.deepcopy(config['frontend_config']))
        model.load_state_dict(state['model_state_dict'], strict=True); del state
        model.cuda().eval(); torch.cuda.reset_peak_memory_stats()
        parity = verify_source_parity(model, records, RUN, DATA, torch.device('cuda'))
        ready = json.loads((RUN.parent / 'launch/readiness.json').read_text())
        # Include the preserved long SpeechFake source recording, beyond first/tail checks.
        parity += verify_source_parity(model, {'speechfake': [ready['long_dev_record']]}, RUN, DATA, torch.device('cuda'))
        check_guards()
        write_json(RUN / 'source_validation.json', {'checkpoint_sha256': digest,
            'source_calibration_sha256': completion['source_calibration_sha256'],
            'source_scores_sha256': completion['source_scores_sha256'],
            'full_source_reload_verified_during_training': True, 'bounded_fresh_reload_parity': parity,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'gpu_allocator_budget_gib': 8, 'config_serialization_compatibility': 'only absent vs null frontend attn_implementation', 'target_access': False})
        fold = FOLDS['f3']
        manifest = {'model': RUN_FAMILY, 'fold': 'f3', 'seed': 1234, 'checkpoint_sha256': digest,
            'training_completion_sha256': checkpoint_digest(RUN / 'training_complete.json'),
            'splits': {**{f'source_dev_{d}.csv': {'dataset': d, 'split': 'dev'} for d in fold['sources']},
                       'target_scores.csv': {'dataset': fold['target'], 'split': fold['target_split']}},
            'score_sha256': dict(completion['source_scores_sha256'])}
        save_score_manifest(RUN, manifest)
        write_json(LAUNCH / 'queue_status.json', {'status': 'target_scoring', 'target_access': True})
        print('SOURCE GPU PARITY PASSED: first/tail plus long full source recording; opening ASVspoof 5 evaluation partition', flush=True)
        target = read_dataset(fold['target'], fold['target_split'], DATA)
        expected = canonical_score_labels(fold['target'], fold['target_split'], DATA, records=target)
        print(f'TARGET ASVspoof 5 eval: {len(target):,} full recordings; batch1 BF16/FP32', flush=True)
        loader = make_eval_loader(FullAudioDataset(target), num_workers=4, prefetch_factor=2)
        write_score_csv(RUN / 'target_scores.csv', score_rows(model, loader, target, fold['target'], DATA, torch.device('cuda')))
        labels, scores = read_score_csv(RUN / 'target_scores.csv', fold['target'], expected_labels=expected)
        manifest['score_sha256']['target_scores.csv'] = checkpoint_digest(RUN / 'target_scores.csv')
        save_score_manifest(RUN, manifest); del model, loader; torch.cuda.empty_cache()
        print('Full target scores verified; computing shared metrics and1000 conditional bootstrap intervals', flush=True)
        report = evaluate_transfer(sources, labels, scores, bootstrap_resamples=1000)
        report.update({'model': RUN_FAMILY, 'fold': 'f3', 'seed': 1234, 'sources': list(fold['sources']),
            'target_domain': fold['target'], 'target_split': fold['target_split'], 'checkpoint_sha256': digest,
            'checkpoint_epoch': completion['best_epoch'], 'score_sha256': manifest['score_sha256'],
            'evaluator_sha256': checkpoint_digest(Path(__file__)), 'precision': config['precision'],
            'input_policy': LAST_MODEL_CONFIG['input_policy'], 'train_input_policy': LAST_MODEL_CONFIG['train_input_policy'],
            'train_max_seconds': 20, 'eval_batch_size': 1, 'num_workers': 4, 'prefetch_factor': 2,
            'gpu_allocator_budget_gib': 8, 'source_data_root': str(DATA), 'target_data_root': str(DATA),
            'source_exports_reused_after_completed_full_reload_and_hash_verification': True,
            'comparison_wide_freeze_confirmed_by_operator': False})
        check_guards()
        write_json(RUN / 'metrics.json.tmp', report); (RUN / 'metrics.json.tmp').replace(RUN / 'metrics.json')
        write_json(RUN / 'evaluation_complete.json', {'fold': 'f3', 'seed': 1234,
            'checkpoint_sha256': digest, 'metrics_sha256': checkpoint_digest(RUN / 'metrics.json'),
            'source_validation_sha256': checkpoint_digest(RUN / 'source_validation.json'),
            'score_manifest_sha256': checkpoint_digest(RUN / 'score_manifest.json'), 'target_access': True})
        write_json(LAUNCH / 'queue_status.json', {'status': 'complete', 'target_access': True})
        print(f'COMPLETE target EER={100*report["target"]["eer"]:.4f}% AUROC={report["target"]["auroc"]:.6f}', flush=True)


if __name__ == '__main__': main()
