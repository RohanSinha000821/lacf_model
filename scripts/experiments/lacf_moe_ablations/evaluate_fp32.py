"""One bounded, fail-closed native evaluation queue for 60 controls and eight paths.

Only completed producer entries advance. Historical cached encoders are reused;
actual head computation, source selection and calibration remain FP32.
"""
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import logging
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from audio_deepfake_detection.metrics import read_score_csv
from audio_deepfake_detection.protocol import FOLDS, canonical_score_labels, checkpoint_digest
from scripts.experiments.lacf_moe_ablations import train_fp32 as core
from scripts.experiments.lacf_moe_embedding_paths import train as paths
from scripts.lacf_moe import evaluate_fp32 as shared

ROOT, CACHE, DATA = core.ROOT, core.CACHE, core.DATA
OUTPUT = ROOT / 'outputs/experiments/lacf_moe_fp32_evaluations'
LAUNCH = OUTPUT / 'launch'
FAMILIES = {'ablations': core.FAMILY, 'embedding_paths': paths.FAMILY}
VARIANTS = {'ablations': tuple(core.CORE[1:]), 'embedding_paths': paths.models.VARIANTS}
read, save_json, replace_json = core.read, shared.save_json, shared.replace_json
GIB = 1024**3


def case_paths(family, variant, fold):
    if family not in FAMILIES or variant not in VARIANTS[family] or fold not in FOLDS:
        raise ValueError('Only the authorized 60 ablations and eight full-fold paths')
    folder = f'{variant}_{core.NAMES[variant]}' if family == 'ablations' else variant
    root = FAMILIES[family]
    return root / folder / fold / '1234', root / 'evaluations' / folder / fold / '1234'


def registry(family):
    return paths.bindings() if family == 'embedding_paths' else nullcontext(core)


def ready_cases(family, provenance, observed):
    """Bind live producers and immutable completed entries; never infer from best.pt."""
    launch = FAMILIES[family] / 'launch'
    identity = provenance['training_producers'][family]
    for name, digest in identity['launch_sha256'].items():
        if checkpoint_digest(launch / name) != digest:
            raise ValueError('Training producer launch changed')
    core.guards(read(launch / 'provenance.json'))
    status = read(launch / 'queue_status.json')
    entries = status['runs']
    keys = [(r['variant'], r['fold']) for r in entries]
    if status['completed_fits'] != len(entries) or len(set(keys)) != len(keys):
        raise ValueError('Invalid producer completion count/duplicates')
    maximum = len(VARIANTS[family]) * len(FOLDS)
    if len(entries) > maximum:
        raise ValueError('Producer exceeds authorized fit scope')
    current = {}
    for entry in entries:
        variant, fold = entry['variant'], entry['fold']
        run, _ = case_paths(family, variant, fold)
        digest = entry['completion_sha256']
        if checkpoint_digest(run / 'training_complete.json') != digest:
            raise ValueError('Producer-verified completion changed')
        current[(variant, fold)] = digest
    previous = observed.setdefault(family, {})
    if any(current.get(key) != digest for key, digest in previous.items()):
        raise ValueError('Producer removed or changed a completed entry')
    exit_path = launch / 'exit.json'
    if exit_path.exists():
        exit_record = read(exit_path)
        if (exit_record.get('trainer_exit') != 0 or exit_record.get('logger_exit') != 0
                or exit_record.get('verified_complete') is not True):
            raise ValueError('Training producer failed/stopped; no evaluation retry')
        completion = read(launch / 'complete.json')
        if (len(entries) != maximum or completion['completed_fits'] != maximum
                or completion['runs'] != entries or completion['target_access'] is not False
                or completion['precision'] != core.PRECISION):
            raise ValueError('Successful producer completion differs')
    else:
        command = Path(f"/proc/{identity['pid']}/cmdline")
        if not command.exists() or identity['module'].encode() not in command.read_bytes():
            raise RuntimeError('Training producer disappeared without verified exit')
    observed[family] = current
    return [(family, variant, fold, digest) for (variant, fold), digest in current.items()]


def verify_training(family, variant, fold, digest):
    run, output = case_paths(family, variant, fold)
    if output.exists():
        raise FileExistsError('Existing evaluation preserved; no retry/resume/overwrite')
    if checkpoint_digest(run / 'training_complete.json') != digest:
        raise ValueError('Queue-bound completion changed')
    labels = {d: canonical_score_labels(d, 'dev', DATA) for d in FOLDS[fold]['sources']}
    with registry(family):
        done = core.verify_completion(run, fold, variant, labels)
        config = read(run / 'config.json')
        core.guards(config['provenance'])
        if (config['precision'] != core.PRECISION
                or Path(config['provenance']['cache']).resolve() != CACHE.resolve()
                or config['provenance']['cache_completion_sha256'] != checkpoint_digest(CACHE / 'complete.json')):
            raise ValueError('Selected precision/source cache differs')
        state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        prototypes = next((v for k,v in state['model_state_dict'].items() if k.endswith('prototypes')), None)
        if prototypes is None:
            reference = torch.load(core.REFERENCE / 'f1/1234/best.pt', map_location='cpu', weights_only=True)
            prototypes = reference['model_state_dict']['backend.prototypes']
        if hashlib.sha256(prototypes.numpy().tobytes()).hexdigest() != config['provenance']['prototypes']['prototype_bytes_sha256']:
            raise ValueError('Original prototype identity changed')
        with torch.random.fork_rng(devices=[]):
            model = core.build_model(variant, prototypes)
        model.load_state_dict(state['model_state_dict'], strict=True)
        inputs = core.FEATURE_INPUTS[variant]
    return run, output, config, done, model.requires_grad_(False).eval(), inputs


@torch.inference_mode()
def score_arrays(model, arrays, device, inputs='concat', *, batch_size=132, last_layer_only=False):
    if inputs not in ('clap', 'weighted', 'concat') or batch_size != 132 or last_layer_only:
        raise ValueError('Preserve authorized feature inputs and head batch132')
    core.assert_fp32(model)
    count = len(arrays['wavlm_layers'])
    if (count < 1 or arrays['wavlm_layers'].shape != (count,13,768)
            or arrays['clap_projected'].shape != (count,512)
            or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays.values())):
        raise ValueError('Invalid finite FP32 frozen feature arrays')
    values = []
    for start in range(0, count, batch_size):
        batch = {}
        if inputs != 'clap':
            batch['wavlm'] = torch.from_numpy(arrays['wavlm_layers'][start:start+batch_size]).to(device)
        if inputs in ('clap', 'concat'):
            batch['clap'] = torch.from_numpy(arrays['clap_projected'][start:start+batch_size]).to(device)
        with torch.autocast(torch.device(device).type, enabled=False):
            scores = model(batch)['logit']
        size = len(next(iter(batch.values())))
        if scores.dtype != torch.float32 or scores.shape != (size,) or not torch.isfinite(scores).all():
            raise ValueError('Final scoring layer must compute finite FP32 logits')
        values.append(scores.cpu())
    return torch.cat(values).numpy()


@contextmanager
def fp32_exporter():
    """Reuse fixed export/metrics gates; replace scorer only in this process."""
    previous = shared.score_arrays
    shared.score_arrays = score_arrays
    try:
        yield
    finally:
        shared.score_arrays = previous


def source_reload(model, run, bank, fold, variant, device, logger):
    scores, metrics, _ = core.score_sources(model, bank, fold, variant, device, logger)
    if metrics != read(run / 'metrics.json')['source_dev']:
        raise ValueError('Fresh full FP32 source metrics differ; target held')
    checks = []
    for domain in FOLDS[fold]['sources']:
        data = bank[domain, 'dev']
        _, saved = read_score_csv(run / f'source_dev_{domain}.csv', domain,
            expected_labels=dict(zip(data.ids, data.labels.tolist())))
        order = sorted(range(len(data.ids)), key=data.ids.__getitem__)
        if not np.array_equal(scores[domain].numpy()[order], saved):
            raise ValueError('Fresh full FP32 source scores differ; target held')
        checks.append({'domain':domain, 'records':len(saved), 'score_bit_exact':True})
    return {'passed':True, 'full_source_score_bit_exact':True, 'checks':checks,
        'precision':core.PRECISION, 'encoders_loaded_or_executed':False, 'target_access':False,
        'queue_script_sha256':checkpoint_digest(Path(__file__)),
        'calibration':'exact frozen FP32 source calibration; no target recalibration'}


def verify_export(output, run, fold, done):
    completion = read(output / 'evaluation_complete.json')
    manifest = read(output / 'score_manifest.json')
    config = read(output / 'evaluation_config.json')
    validation = read(output / 'source_validation.json')
    if (completion['fold'] != fold or completion['seed'] != 1234
            or completion['checkpoint_sha256'] != done['best_checkpoint_sha256']
            or not completion['no_target_tuning'] or not completion['source_selected_checkpoint_only']
            or completion['metrics_sha256'] != checkpoint_digest(output / 'metrics.json')
            or completion['source_validation_sha256'] != checkpoint_digest(output / 'source_validation.json')
            or not validation['passed'] or not validation['full_source_score_bit_exact']
            or {c['domain'] for c in validation['checks']} != set(FOLDS[fold]['sources'])
            or not all(c['score_bit_exact'] for c in validation['checks'])
            or completion['score_sha256'] != manifest['score_sha256']
            or config['inference']['precision'] != core.PRECISION
            or config['inference']['encoder_execution'] is not False
            or config['training_completion_sha256'] != checkpoint_digest(run / 'training_complete.json')
            or manifest['training_completion_sha256'] != config['training_completion_sha256']):
        raise ValueError('Completed evaluation bindings differ')
    for name in ('best.pt','config.json','training_complete.json','source_calibration.json'):
        if checkpoint_digest(output / name) != checkpoint_digest(run / name):
            raise ValueError('Copied training artifacts changed')
    expected = {f'source_dev_{d}.csv' for d in FOLDS[fold]['sources']} | {'target_scores.csv'}
    if set(completion['score_sha256']) != expected:
        raise ValueError('Source/target export set differs')
    for name,digest in completion['score_sha256'].items():
        if checkpoint_digest(output / name) != digest:
            raise ValueError('Exported scores changed')
        if name != 'target_scores.csv' and done['source_scores_sha256'][name] != digest:
            raise ValueError('Source exports changed')
    spec = FOLDS[fold]
    read_score_csv(output / 'target_scores.csv', spec['target'],
        expected_labels=canonical_score_labels(spec['target'],spec['target_split'],DATA))


def evaluate_case(case, bank, provenance, logger, *, device='cuda', bootstrap_resamples=1000):
    family, variant, fold, digest = case
    run, output, config, done, model, inputs = verify_training(family,variant,fold,digest)
    model = model.to(device)
    with registry(family):
        validation = source_reload(model,run,bank,fold,variant,device,logger)
    core.guards(config['provenance'])
    current = {**provenance, 'selected_training_completion_sha256':digest,
        'selected_checkpoint_sha256':done['best_checkpoint_sha256'], 'study':family,'variant':variant}
    with fp32_exporter():
        report = shared.finish_evaluation(model,run,output,fold,config['model_name'],
            f'lacf_moe_{family}_fp32',config,done,validation,current,device,logger,
            inputs=inputs,bootstrap_resamples=bootstrap_resamples)
    verify_export(output,run,fold,done)
    del model
    gc.collect()
    if torch.device(device).type == 'cuda':torch.cuda.empty_cache()
    return {'family':family,'variant':variant,'fold':fold,'training_completion_sha256':digest,
        'evaluation':str(output),'evaluation_completion_sha256':checkpoint_digest(output/'evaluation_complete.json'),
        'target_eer':report['target']['eer'],'target_auroc':report['target']['auroc']}


def gpu_free():
    return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
        '--format=csv,noheader,nounits','-i','0'],text=True).strip()) * 1024**2


def load_dev_bank(retained, fold, provenance, logger):
    needed = {(d,'dev') for d in FOLDS[fold]['sources']}
    for key in list(retained):
        if key not in needed:del retained[key]
    gc.collect()
    marker = read(CACHE/'complete.json')
    required = sum(marker['splits'][f'{d}/dev']['unique_rows']*(13*768+512)*4
                   for d,_ in needed if (d,'dev') not in retained) + 8*GIB
    if paths.available_ram() < required or gpu_free() < 2*GIB:
        # Avoid retaining a large bank while another producer is changing folds.
        retained.clear();gc.collect()
        return None
    return shared.source_bank(CACHE,fold,DATA,logger,retained)


def main():
    if (LAUNCH/'started.json').exists():
        raise FileExistsError('Existing queue preserved; no automatic retry/resume')
    provenance = read(LAUNCH/'provenance.json');core.guards(provenance)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logger = logging.getLogger('lacf_moe_fp32_evaluations')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.nnpack.set_flags(False)
    save_json(LAUNCH/'started.json',{'pid':os.getpid(),'encoder_execution':False,'expected_evaluations':68})
    retained, observed, completed = {},{},[]
    last_family, cuda_ready, last_fold = None,False,None
    while len(completed)<68:
        core.guards(provenance)
        ready = [case for family in FAMILIES for case in ready_cases(family,provenance,observed)]
        finished = {(r['family'],r['variant'],r['fold']) for r in completed}
        pending = [c for c in ready if c[:3] not in finished]
        pending.sort(key=lambda c:(c[0]==last_family,c[2]!=last_fold,c[2],c[1]))
        if not pending:
            retained.clear();gc.collect()
            replace_json(LAUNCH/'progress.json',{'phase':'waiting for verified training completions',
                'completed_evaluations':len(completed),'expected_evaluations':68})
            logger.info('WAIT | verified completions; %d/68 evaluated; encoders never loaded',len(completed))
            time.sleep(30);continue
        case = pending[0]; family,variant,fold,digest = case
        replace_json(LAUNCH/'progress.json',{'phase':'waiting for evaluation resources','family':family,
            'variant':variant,'fold':fold,'completed_evaluations':len(completed),'expected_evaluations':68})
        if gpu_free()<2*GIB or paths.available_ram()<8*GIB:
            retained.clear();gc.collect();time.sleep(30);continue
        # CPU completion/config gates precede source cache allocation and any target access.
        checked = verify_training(family,variant,fold,digest);del checked
        bank = load_dev_bank(retained,fold,provenance,logger)
        if bank is None:
            logger.info('WAIT | >=2GiB GPU and source-dev allocation plus8GiB RAM reserve')
            time.sleep(30);continue
        if not cuda_ready:
            if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<2*GIB:
                raise RuntimeError('CUDA headroom changed; preserve all active jobs')
            torch.cuda.set_per_process_memory_fraction(GIB/torch.cuda.get_device_properties(0).total_memory)
            cuda_ready=True
        replace_json(LAUNCH/'progress.json',{'phase':'source verification then cached target scoring',
            'family':family,'variant':variant,'fold':fold,'completed_evaluations':len(completed),'expected_evaluations':68})
        completed.append(evaluate_case(case,bank,provenance,logger))
        replace_json(LAUNCH/'queue_status.json',{'completed_evaluations':len(completed),'runs':completed})
        last_family,last_fold=family,fold
        del bank
        if paths.available_ram()<12*GIB:retained.clear();gc.collect()
    core.guards(provenance)
    save_json(LAUNCH/'complete.json',{'completed_evaluations':68,'runs':completed,
        'precision':core.PRECISION,'encoder_execution':False,'no_target_tuning':True})
    replace_json(LAUNCH/'progress.json',{'phase':'all 68 evaluations verified complete','completed_evaluations':68})


if __name__=='__main__':
    try:main()
    except Exception:
        if LAUNCH.is_dir():replace_json(LAUNCH/'progress.json',{'phase':'FAILED; stopped without retry'})
        logging.exception('FP32 evaluation queue stopped; preserve all evidence')
        raise
