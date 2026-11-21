"""Eight sequential full-fold FP32 fits; cached encoders, original MoE reference."""
from contextlib import contextmanager
import gc
import json
import logging
import os
from pathlib import Path
import time

import torch

from audio_deepfake_detection.experiments.lacf_moe_embedding_paths import model as models
from audio_deepfake_detection.lacf.probes import load_split
from scripts.experiments.lacf_moe_ablations import train_fp32 as core

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / 'outputs/experiments/lacf_moe_embedding_paths/fp32'
LAUNCH = FAMILY / 'launch'
EXPECTED_FITS = 8


@contextmanager
def bindings():
    """Reuse tested FP32 fitting/gates in this process; no source files are changed."""
    replacements = {'CORE':('reference', *models.VARIANTS), 'NAMES':models.NAMES,
        'PARAMETERS':models.PARAMETERS, 'FEATURE_INPUTS':models.FEATURE_INPUTS,
        'OBJECTIVES':{v:models.OBJECTIVE for v in models.VARIANTS},
        'build_model':models.build_model, 'STUDY':'lacf_moe_embedding_paths', 'FAMILY':FAMILY}
    previous = {key:getattr(core,key) for key in replacements}
    try:
        for key,value in replacements.items():setattr(core,key,value)
        yield core
    finally:
        for key,value in previous.items():setattr(core,key,value)


def available_ram():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def wait_resources(provenance, logger, *, ram_bytes, phase):
    while True:
        core.guards(provenance)
        ram = available_ram()
        free,_ = torch.cuda.mem_get_info()
        if ram >= ram_bytes and free >= 2 * 1024**3:
            return
        core.progress(LAUNCH, phase=phase, available_ram_bytes=ram, free_gpu_bytes=free)
        logger.info('WAIT | RAM %.1fGiB (need%.1f), GPU %.1fGiB (need2); existing jobs preserved',
                    ram/1024**3, ram_bytes/1024**3, free/1024**3)
        time.sleep(30)


def load_fold_bank(bank, fold, provenance, logger):
    """Reuse shared source partitions, release excluded ones before allocating more.

    Original load_split checks canonical membership, crop aliases, manifest and
    every payload hash. No audio is decoded and no features are re-extracted.
    """
    core.guards(provenance)
    manifest = core.read(core.CACHE / 'complete.json')
    expected = {f'{d}/{s}' for d in ('asv2019','asv5','cfad','speechfake') for s in ('train','dev')}
    if manifest.get('step') != 2 or set(manifest['splits']) != expected:
        raise ValueError('Frozen source cache completion identity differs')
    needed = {(d,s) for d in core.FOLDS[fold]['sources'] for s in ('train','dev')}
    for key in list(bank):
        if key not in needed:del bank[key]
    gc.collect()
    for domain in core.FOLDS[fold]['sources']:
        for split in ('train','dev'):
            if (domain,split) in bank:continue
            descriptor = manifest['splits'][f'{domain}/{split}']
            # Contiguous FP32 layer/audio arrays plus Python metadata and reserve.
            required = descriptor['unique_rows'] * (13*768+512) * 4 + 10 * 1024**3
            wait_resources(provenance, logger, ram_bytes=required, phase=f'waiting to load {domain}/{split}')
            core.progress(LAUNCH, phase='verifying/loading cached source features', fold=fold,
                          partition=f'{domain}/{split}')
            ds = load_split(core.CACHE, domain, split, descriptor, core.DATA, logger=logger)
            for offset in range(0,len(ds.wavlm),8192):
                if not torch.isfinite(ds.wavlm[offset:offset+8192]).all() or not torch.isfinite(ds.clap[offset:offset+8192]).all():
                    raise ValueError('Nonfinite source feature cache')
            bank[domain,split] = ds
    core.assert_bank(bank,fold)
    return bank


def verify_reference(bank, fold, provenance):
    run = core.REFERENCE / fold / '1234'
    for name,digest in provenance['reference_sha256'][fold].items():
        if core.checkpoint_digest(run/name) != digest:raise ValueError('FP32 baseline artifact changed')
    core.verify_reference(run,fold,expected_labels=core.labels_for(bank,fold))
    link = FAMILY / 'reference' / fold / '1234'
    link.parent.mkdir(parents=True,exist_ok=False)
    link.symlink_to(os.path.relpath(run,link.parent),target_is_directory=True)
    core.save_json(LAUNCH/f'reference_{fold}.json',{'fold':fold,'reference_path':str(run),
        'training_rerun':False,'reference_sha256':provenance['reference_sha256'][fold],'target_access':False})


def main():
    if (LAUNCH/'started.json').exists() or any((FAMILY/v).exists() for v in (*models.VARIANTS,'reference')):
        raise FileExistsError('Preserve existing launch/runs; no automatic retry/resume/overwrite')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logger = logging.getLogger('lacf_moe_embedding_paths')
    provenance = core.read(LAUNCH/'provenance.json');core.guards(provenance)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no fallback')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False;torch.backends.nnpack.set_flags(False)
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    core.save_json(LAUNCH/'started.json',{'pid':os.getpid(),'target_access':False,'precision':core.PRECISION})
    core.save_json(LAUNCH/'queue_config.json',{'variants':models.NAMES,'folds':list(core.FOLDS),'seed_plan':[1234],
        'expected_fits':8,'precision':core.PRECISION,'recipe':'same cached full-fold MoE/BCE settings',
        'order':'fold F1->F4; CLAP path then matched WavLM path within each fold',
        'no_inner_rotations':True,'target_evaluation_queued':False,'provenance':provenance})
    core.save_json(LAUNCH/'queue_status.json',{'completed_fits':0,'runs':[]})
    prototypes,evidence = core.fixed_prototypes()
    if evidence != provenance['prototypes']:raise ValueError('Original prototype bank changed')
    bank,completed = {},[]
    for fold in core.FOLDS:
        load_fold_bank(bank,fold,provenance,logger)
        verify_reference(bank,fold,provenance)
        with bindings():
            for variant in models.VARIANTS:
                wait_resources(provenance,logger,ram_bytes=2*1024**3,phase='waiting for fit resources')
                core.progress(LAUNCH,phase='source preflight',variant=variant,fold=fold,completed_fits=len(completed))
                proof = LAUNCH/f'source_validation_{variant}_{fold}.json'
                core.save_json(proof,core.preflight(bank,prototypes,fold,variant,provenance,'cuda',logger))
                current = {**provenance,'source_validation_record':str(proof),'source_validation_sha256':core.checkpoint_digest(proof)}
                run = FAMILY/variant/fold/'1234'
                core.fit(bank,prototypes,run,fold,variant,current,'cuda',logger,launch=LAUNCH)
                core.verify_completion(run,fold,variant,core.labels_for(bank,fold))
                completed.append({'variant':variant,'fold':fold,'completion_sha256':core.checkpoint_digest(run/'training_complete.json')})
                status = LAUNCH/'queue_status.json';tmp = status.with_suffix('.json.tmp')
                tmp.write_text(json.dumps({'completed_fits':len(completed),'runs':completed},indent=2)+'\n');tmp.replace(status)
                core.progress(LAUNCH,phase='advancing after verified completion',variant=variant,fold=fold,completed_fits=len(completed))
                gc.collect();torch.cuda.empty_cache()
    if len(completed)!=EXPECTED_FITS:raise ValueError('Incomplete embedding-path queue')
    core.guards(provenance)
    core.save_json(LAUNCH/'complete.json',{'completed_fits':8,'runs':completed,'reference_fits_reused':4,
        'target_access':False,'precision':core.PRECISION})
    core.progress(LAUNCH,phase='all eight source fits verified complete',completed_fits=8)


if __name__=='__main__':
    try:main()
    except Exception:
        if LAUNCH.is_dir():core.progress(LAUNCH,phase='FAILED; stopped without retry')
        logging.exception('Embedding-path queue stopped; preserve all evidence')
        raise
