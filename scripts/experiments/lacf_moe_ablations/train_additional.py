"""Complete the predefined B1–B6 controls with the unchanged cached fitting loop."""

from contextlib import contextmanager
import logging
from pathlib import Path
import time

import torch

from audio_deepfake_detection.experiments.lacf_moe_ablations import additional_model as models
from audio_deepfake_detection.lacf.probes import load_bank
from scripts.experiments.lacf_moe_ablations import train as core

FAMILY=core.FAMILY/'additional'
LAUNCH=FAMILY/'launch'


@contextmanager
def additional_controls():
    """Bind only this process's fixed model registry; restore it even on failure.

    Reuse the tested fitter/preflight/export code rather than copy its training
    loop. No core source bytes, running processes or A-series registry change.
    These bindings and their implementation are pinned in every run provenance.
    """
    replacements={'CORE':('A0',*models.VARIANTS),'NAMES':models.NAMES,'PARAMETERS':models.PARAMETERS,
        'FEATURE_INPUTS':models.FEATURE_INPUTS,'build_model':models.build_model,'FAMILY':FAMILY}
    before={name:getattr(core,name) for name in replacements}
    try:
        for name,value in replacements.items():setattr(core,name,value)
        yield core
    finally:
        for name,value in before.items():setattr(core,name,value)


def main():
    if (LAUNCH/'started.json').exists() or any((FAMILY/f'{v}_{models.NAMES[v]}').exists() for v in models.VARIANTS):
        raise FileExistsError('Prior additional launch/runs preserved; no retry/resume/overwrite')
    provenance=core.read(LAUNCH/'provenance.json');core.guards(provenance)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native BF16 CUDA required; no fallback')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.nnpack.set_flags(False);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');logger=logging.getLogger('moe_additional')
    core.save_json(LAUNCH/'started.json',{'phase':'source cache/resource checks','target_access':False})
    while True:
        core.guards(provenance)
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        free,_=torch.cuda.mem_get_info()
        if available>=75*1024**3 and free>=2*1024**3:break
        core.progress(LAUNCH,phase='waiting for resources',available_ram_bytes=available,free_gpu_bytes=free)
        logger.info('WAIT: need75GiB RAM/2GiB GPU; other jobs preserved');time.sleep(10)
    core.progress(LAUNCH,phase='verifying/loading frozen source train/dev cache',completed_fits=0)
    bank=load_bank(core.CACHE,core.DATA,logger=logger)
    for ds in bank.values():
        if not torch.isfinite(ds.wavlm).all() or not torch.isfinite(ds.clap).all():raise ValueError('Nonfinite source cache')
    core.save_json(LAUNCH/'cache_validation.json',{'all_canonical_memberships_and_payload_sha256_verified':True,
        'all_features_finite':True,'target_access':False,'splits':{f'{d}/{s}':len(v.labels) for (d,s),v in bank.items()}})
    prototypes,evidence=core.fixed_prototypes();provenance['prototypes']=evidence
    core.save_json(LAUNCH/'queue_config.json',{'variants':models.NAMES,'parameters':models.PARAMETERS,
        'folds':list(core.FOLDS),'seed_plan':[1234],'expected_fits':24,'target_access':False,
        'recipe':'unchanged core cached fitting loop; original prototypes; detection BCE only',
        'process_local_registry':str(Path(__file__).resolve()),'provenance':provenance})
    core.save_json(LAUNCH/'queue_status.json',{'completed_fits':0,'runs':[]})
    with additional_controls() as runner:
        completed=runner.run_queue(bank,prototypes,provenance,'cuda',LAUNCH,logger)
    core.save_json(LAUNCH/'complete.json',{'study':'lacf_moe_ablations','scope':'additional B1–B6',
        'completed_fits':len(completed),'runs':completed,'target_access':False})
    core.progress(LAUNCH,phase='all additional source training complete',completed_fits=24)


if __name__=='__main__':main()
