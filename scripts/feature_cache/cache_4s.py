"""Extract one shared four-second source cache; no model fitting or targets."""
import argparse
import gc
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader

from audio_deepfake_detection.features.frozen_4s import (
    SCHEMA,COUNTS,CROP_SEED,WAVLM_REVISION,CLAP_REVISION,SingleCropDataset,
)
from audio_deepfake_detection.lacf.data import _initialize_worker
from audio_deepfake_detection.lacf.diagnostic_cache import FixedCropCollator
from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.pretrained import load_local_model
from audio_deepfake_detection.lacf.training import device_batch
from audio_deepfake_detection.protocol import checkpoint_digest
from scripts.experiments.lacf_diagnostics.cache_diagnostics import source_records,encode_chunk

ROOT=Path(__file__).resolve().parents[2]
FAMILY=ROOT/'outputs/frozen_features/4s'
LAUNCH=FAMILY/'launch'
DATA=Path('/mnt/drive/audio-deepfake-cache')
MEMORY_CAP=10*1024**3
MINIMUM_FREE=12*1024**3


def save_json(path,value,*,replace=False):
    path=Path(path)
    if not replace:
        with path.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')
    else:
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def guards(evidence):
    for name,digest in evidence['code_sha256'].items():
        if checkpoint_digest(ROOT/name)!=digest:
            raise ValueError('Guarded source changed: '+name)


def progress(**status):
    save_json(LAUNCH/'progress.json',{'at_unix':time.time(),**status,'target_access':False},replace=True)


def stream(records,ids,collator,*,training,workers):
    options={'prefetch_factor':2,'persistent_workers':True}if workers else{}
    return DataLoader(SingleCropDataset(records,ids,training=training),batch_size=None,
        collate_fn=FixedCropCollator(collator),num_workers=workers,pin_memory=True,
        worker_init_fn=_initialize_worker,generator=torch.Generator().manual_seed(CROP_SEED),**options)


def pinned_files():
    hub=Path('/home/ubuntu/.cache/huggingface/hub')
    result={}
    for folder,revision in [('models--microsoft--wavlm-base-plus',WAVLM_REVISION),
                            ('models--laion--clap-htsat-unfused',CLAP_REVISION)]:
        model=hub/folder
        if (model/'refs/main').read_text().strip()!=revision:
            raise ValueError('Pinned local encoder reference changed')
        snapshot=model/'snapshots'/revision
        result[folder]={'revision':revision,'files_sha256':{p.name:checkpoint_digest(p)for p in sorted(snapshot.iterdir())if p.is_file()}}
        if not any(n.endswith(('.safetensors','.bin'))for n in result[folder]['files_sha256']):
            raise FileNotFoundError('Pretrained local weights missing')
    return result


def checked_sources():
    sources=source_records(DATA)
    previous=json.loads((ROOT/'outputs/experiments/lacf_diagnostics/frozen_features/complete.json').read_text())
    for (domain,split),(records,ids)in sources.items():
        member=hashlib.sha256(json.dumps(list(zip(ids,[label for _,label in records]))).encode()).hexdigest()
        if member!=previous['splits'][f'{domain}/{split}']['canonical_ids_labels_sha256']:
            raise ValueError('Source membership differs from the verified ten-second cache')
    return sources


def source_probe(model,collator,sources,evidence):
    prepared=[]
    for (domain,split),(records,ids)in sources.items():
        indices=[0,len(records)-1]
        sample=[records[i]for i in indices]
        names=[f'{domain}/{split}/{ids[i]}'for i in indices]
        prepared.extend(list(stream(sample,names,collator,training=split=='train',workers=0)))
    progress(phase='source-only native feature parity',probe_recordings=len(prepared))
    native=encode_chunk(model,prepared,1,1)
    direct=[]
    for i,row in enumerate(prepared):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=model.encode_audio(device_batch(row['views'][0],'cuda'))
        w=torch.from_numpy(native['wavlm_layers'][i:i+1,-1]).cuda()
        c=torch.from_numpy(native['clap_projected'][i:i+1]).cuda()
        direct.append({'id':row['id'],'wavlm_last_state_bit_exact':torch.equal(w,result['z_w']),
                       'normalized_clap_bit_exact':torch.equal(unit_vectors(unit_vectors(c)),result['v_c'])})
    if not all(r['wavlm_last_state_bit_exact']and r['normalized_clap_bit_exact']for r in direct):
        raise ValueError('Four-second native cache/direct-audio parity failed')
    longest=[r for r in prepared if r['views'][0]['wavlm']['input_values'].shape[-1]==64000]
    if not longest:
        raise ValueError('Source timing set lacks full four-second inputs')
    benchmark=[longest[i%len(longest)]for i in range(32)]
    expected=encode_chunk(model,benchmark,1,1)
    measurements=[]
    for size in (1,4,8,16):
        guards(evidence);progress(phase='source-only extraction timing',encoder_batch=size)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.monotonic()
        try:
            arrays=encode_chunk(model,benchmark,size,size);torch.cuda.synchronize()
            elapsed=time.monotonic()-started
            if not all(np.isfinite(v).all()for v in arrays.values()):
                raise ValueError('Nonfinite encoder features')
            measurements.append({'batch_size':size,'seconds':elapsed,'recordings_per_second':32/elapsed,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'native_bit_exact':all(np.array_equal(arrays[k],expected[k])for k in arrays),
                'max_absolute_difference':{k:float(np.abs(arrays[k]-expected[k]).max())for k in arrays},
                'minimum_cosine':{k:float(torch.nn.functional.cosine_similarity(torch.from_numpy(arrays[k]).reshape(-1,arrays[k].shape[-1]),
                    torch.from_numpy(expected[k]).reshape(-1,expected[k].shape[-1]),dim=-1).min())for k in arrays}})
            del arrays
        except torch.cuda.OutOfMemoryError:
            measurements.append({'batch_size':size,'failure':'OOM within this process allocator cap; no precision/crop change'})
            gc.collect();torch.cuda.empty_cache()
        logging.info('Source extraction batch%d: %s',size,measurements[-1])
    selected=min((r for r in measurements if 'failure'not in r),key=lambda r:r['seconds'])['batch_size']
    a=encode_chunk(model,prepared,selected,selected)
    b=encode_chunk(model,prepared,selected,selected)
    if not all(np.array_equal(a[k],b[k])for k in a):
        raise ValueError('Repeated selected-batch extraction is not bit-exact')
    # A concrete disk serialize/reload check, independent of later trainable heads.
    probe_dir=LAUNCH/'source_probe';probe_dir.mkdir()
    for name,array in a.items():
        path=probe_dir/f'{name}.npy'
        with path.open('xb')as handle:np.save(handle,array,allow_pickle=False)
        if not np.array_equal(np.load(path,allow_pickle=False),array):
            raise ValueError('Serialized frozen features changed')
    timing_records=[];timing_ids=[]
    for (domain,split),(records,ids)in sources.items():
        if split=='train':
            timing_records.extend(records[:32]);timing_ids.extend(f'{domain}/{split}/{v}'for v in ids[:32])
    loading=[]
    for workers in (8,16):
        progress(phase='source-only preprocessing timing',workers=workers)
        started=time.monotonic();loader=stream(timing_records,timing_ids,collator,training=True,workers=workers)
        seen=[(r['id'],r['intervals'])for r in loader]
        loading.append({'workers':workers,'seconds':time.monotonic()-started,'recordings':len(seen)})
        if workers==8:first=seen
        elif first!=seen:raise ValueError('Worker count changed single-crop identity/order')
        del loader;gc.collect()
    workers=min(loading,key=lambda r:r['seconds'])['workers']
    result={'passed':True,'source_only':True,'target_access':False,'direct_parity':direct,
            'batch_measurements':measurements,'worker_measurements':loading,'selected_batch_size':selected,
            'selected_workers':workers,'repeated_selected_batch_bit_exact':True,'serialized_features_bit_exact':True,
            'scope':'16 first/tail source train/dev recordings and32 repeated4s views;128 source train CPU timing inputs',
            'numerical_disclosure':'Selected batched BF16 kernels may differ from native batch1; changes above measured only on bounded source probe. All future heads share this identical cache.',
            'code_sha256':evidence['code_sha256']}
    save_json(LAUNCH/'source_validation.json',result)
    return result


def write_chunk(directory,index,prepared,arrays,evidence):
    if any(len(r['views'])!=1 or len(r['view_rows'])!=1 for r in prepared):
        raise ValueError('Exactly one physical and logical crop per recording is required')
    destination=directory/f'chunk_{index:05d}'
    temporary=directory/f'chunk_{index:05d}.partial'
    temporary.mkdir(exist_ok=False)
    metadata=[]
    for local,row in enumerate(prepared):
        metadata.append({**{k:v for k,v in row.items()if k!='views'},'view_rows':[local]})
    for name,shape in [('wavlm_layers',(len(prepared),13,768)),('clap_projected',(len(prepared),512))]:
        a=arrays[name]
        if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():
            raise ValueError('Invalid single-view feature payload')
        with (temporary/f'{name}.npy').open('xb')as handle:np.save(handle,a,allow_pickle=False)
    save_json(temporary/'records.json',metadata)
    hashes={name:checkpoint_digest(temporary/name)for name in ('wavlm_layers.npy','clap_projected.npy','records.json')}
    save_json(temporary/'complete.json',{'records':len(prepared),'unique_rows':len(prepared),
              'files_sha256':hashes,'source_validation_sha256':checkpoint_digest(LAUNCH/'source_validation.json')})
    if destination.exists():raise FileExistsError('Preserve existing shard')
    temporary.rename(destination)
    return {'path':destination.name,'records':len(prepared),'unique_rows':len(prepared),
            'completion_sha256':checkpoint_digest(destination/'complete.json')}


def build(model,collator,sources,evidence,proof,weights):
    batch=proof['selected_batch_size'];workers=proof['selected_workers']
    prototypes=model.prototypes.detach().cpu().numpy()
    with (FAMILY/'prototypes.npy').open('xb')as handle:np.save(handle,prototypes,allow_pickle=False)
    config={'schema':SCHEMA,'segment_seconds':4,'source_splits_only':True,'views_per_record':1,
            'views':{'train':['fixed_random'],'dev':['center']},'crop_seed':CROP_SEED,
            'crop_rng':'sha256(seed,NUL,canonical domain/split/id,NUL,single_random_4s); torch.Generator uniform start',
            'precision':'BF16 encoder forward; FP32 normalization/masked temporal pooling and feature storage',
            'wavlm_shape':[13,768],'clap_shape':[512],'pretrained_revisions':{'wavlm':WAVLM_REVISION,'clap':CLAP_REVISION},
            'pretrained_local_files':weights,'prototypes_sha256':checkpoint_digest(FAMILY/'prototypes.npy'),
            'processors':collator.configuration(),'encoder_batch_size':batch,'workers':workers,'prefetch':2,
            'chunk_records':512,'batching':'equal native WavLM lengths, no padding; CLAP fixed mel shape batched separately',
            'source_validation_sha256':checkpoint_digest(LAUNCH/'source_validation.json'),
            'torch':str(torch.__version__),'transformers':transformers.__version__,'cuda':torch.version.cuda,
            'code_sha256':evidence['code_sha256'],'logical_views_per_record':1,
            'scope':'One shared source train/dev bank across folds; no target data or trainable checkpoint used'}
    save_json(FAMILY/'config.json',config)
    totals={};overall=time.monotonic();overall_count=0
    for (domain,split),(records,ids)in sources.items():
        directory=FAMILY/domain/split;directory.mkdir(parents=True,exist_ok=False)
        names=[f'{domain}/{split}/{v}'for v in ids]
        loader=stream(records,names,collator,training=split=='train',workers=workers)
        iterator=iter(loader);done=0;chunks=[];started=time.monotonic()
        while prepared:=list(itertools.islice(iterator,512)):
            guards(evidence)
            if [r['index']for r in prepared]!=list(range(done,done+len(prepared))):
                raise ValueError('Source cache order changed')
            arrays=encode_chunk(model,prepared,batch,batch)
            chunks.append(write_chunk(directory,len(chunks),prepared,arrays,evidence))
            done+=len(prepared);overall_count+=len(prepared)
            elapsed=time.monotonic()-overall
            progress(phase='extracting four-second source features',partition=f'{domain}/{split}',
                     partition_done=done,partition_total=len(records),overall_done=overall_count,overall_total=1243402,
                     overall_recordings_per_second=overall_count/max(elapsed,.001),
                     estimated_remaining_seconds=(1243402-overall_count)*elapsed/overall_count)
            logging.info('%s/%s | %d/%d | total %d/1243402 | %.1f recordings/s | remaining %.2fh',
                         domain,split,done,len(records),overall_count,overall_count/elapsed,
                         (1243402-overall_count)*elapsed/overall_count/3600)
            del arrays,prepared
        if done!=len(records):raise ValueError('Incomplete source partition')
        totals[f'{domain}/{split}']={'records':done,'unique_rows':done,'logical_views':done,
            'canonical_ids_labels_sha256':hashlib.sha256(json.dumps(list(zip(ids,[label for _,label in records]))).encode()).hexdigest(),
            'seconds':time.monotonic()-started,'chunks':chunks}
        save_json(directory/'complete.json',totals[f'{domain}/{split}'])
        del loader,iterator;gc.collect()
    guards(evidence)
    if sum(p['records']for p in totals.values())!=1243402:raise ValueError('Missing source records')
    save_json(FAMILY/'complete.json',{'schema':SCHEMA,'config_sha256':checkpoint_digest(FAMILY/'config.json'),
              'splits':totals,'target_access':False,'source_only':True,'records':overall_count})
    progress(phase='all source features complete; models ready for separately requested training',overall_done=overall_count)


def main():
    if (LAUNCH/'started.json').exists() or (FAMILY/'config.json').exists() or (FAMILY/'complete.json').exists():
        raise FileExistsError('Preserve existing extraction; no automatic retry/resume/overwrite')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    evidence=json.loads((LAUNCH/'provenance.json').read_text());guards(evidence)
    # Wait without initializing CUDA or loading models; never stop another job here.
    while True:
        guards(evidence)
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
        if free>=MINIMUM_FREE:break
        progress(phase='waiting for extraction GPU headroom',free_gpu_bytes=free,required_bytes=MINIMUM_FREE)
        logging.info('WAIT GPU %.2fGiB; need12GiB; current jobs preserved',free/1024**3);time.sleep(30)
    if shutil.disk_usage(FAMILY).free<65*10**9:raise RuntimeError('Need at least65GB free disk')
    save_json(LAUNCH/'started.json',{'pid':os.getpid(),'target_access':False,'source_records':1243402})
    torch.set_num_threads(2);torch.manual_seed(CROP_SEED);np.random.seed(CROP_SEED)
    torch.backends.nnpack.set_flags(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available()or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(MEMORY_CAP/torch.cuda.get_device_properties(0).total_memory)
    progress(phase='checking pinned local weights and canonical source membership')
    weights=pinned_files();sources=checked_sources()
    progress(phase='loading frozen local Base+ and CLAP encoders')
    model,collator,revisions=load_local_model(segment_seconds=4)
    if revisions['wavlm_revision']!=WAVLM_REVISION or revisions['clap_revision']!=CLAP_REVISION:
        raise ValueError('Loaded encoder revision differs')
    model=model.cuda().eval()
    if any(p.requires_grad for encoder in (model.wavlm,model.clap)for p in encoder.parameters()):
        raise ValueError('Frozen encoders required')
    proof=source_probe(model,collator,sources,evidence)
    build(model,collator,sources,evidence,proof,weights)


if __name__=='__main__':
    try:main()
    except Exception:
        if LAUNCH.is_dir():progress(phase='FAILED; extraction stopped without retry')
        logging.exception('Four-second source extraction failed; preserve all evidence')
        raise
