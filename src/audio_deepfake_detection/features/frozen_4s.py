"""One physical four-second view per recording; source-only reusable features."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from audio_deepfake_detection.data import resample_audio
from audio_deepfake_detection.lacf.probes import FeatureBatches, FeatureSplit
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, read_dataset, utterance_id

SCHEMA = 'frozen_audio_features_4s_single_view_v1'
SECONDS = 4
CROP_SEED = 1234
WAVLM_REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
CLAP_REVISION = '8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a'
COUNTS = {'asv2019': (25380,24844), 'asv5': (182357,140950),
          'cfad': (38400,14400), 'speechfake': (699608,117463)}


def crop_interval(frames, rate, identifier, *, training, seed=CROP_SEED):
    if frames <= 0 or rate <= 0:
        raise ValueError('Empty waveform or invalid rate')
    length = min(frames, round(SECONDS*rate))
    maximum = frames-length
    if not training:
        return maximum//2, length
    key = '\0'.join((str(seed),identifier,'single_random_4s'))
    digest = hashlib.sha256(key.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8],'big'))
    return int(torch.randint(maximum+1,(1,),generator=generator)), length


class SingleCropDataset(Dataset):
    def __init__(self, records, identifiers, *, training):
        if len(records)!=len(identifiers) or len(set(identifiers))!=len(identifiers):
            raise ValueError('Unique canonical IDs and aligned records required')
        self.records, self.identifiers, self.training = records, identifiers, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        path,label = self.records[index]
        if label not in (0,1):
            raise ValueError('Binary labels required')
        with sf.SoundFile(path) as stream:
            frames,rate = len(stream),stream.samplerate
            start,length = crop_interval(frames,rate,self.identifiers[index],training=self.training)
            stream.seek(start)
            samples = stream.read(length,dtype='float32',always_2d=True)
        if len(samples)!=length or not np.isfinite(samples).all():
            raise ValueError('Incomplete or nonfinite waveform')
        common = torch.from_numpy(samples).mean(1).contiguous()
        views = [(resample_audio(common,rate,16000).float(),
                  resample_audio(common,rate,48000).float(),label)]
        return {'index':index,'id':self.identifiers[index],'label':label,'rate':rate,'frames':frames,
                'intervals':[[start,length]],'view_rows':[0],'views':views}


class SingleViewBatches(FeatureBatches):
    """Same feature interface, with one actual row and no random view aliases."""
    def __init__(self,*args,sampling='domain_class',**kwargs):
        super().__init__(*args,**kwargs)
        if any(d.view_rows.shape!=(len(d.labels),1) for d in self.datasets):
            raise ValueError('Single-view cache required, not the old three-slot cache')
        if sampling not in ('domain_class','domain'):
            raise ValueError('Unknown source sampling')
        if self.training and sampling=='domain':
            weights = torch.cat([torch.full((len(d.labels),),1/len(d.labels),dtype=torch.double) for d in self.datasets])
            self.sampler = WeightedRandomSampler(weights,len(weights),replacement=True,
                                                generator=torch.Generator().manual_seed(kwargs['seed']))

    def __iter__(self):
        order = (torch.multinomial(self.sampler.weights,self.count,replacement=True,generator=self.sampler.generator)
                 if self.training else torch.arange(self.count))
        transfer = None
        for start in range(0,self.count,self.batch_size):
            if transfer is not None:
                transfer.synchronize()
            selected = order[start:start+self.batch_size]
            size = len(selected)
            for index,dataset in enumerate(self.datasets):
                positions = ((selected>=self.offsets[index])&(selected<self.offsets[index+1])).nonzero().flatten()
                if not len(positions):
                    continue
                records = selected[positions]-self.offsets[index]
                rows = dataset.view_rows[records,0]
                self.buffers['labels'][positions] = dataset.labels[records]
                if 'wavlm' in self.buffers:
                    source = dataset.wavlm[:,-1] if self.last_layer_only and dataset.wavlm.ndim==3 else dataset.wavlm
                    self.buffers['wavlm'][positions] = source[rows]
                if 'clap' in self.buffers:
                    self.buffers['clap'][positions] = dataset.clap[rows]
            batch = {key:value[:size].to(self.device,non_blocking=True)for key,value in self.buffers.items()}
            if self.device.type=='cuda':
                transfer = torch.cuda.Event()
                transfer.record()
            yield batch


def verify_cache_header(root):
    root = Path(root)
    config = json.loads((root/'config.json').read_text())
    done = json.loads((root/'complete.json').read_text())
    if (config.get('schema')!=SCHEMA or done.get('schema')!=SCHEMA or config.get('segment_seconds')!=4
            or config.get('source_splits_only') is not True or config.get('views_per_record')!=1
            or config.get('crop_seed')!=CROP_SEED or config.get('pretrained_revisions')!={'wavlm':WAVLM_REVISION,'clap':CLAP_REVISION}
            or checkpoint_digest(root/'config.json')!=done['config_sha256']
            or set(done['splits'])!={f'{d}/{s}'for d in COUNTS for s in ('train','dev')}
            or checkpoint_digest(root/'prototypes.npy')!=config['prototypes_sha256']):
        raise ValueError('Four-second source cache identity differs or is incomplete')
    return config,done


def load_source_split(root,domain,split,data_root,*,logger=None):
    if domain not in COUNTS or split not in ('train','dev'):
        raise ValueError('Only source train/dev may be loaded')
    root = Path(root)
    config,done = verify_cache_header(root)
    descriptor = done['splits'][f'{domain}/{split}']
    if json.loads((root/domain/split/'complete.json').read_text())!=descriptor:
        raise ValueError('Partition completion changed')
    records = read_dataset(domain,split,data_root)
    ids = [utterance_id(p,domain,data_root)for p,_ in records]
    labels = torch.tensor([label for _,label in records])
    member = hashlib.sha256(json.dumps(list(zip(ids,labels.tolist()))).encode()).hexdigest()
    count = len(records)
    if count!=descriptor['records'] or count!=descriptor['unique_rows'] or member!=descriptor['canonical_ids_labels_sha256']:
        raise ValueError('Source membership/labels changed')
    w = torch.empty(count,13,768,dtype=torch.float32)
    c = torch.empty(count,512,dtype=torch.float32)
    offset = 0
    for index,chunk in enumerate(descriptor['chunks']):
        folder = root/domain/split/chunk['path']
        if chunk['path']!=f'chunk_{index:05d}' or checkpoint_digest(folder/'complete.json')!=chunk['completion_sha256']:
            raise ValueError('Shard completion changed')
        marker = json.loads((folder/'complete.json').read_text())
        for name,digest in marker['files_sha256'].items():
            if checkpoint_digest(folder/name)!=digest:
                raise ValueError('Source cache payload changed')
        metadata = json.loads((folder/'records.json').read_text())
        n = len(metadata)
        if n!=marker['records'] or n!=marker['unique_rows'] or n!=chunk['records']:
            raise ValueError('Single-view shard row count changed')
        for local,row in enumerate(metadata):
            j=offset+local
            expected=crop_interval(row['frames'],row['rate'],f'{domain}/{split}/{ids[j]}',training=split=='train')
            if (row['index']!=j or row['id']!=f'{domain}/{split}/{ids[j]}' or row['label']!=int(labels[j])
                    or row['intervals']!=[list(expected)] or row['view_rows']!=[local]):
                raise ValueError('Wrong single crop, row mapping or canonical label')
        for name,destination,shape in [('wavlm_layers',w,(n,13,768)),('clap_projected',c,(n,512))]:
            array=np.load(folder/f'{name}.npy',mmap_mode='r',allow_pickle=False)
            if array.dtype!=np.float32 or array.shape!=shape or not np.isfinite(array).all():
                raise ValueError('Invalid frozen features')
            np.copyto(destination.numpy()[offset:offset+n],array)
        offset+=n
        if logger is not None and ((index+1)%50==0 or offset==count):
            logger.info('%s/%s single4s cache %d/%d',domain,split,offset,count)
    if offset!=count:
        raise ValueError('Incomplete source partition')
    return FeatureSplit(records,ids,labels,torch.arange(count)[:,None],w,c)


def load_fold(root,fold,data_root,*,logger=None):
    if fold not in FOLDS:
        raise ValueError('Unknown fold')
    return {(d,s):load_source_split(root,d,s,data_root,logger=logger)
            for d in FOLDS[fold]['sources'] for s in ('train','dev')}
