"""Single-crop identity, source-only cache loading and shared model interfaces."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from audio_deepfake_detection.features import frozen_4s as features
from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureSplit
from audio_deepfake_detection.lacf_moe.model import LACFMoE
from audio_deepfake_detection.protocol import checkpoint_digest
from audio_deepfake_detection.sota.wavlm_wa_frozen import FrozenWavLMBasePlusWA
from scripts.feature_cache import cache_4s


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def toy_cache(tmp_path,monkeypatch):
    root=tmp_path/'cache';root.mkdir()
    np.save(root/'prototypes.npy',unit_vectors(torch.randn(8,512)).numpy())
    config={'schema':features.SCHEMA,'segment_seconds':4,'source_splits_only':True,'views_per_record':1,
            'crop_seed':1234,'pretrained_revisions':{'wavlm':features.WAVLM_REVISION,'clap':features.CLAP_REVISION},
            'prototypes_sha256':checkpoint_digest(root/'prototypes.npy')}
    save(root/'config.json',config)
    partitions={};source={}
    for domain in features.COUNTS:
        for split in ('train','dev'):
            records=[(str(tmp_path/domain/split/f'{i}.wav'),i%2)for i in range(4)]
            ids=[f'{i}.wav'for i in range(4)];source[domain,split]=records
            folder=root/domain/split/'chunk_00000';folder.mkdir(parents=True)
            meta=[]
            for i in range(4):
                identifier=f'{domain}/{split}/{ids[i]}'
                meta.append({'index':i,'id':identifier,'label':i%2,'frames':96000,'rate':16000,
                             'intervals':[list(features.crop_interval(96000,16000,identifier,training=split=='train'))],
                             'view_rows':[i]})
            save(folder/'records.json',meta)
            np.save(folder/'wavlm_layers.npy',np.random.default_rng(1234).normal(size=(4,13,768)).astype(np.float32))
            np.save(folder/'clap_projected.npy',np.random.default_rng(1234).normal(size=(4,512)).astype(np.float32))
            hashes={n:checkpoint_digest(folder/n)for n in ('records.json','wavlm_layers.npy','clap_projected.npy')}
            save(folder/'complete.json',{'records':4,'unique_rows':4,'files_sha256':hashes})
            member=hashlib.sha256(json.dumps(list(zip(ids,[0,1,0,1]))).encode()).hexdigest()
            descriptor={'records':4,'unique_rows':4,'canonical_ids_labels_sha256':member,
                        'chunks':[{'path':'chunk_00000','records':4,'unique_rows':4,'completion_sha256':checkpoint_digest(folder/'complete.json')}]}
            save(root/domain/split/'complete.json',descriptor);partitions[f'{domain}/{split}']=descriptor
    save(root/'complete.json',{'schema':features.SCHEMA,'config_sha256':checkpoint_digest(root/'config.json'),'splits':partitions})
    monkeypatch.setattr(features,'read_dataset',lambda d,s,r:source[d,s])
    monkeypatch.setattr(features,'utterance_id',lambda p,d,r:Path(p).name)
    return root


@pytest.mark.parametrize('frames,rate',[(32000,16000),(64000,16000),(64001,16000),(320000,32000)])
def test_crop_limits_short_alias_and_rng_independence(frames,rate):
    before=features.crop_interval(frames,rate,'asv5/train/one.wav',training=True)
    torch.manual_seed(999);torch.rand(100)
    assert before==features.crop_interval(frames,rate,'asv5/train/one.wav',training=True)
    start,length=before
    assert 0<=start<=frames-length and length==min(frames,4*rate)
    assert features.crop_interval(frames,rate,'any',training=False)==((frames-length)//2,length)
    if frames<=4*rate:assert before==(0,frames)


def test_both_branches_use_exact_same_physical_interval(tmp_path):
    rate=32000
    samples=np.stack((np.linspace(-.5,.5,6*rate,dtype=np.float32),np.linspace(.5,-.25,6*rate,dtype=np.float32)),1)
    path=tmp_path/'audio.wav';sf.write(path,samples,rate,subtype='FLOAT')
    row=features.SingleCropDataset([(str(path),0)],['asv5/train/a'],training=True)[0]
    assert len(row['views'])==1 and row['view_rows']==[0]
    start,length=row['intervals'][0]
    common=torch.from_numpy(samples[start:start+length]).mean(1)
    w,c,label=row['views'][0]
    assert label==0 and w.shape==(64000,)and c.shape==(192000,)
    assert torch.equal(w,features.resample_audio(common,rate,16000))
    assert torch.equal(c,features.resample_audio(common,rate,48000))


def test_load_fold_never_loads_held_target_and_both_models_reload(tmp_path,monkeypatch):
    root=toy_cache(tmp_path,monkeypatch)
    bank=features.load_fold(root,'f1',tmp_path)
    assert set(bank)=={(d,s)for d in ('asv2019','asv5','cfad')for s in ('train','dev')}
    assert all(d.view_rows.shape==(4,1)for d in bank.values())
    batch=next(iter(features.SingleViewBatches([bank['asv2019','train']],'concat','cpu',training=True,seed=1234,batch_size=4)))
    for model in (FrozenWavLMBasePlusWA(),LACFMoE(unit_vectors(torch.randn(8,512)))):
        model.eval()
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):a=model(batch)['logit']
        path=tmp_path/(type(model).__name__+'.pt');torch.save(model.state_dict(),path)
        model.load_state_dict(torch.load(path,weights_only=True))
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):b=model(batch)['logit']
        assert torch.equal(a,b)and a.dtype==torch.bfloat16 and torch.isfinite(a).all()
    with pytest.raises(ValueError,match='Only source'):
        features.load_source_split(root,'asv5','eval',tmp_path)


@pytest.mark.parametrize('sampling',['domain','domain_class'])
def test_sampling_seed_and_one_view_per_record(tmp_path,monkeypatch,sampling):
    root=toy_cache(tmp_path,monkeypatch);ds=features.load_source_split(root,'asv2019','train',tmp_path)
    def batches(seed):return list(features.SingleViewBatches([ds],'concat','cpu',training=True,seed=seed,batch_size=3,sampling=sampling))
    first,again=batches(1234),batches(1234)
    assert all(torch.equal(a[k],b[k])for a,b in zip(first,again)for k in a)
    assert not torch.equal(first[0]['wavlm'],batches(1235)[0]['wavlm'])
    for batch in first:
        for value,label in zip(batch['wavlm'],batch['labels']):
            matching=(ds.wavlm==value).all(-1).all(-1).nonzero().flatten()
            assert len(matching)==1 and ds.labels[matching[0]]==label


def test_crop_semantic_tampering_rejected_even_with_fresh_hash_chain(tmp_path,monkeypatch):
    root=toy_cache(tmp_path,monkeypatch);folder=root/'asv2019/train/chunk_00000'
    rows=json.loads((folder/'records.json').read_text());rows[0]['intervals'][0][0]+=1;save(folder/'records.json',rows)
    marker=json.loads((folder/'complete.json').read_text());marker['files_sha256']['records.json']=checkpoint_digest(folder/'records.json');save(folder/'complete.json',marker)
    desc=json.loads((root/'asv2019/train/complete.json').read_text());desc['chunks'][0]['completion_sha256']=checkpoint_digest(folder/'complete.json');save(root/'asv2019/train/complete.json',desc)
    done=json.loads((root/'complete.json').read_text());done['splits']['asv2019/train']=desc;save(root/'complete.json',done)
    with pytest.raises(ValueError,match='Wrong single crop'):
        features.load_source_split(root,'asv2019','train',tmp_path)


def test_ten_second_or_incomplete_cache_rejected(tmp_path,monkeypatch):
    root=toy_cache(tmp_path,monkeypatch)
    config=json.loads((root/'config.json').read_text());config['segment_seconds']=10;save(root/'config.json',config)
    with pytest.raises(ValueError,match='identity'):
        features.verify_cache_header(root)


def test_wavlm_backend_only_gradients_and_weighted_average():
    torch.manual_seed(1234);model=FrozenWavLMBasePlusWA();values=torch.randn(5,13,768)
    assert set(dict(model.named_parameters()))=={'layer_weights','classifier.weight','classifier.bias'}
    assert sum(p.numel()for p in model.parameters())==1551
    result=model({'wavlm':values})
    assert torch.allclose(result['logits'],model.classifier(values.mean(1)),atol=2e-7)
    assert torch.equal(result['logit'],result['logits'][:,1]-result['logits'][:,0])
    loss=torch.nn.functional.cross_entropy(result['logits'],torch.tensor([0,1,0,1,0]));loss.backward()
    assert model.layer_weights.grad is not None and model.classifier.weight.grad is not None
    assert values.grad is None


def test_atomic_cache_shard_and_no_overwrite(tmp_path,monkeypatch):
    launch=tmp_path/'launch';launch.mkdir();save(launch/'source_validation.json',{'passed':True});monkeypatch.setattr(cache_4s,'LAUNCH',launch)
    folder=tmp_path/'source';folder.mkdir()
    row={'index':0,'id':'x','label':0,'frames':64000,'rate':16000,'intervals':[[0,64000]],'view_rows':[0],'views':[{}]}
    arrays={'wavlm_layers':np.zeros((1,13,768),np.float32),'clap_projected':np.zeros((1,512),np.float32)}
    descriptor=cache_4s.write_chunk(folder,0,[row],arrays,{})
    assert descriptor['records']==descriptor['unique_rows']==1
    assert (folder/'chunk_00000/complete.json').is_file()
    with pytest.raises(FileExistsError):cache_4s.write_chunk(folder,0,[row],arrays,{})
    row['views']=[{},{}]
    with pytest.raises(ValueError,match='Exactly one'):cache_4s.write_chunk(folder,1,[row],arrays,{})
