"""Prepared frozen WA Base+ / main MoE source fits; never evaluate targets."""
import argparse
import copy
import gc
import io
import itertools
import json
import logging
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from torch.nn import functional as F

from audio_deepfake_detection.features.frozen_4s import load_fold,verify_cache_header,SingleViewBatches
from audio_deepfake_detection.lacf_moe.model import LACFMoE
from audio_deepfake_detection.sota.wavlm_wa_frozen import FrozenWavLMBasePlusWA
from audio_deepfake_detection.metrics import compute_eer,compute_auroc,calibrate_source_scores,write_score_csv,read_score_csv
from audio_deepfake_detection.protocol import FOLDS,checkpoint_digest
from scripts.feature_cache.cache_4s import save_json

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'outputs/frozen_features/4s'
DATA=Path('/mnt/drive/audio-deepfake-cache')
FAMILIES={'wavlm_wa':ROOT/'outputs/wavlm_wa_frozen_base_plus/4s','lacf_moe':ROOT/'outputs/lacf_moe/4s'}


def recipe(model):
    common={'segment_seconds':4,'train_views':'one fixed random common interval','dev_views':'one centered common interval',
            'precision':'BF16 trainable forward/final scores; sensitive operations FP32','encoders':'frozen Base+ and CLAP; no encoder execution',
            'selection':'strict equal-domain macro source-dev EER; ties keep earlier','patience':5,'accumulation':1,
            'training_scope':'full three-source fold; excluded target untouched','target_access':False}
    if model=='wavlm_wa':
        return {**common,'name':'Frozen WavLM-WA (Base+)','trainable_parameters':1551,'inputs':'weighted',
                'batch_size':96,'sampling':'domain','optimizer':'Adam','lr':.005,'weight_decay':0,
                'scheduler':'ExponentialLR .95','max_epochs':100,'loss':'weighted CE [9,1]',
                'disclosure':'Adapted frozen Base+ and4sdev policy; original full-FT Base baseline preserved'}
    if model=='lacf_moe':
        return {**common,'name':'LACF-MoE (3 experts,4s)','trainable_parameters':967338,'inputs':'concat',
                'batch_size':132,'sampling':'domain_class','optimizer':'AdamW','lr':.0003,'weight_decay':.0001,
                'scheduler':'constant','max_epochs':30,'loss':'detection BCE only','concepts':'original3bona-fide+5spoof',
                'groups':[[0,4],[4,8],[8,13]],'relation_features':35}
    raise ValueError('Unknown prepared model')


def factory(name,prototypes):
    return FrozenWavLMBasePlusWA()if name=='wavlm_wa'else LACFMoE(prototypes)


def batches(bank,fold,config,device,*,training,seed,domain=None,max_batches=None):
    domains=FOLDS[fold]['sources']if training else [domain]
    sets=[bank[d,'train'if training else'dev']for d in domains]
    return SingleViewBatches(sets,config['inputs'],device,training=training,seed=seed,
                            batch_size=config['batch_size'],sampling=config['sampling'],max_batches=max_batches)


def train_epoch(model,loader,optimizer,name,device):
    model.train();total=torch.zeros((),device=device);count=0
    class_weights=torch.tensor([9.,1.],device=device)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(torch.device(device).type,dtype=torch.bfloat16):
            out=model(batch)
        if out['logit'].dtype!=torch.bfloat16:raise TypeError('Expected actual BF16 final scores')
        with torch.autocast(torch.device(device).type,enabled=False):
            loss=(F.cross_entropy(out['logits'].float(),batch['labels'],weight=class_weights)if name=='wavlm_wa'
                  else model.objective(out,batch['labels'])['total'])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite source loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None):
            raise RuntimeError('Nonfinite source gradient')
        optimizer.step();n=len(batch['labels']);total+=loss.detach()*n;count+=n
    return float(total/count)


@torch.inference_mode()
def score_sources(model,bank,fold,config,device,seed):
    model.eval();scores={};metrics={}
    for domain in FOLDS[fold]['sources']:
        values=[]
        for batch in batches(bank,fold,config,device,training=False,seed=seed,domain=domain):
            with torch.autocast(torch.device(device).type,dtype=torch.bfloat16):s=model(batch)['logit']
            if s.dtype!=torch.bfloat16:raise TypeError('Source final logit precision changed')
            values.append(s.float().cpu())
        s=torch.cat(values)
        if not torch.isfinite(s).all():raise ValueError('Nonfinite source scores')
        labels=bank[domain,'dev'].labels.numpy();scores[domain]=s
        metrics[domain]={'eer':compute_eer(labels,s.numpy()),'auroc':compute_auroc(labels,s.numpy())}
    return scores,{'by_domain':metrics,'macro_eer':statistics.mean(r['eer']for r in metrics.values()),
                   'macro_auroc':statistics.mean(r['auroc']for r in metrics.values())}


def fit(bank,prototypes,name,fold,seed,output,provenance,device,*,max_epochs=None):
    expected={(d,s)for d in FOLDS[fold]['sources']for s in ('train','dev')}
    if set(bank)!=expected:raise ValueError('Only current fold source partitions may fit')
    if output.exists():raise FileExistsError('Preserve prior results; no resume/overwrite')
    config=recipe(name);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    # Discard a bounded source update and serialized reload, then reinitialize.
    check=factory(name,prototypes).to(device)
    optimizer=(torch.optim.Adam if name=='wavlm_wa'else torch.optim.AdamW)(check.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    train_epoch(check,batches(bank,fold,config,device,training=True,seed=seed,max_batches=2),optimizer,name,device)
    before,metrics=score_sources(check,bank,fold,config,device,seed)
    stream=io.BytesIO();torch.save(check.state_dict(),stream);stream.seek(0)
    other=factory(name,prototypes).to(device);other.load_state_dict(torch.load(stream,map_location=device,weights_only=True))
    after,again=score_sources(other,bank,fold,config,device,seed)
    if metrics!=again or any(not torch.equal(before[d],after[d])for d in before):
        raise ValueError('Discarded source preflight reload failed; no final fit')
    del check,other,optimizer;gc.collect()
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=factory(name,prototypes).to(device)
    optimizer=(torch.optim.Adam if name=='wavlm_wa'else torch.optim.AdamW)(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,.95)if name=='wavlm_wa'else None
    output.mkdir(parents=True,exist_ok=False)
    complete_config={**config,'model':model.configuration(),'fold':fold,'seed':seed,'sources':list(FOLDS[fold]['sources']),
                     'excluded_target':FOLDS[fold]['target'],'cache_provenance':provenance}
    save_json(output/'config.json',complete_config)
    best=float('inf');best_epoch=0;stale=0;chosen=None
    loader=batches(bank,fold,config,device,training=True,seed=seed)
    for epoch in range(1,(max_epochs or config['max_epochs'])+1):
        loss=train_epoch(model,loader,optimizer,name,device)
        source,metrics=score_sources(model,bank,fold,config,device,seed)
        state={'model_state_dict':{k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
               'epoch':epoch,'fold':fold,'seed':seed,'config':complete_config,'source_dev':metrics}
        temporary=output/'last.pt.tmp';torch.save(state,temporary);temporary.replace(output/'last.pt')
        if metrics['macro_eer']<best:
            best=metrics['macro_eer'];best_epoch=epoch;stale=0;chosen={d:v.clone()for d,v in source.items()}
            temporary=output/'best.pt.tmp';torch.save(state,temporary);temporary.replace(output/'best.pt')
        else:stale+=1
        with (output/'history.jsonl').open('a')as f:f.write(json.dumps({'epoch':epoch,'loss':loss,'source_dev':metrics,'best_epoch':best_epoch,'patience':stale})+'\n')
        logging.info('%s %s epoch%d | sourceEER %.6f%% | best epoch%d',name,fold,epoch,100*metrics['macro_eer'],best_epoch)
        if scheduler:scheduler.step()
        if stale>=config['patience']:break
    saved=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    fresh=factory(name,prototypes).to(device);fresh.load_state_dict(saved['model_state_dict'])
    scores,metrics=score_sources(fresh,bank,fold,config,device,seed)
    if metrics['macro_eer']!=best or any(not torch.equal(chosen[d],scores[d])for d in chosen):
        raise ValueError('Selected full-source reload mismatch')
    exports={};calibration_input={}
    for domain in FOLDS[fold]['sources']:
        ds=bank[domain,'dev'];path=output/f'source_dev_{domain}.csv'
        write_score_csv(path,zip(ds.ids,itertools.repeat(domain),ds.labels.tolist(),scores[domain].tolist()))
        calibration_input[domain]=read_score_csv(path,domain,expected_labels=dict(zip(ds.ids,ds.labels.tolist())))
        exports[path.name]=checkpoint_digest(path)
    calibration=calibrate_source_scores(calibration_input)
    if abs(calibration['source_dev']['macro_eer']-best)>1e-12:raise ValueError('Exported source calibration differs')
    save_json(output/'source_calibration.json',calibration)
    save_json(output/'source_metrics.json',metrics)
    done={'fold':fold,'seed':seed,'best_epoch':best_epoch,'final_epoch':epoch,'best_macro_source_dev_eer':best,
          'selected_checkpoint_reloaded_and_source_verified':True,'full_source_scores_bit_exact':True,
          'best_checkpoint_sha256':checkpoint_digest(output/'best.pt'),'config_sha256':checkpoint_digest(output/'config.json'),
          'source_scores_sha256':exports,'source_calibration_sha256':checkpoint_digest(output/'source_calibration.json'),
          'cache_provenance':provenance,'target_access':False}
    save_json(output/'training_complete.json',done)
    return done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=FAMILIES,required=True)
    parser.add_argument('--fold',choices=FOLDS,default='f1')
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--prepare',action='store_true',help='Write recipe only; no cache load, CUDA or fit')
    args=parser.parse_args();family=FAMILIES[args.model]
    if args.prepare:
        family.mkdir(parents=True,exist_ok=True)
        save_json(family/'preparation.json',{'recipe':recipe(args.model),'cache':str(CACHE),'folds':list(FOLDS),
                  'seed_plan':[1234],'training_started':False,'pending':'Complete4s source bank; separate explicit fit launch'})
        return
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    output=family/args.fold/str(args.seed)
    if output.exists():raise FileExistsError('Prior final run preserved')
    config,done=verify_cache_header(CACHE)
    if not torch.cuda.is_available()or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('CUDA with2GiBfree required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    bank=load_fold(CACHE,args.fold,DATA,logger=logging.getLogger(__name__))
    prototypes=torch.from_numpy(np.load(CACHE/'prototypes.npy',allow_pickle=False))
    proof={'cache':str(CACHE),'config_sha256':checkpoint_digest(CACHE/'config.json'),'completion_sha256':checkpoint_digest(CACHE/'complete.json')}
    fit(bank,prototypes,args.model,args.fold,args.seed,output,proof,'cuda')


if __name__=='__main__':main()
