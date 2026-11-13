"""Fresh four-expert LACF folds, seed1234; sequential source-only validation."""

import argparse
import csv
import gc
import itertools
import json
import logging
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from audio_deepfake_detection.experiments.layer_fusion.four_experts import FourExpertDetector, GROUPS, TRAINABLE_PARAMETERS
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_split, score_probe
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, save_score_manifest, verify_completed_run, verify_source_macro
from scripts.experiments.layer_fusion.train_full_f1 import checkpoint
from scripts.lacf.train_ablations import fixed_prototypes
from scripts.lacf.train_probes import save_json, seed_all
from scripts.token_fusion.train import ROOT, check_guards


FAMILY = ROOT / 'outputs/experiments/layer_fusion/lacf_layer_moe4'
OUTPUT = FAMILY / 'f1/1234'
LAUNCH = FAMILY / 'launch'
SOURCES = FOLDS['f1']['sources']  # Default retained for the original F1 API.


def sources_for(fold):
    if fold not in FOLDS:
        raise ValueError('Expected a canonical F1-F4 fold')
    return FOLDS[fold]['sources']


def assert_bank(bank, fold='f1'):
    if set(bank) != {(d,s) for d in sources_for(fold) for s in ('train','dev')}:
        raise ValueError('Only the selected fold source train/dev partitions may enter training')


def load_sources(cache, data_root, folds, logger):
    manifest = json.loads((cache/'complete.json').read_text())
    if (manifest.get('step') != 2 or checkpoint_digest(cache/'config.json') != manifest['config_sha256']
            or checkpoint_digest(cache/'execution_accelerated.json') != manifest['accelerated_execution_sha256']):
        raise ValueError('Frozen source cache identity differs')
    domains = tuple(dict.fromkeys(d for fold in folds for d in sources_for(fold)))
    bank = {}
    for domain,split in itertools.product(domains,('train','dev')):
        logger.info('CACHE checking/loading %s/%s once for requested source folds',domain,split)
        data = load_split(cache,domain,split,manifest['splits'][f'{domain}/{split}'],data_root,logger=logger)
        if not torch.isfinite(data.wavlm).all() or not torch.isfinite(data.clap).all():
            raise ValueError('Nonfinite source features')
        bank[domain,split] = data
    return bank


def score_sources(model, bank, device, logger=None, *, fold='f1'):
    scores,domains = {},{}
    for domain in sources_for(fold):
        loader = FeatureBatches([bank[domain,'dev']],'concat',device,training=False,seed=1234)
        scores[domain],domains[domain] = score_probe(model,loader)
        if logger:
            logger.info('%s DEV %s | EER %.4f%% AUROC %.6f',fold.upper(),domain,
                        domains[domain]['eer']*100,domains[domain]['auroc'])
    return scores,{'by_domain':domains,'macro_eer':statistics.mean(v['eer'] for v in domains.values()),
                  'macro_auroc':statistics.mean(v['auroc'] for v in domains.values())}


@torch.inference_mode()
def routing_summary(model, bank, device, *, fold='f1'):
    model.eval()
    result = {'expert_count':4, 'groups_half_open':[list(g) for g in GROUPS], 'domains':{},
              'scope':'source development only; descriptive, never checkpoint selection',
              'interpretation':'Uniform-routing argmax ties select expert0; top counts alone do not establish collapse or domain shortcuts'}
    for domain in sources_for(fold):
        totals = torch.zeros(3,14,dtype=torch.float64,device=device)
        loader = FeatureBatches([bank[domain,'dev']], 'weighted', device, training=False, seed=1234)
        for batch in loader:
            with torch.autocast(torch.device(device).type,dtype=torch.bfloat16):
                weights = model.layer_fusion.routing_weights(batch['wavlm'])
            entropy = -(weights*weights.clamp_min(1e-8).log()).sum(-1,keepdim=True)
            top = torch.nn.functional.one_hot(weights.argmax(-1),4)
            values = torch.cat((torch.ones_like(entropy),weights,weights.square(),entropy,top),-1).double()
            masks = torch.stack((torch.ones_like(batch['labels']),batch['labels']==0,batch['labels']==1)).double()
            totals += masks @ values
        groups = {}
        for name,row in zip(('all','bonafide','spoof'),totals.cpu()):
            count = int(row[0]);groups[name] = {'count':count}
            if count:
                mean = row[1:5]/count
                groups[name].update(mean_weights=mean.tolist(),std_weights=(row[5:9]/count-mean.square()).clamp_min(0).sqrt().tolist(),
                                    mean_entropy_nats=float(row[9]/count),top_expert_counts=row[10:14].long().tolist())
        result['domains'][domain] = groups
    return result


def verify_completion(run, fold='f1'):
    sources = sources_for(fold)
    done,digest = verify_completed_run(run,fold,1234)
    config = json.loads((run/'config.json').read_text())
    if (config['model']['expert_count'] != 4 or config['model']['trainable_parameters'] != TRAINABLE_PARAMETERS
            or config['sources'] != list(sources) or config['model']['objective'] != 'detection BCE only'
            or done['best_checkpoint_sha256'] != digest or not done['selected_checkpoint_reloaded_and_source_verified']):
        raise ValueError('Four-expert completion identity differs')
    for filename,key in (('config.json','config_sha256'),('metrics.json','metrics_sha256'),
                         ('routing.json','routing_sha256'),('source_calibration.json','calibration_sha256')):
        if checkpoint_digest(run/filename) != done[key]:
            raise ValueError('Completion-bound artifact changed: '+filename)
    source = {}
    for domain in sources:
        filename = f'source_dev_{domain}.csv'
        if checkpoint_digest(run/filename) != done['source_scores_sha256'][filename]:
            raise ValueError('Source export changed')
        # Match fit()'s sorted-ID calibration order: floating-point reductions
        # can differ in their last bits when the same CSV is read in file order.
        with (run/filename).open(newline='') as stream:
            labels = {row['utterance_id']: int(row['label']) for row in csv.DictReader(stream)}
        source[domain] = read_score_csv(run/filename,domain,expected_labels=labels)
    calibration = calibrate_source_scores(source)
    if calibration != json.loads((run/'source_calibration.json').read_text()):
        raise ValueError('Frozen source calibration differs')
    verify_source_macro(calibration['source_dev']['macro_eer'],done['best_macro_source_dev_eer'])
    return done


def fit(bank,prototypes,output,provenance,device,logger,*,fold='f1',max_epochs=30,patience=5):
    sources = sources_for(fold)
    assert_bank(bank,fold);check_guards(provenance)
    output.mkdir(parents=True,exist_ok=False)
    handler = logging.FileHandler(output/'training.log');handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler)
    try:
        seed_all(1234)
        model = FourExpertDetector(prototypes).to(device);optimizer = make_optimizer(model)
        train = FeatureBatches([bank[d,'train'] for d in sources],'concat',device,training=True,seed=1234)
        config = {'model_name':'LACF Layer-MoE (4 experts)','fold':fold,'seed':1234,'seed_plan':[1234],
            'sources':list(sources),'excluded_target':FOLDS[fold]['target'],'target_access':False,'model':model.configuration(),
            'scope':f'full {fold.upper()} source train/dev; exploratory source selection, not inner-held-domain validation',
            'batch_size':132,'accumulation':1,'optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'scheduler':'constant',
            'max_epochs':max_epochs,'patience':patience,'precision':'BF16 mixed; sensitive FP32',
            'sampling':'equal domain/class mass; replacement; recording count per epoch',
            'training_views':'centered plus two fixed random common up-to10s views; uniform logical choice',
            'development_views':'centered cached native unpadded features; head batch132',
            'selection':'strict equal-domain macro source-dev EER; ties earlier','resume':False,'foundation_encoders_loaded':False,
            'feature_workers':0,'train_records':train.count,'dev_records':{d:len(bank[d,'dev'].labels) for d in sources},
            'comparison':'same original text bank/backend/objective/recipe as completed three-expert MoE; group partition and parameter count change',
            'provenance':provenance}
        save_json(output/'config.json',config);digest=checkpoint_digest(output/'config.json')
        best,best_epoch,stale,selected_scores=float('inf'),0,0,None
        logger.info('START four-expert LACF %s | %d training recordings | %d trainable | detection BCE only',fold.upper(),train.count,TRAINABLE_PARAMETERS)
        with (output/'history.jsonl').open('x') as history:
            for epoch in range(1,max_epochs+1):
                check_guards(provenance);tick=time.monotonic()
                losses,updates=train_epoch(model,train,optimizer,device,accumulation_steps=1,logger=logger,phase=f'MoE4 epoch{epoch}')
                scores,metrics=score_sources(model,bank,device,logger,fold=fold)
                if metrics['macro_eer']<best:
                    best,best_epoch,stale,selected_scores=metrics['macro_eer'],epoch,0,scores
                    checkpoint(output/'best.pt',{'epoch':epoch,'config_sha256':digest,'source_dev':metrics,'model_state_dict':model.state_dict()})
                else:stale+=1
                checkpoint(output/'last.pt',{'epoch':epoch,'config_sha256':digest,'source_dev':metrics,
                    'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'best_epoch':best_epoch,'patience':stale})
                row={'epoch':epoch,'losses':losses,'updates':updates,'source_dev':metrics,'best_epoch':best_epoch,'patience':stale,'seconds':time.monotonic()-tick}
                history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
                logger.info('EPOCH %d | source EER %.4f%% AUROC %.6f | best%d patience%d/%d | %.1fs',epoch,metrics['macro_eer']*100,metrics['macro_auroc'],best_epoch,stale,patience,row['seconds'])
                if stale>=patience:break
        state=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        if state['config_sha256']!=digest or checkpoint_digest(output/'config.json')!=digest:
            raise ValueError('Selected checkpoint/config differs')
        del model,optimizer;gc.collect()
        model=FourExpertDetector(prototypes).to(device);model.load_state_dict(state['model_state_dict'],strict=True)
        scores,metrics=score_sources(model,bank,device,logger,fold=fold)
        if metrics!=state['source_dev'] or any(not torch.equal(scores[d],selected_scores[d]) for d in sources):
            raise ValueError('Fresh selected-checkpoint full source reload differs')
        inputs,hashes={},{}
        for domain in sources:
            data=bank[domain,'dev'];filename=f'source_dev_{domain}.csv';path=output/filename
            write_score_csv(path,zip(data.ids,itertools.repeat(domain),data.labels.tolist(),scores[domain].tolist()))
            inputs[domain]=read_score_csv(path,domain,expected_labels=dict(zip(data.ids,data.labels.tolist())))
            order=sorted(range(len(data.ids)),key=data.ids.__getitem__)
            if not np.array_equal(inputs[domain][1],scores[domain].numpy()[order]):raise ValueError('Canonical source export differs')
            hashes[filename]=checkpoint_digest(path)
        calibration=calibrate_source_scores(inputs);verify_source_macro(calibration['source_dev']['macro_eer'],best)
        save_json(output/'source_calibration.json',calibration)
        save_json(output/'routing.json',routing_summary(model,bank,device,fold=fold))
        save_json(output/'metrics.json',{'model_name':config['model_name'],'fold':fold,'seed':1234,'source_dev':metrics,'best_epoch':best_epoch,'final_epoch':epoch,'target_access':False})
        done={'fold':fold,'seed':1234,'best_epoch':best_epoch,'final_epoch':epoch,'best_macro_source_dev_eer':best,
            'selected_checkpoint_reloaded_and_source_verified':True,'best_checkpoint_sha256':checkpoint_digest(output/'best.pt'),
            'config_sha256':digest,'metrics_sha256':checkpoint_digest(output/'metrics.json'),'source_scores_sha256':hashes,
            'calibration_sha256':checkpoint_digest(output/'source_calibration.json'),'routing_sha256':checkpoint_digest(output/'routing.json'),'target_access':False}
        save_json(output/'training_complete.json',done)
        save_score_manifest(output,{'fold':fold,'seed':1234,'sources':list(sources),'target_access':False,
            'checkpoint_sha256':done['best_checkpoint_sha256'],'training_completion_sha256':checkpoint_digest(output/'training_complete.json'),
            'score_sha256':hashes})
        check_guards(provenance);verify_completion(output,fold)
        logger.info('COMPLETE four-expert %s | epoch%d source EER %.4f%%; reload/export/calibration passed; no target',fold.upper(),best_epoch,best*100)
        return done
    finally:
        logger.removeHandler(handler);handler.close()


def preflight(bank,prototypes,device,logger,*,fold='f1',launch=LAUNCH):
    sources = sources_for(fold)
    seed_all(1234);model=FourExpertDetector(prototypes).to(device);optimizer=make_optimizer(model)
    loader=FeatureBatches([bank[d,'train'] for d in sources],'concat',device,training=True,seed=1234,max_batches=32)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
    losses,updates=train_epoch(model,loader,optimizer,device,accumulation_steps=1)
    torch.cuda.synchronize();train_seconds=time.monotonic()-tick
    scores,_=score_sources(model,bank,device,logger,fold=fold)
    checkpoint(launch/f'preflight_{fold}.pt',{'model_state_dict':model.state_dict()})
    fresh=FourExpertDetector(prototypes).to(device)
    fresh.load_state_dict(torch.load(launch/f'preflight_{fold}.pt',map_location='cpu',weights_only=True)['model_state_dict'])
    tick=time.monotonic();other,_=score_sources(fresh,bank,device,logger,fold=fold);dev_seconds=time.monotonic()-tick
    if any(not torch.equal(scores[d],other[d]) for d in sources):raise ValueError('Source preflight reload differs')
    record={'passed':True,'target_access':False,'losses':losses,'updates':updates,'source_reload_bit_exact':True,
        'train_examples_per_second':loader.count/train_seconds,'dev_examples_per_second':sum(len(v) for v in other.values())/dev_seconds,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'probe_weights_used_for_final_training':False}
    record['fold']=fold
    record['rough_epoch_seconds']=sum(len(bank[d,'train'].labels) for d in sources)/record['train_examples_per_second']+dev_seconds
    logger.info('PREFLIGHT PASSED | %.1f train audio/s | %.1fs rough epoch | %.3fGiB allocated',record['train_examples_per_second'],record['rough_epoch_seconds'],record['peak_allocated_bytes']/1024**3)
    del model,optimizer,fresh;gc.collect();torch.cuda.empty_cache()
    return record


def run_folds(bank, prototypes, folds, provenance, device, logger, launch):
    for fold in folds:
        subset={(d,s):bank[d,s] for d in sources_for(fold) for s in ('train','dev')}
        validation=launch/f'source_validation_{fold}.json'
        save_json(validation,preflight(subset,prototypes,device,logger,fold=fold,launch=launch))
        current={**provenance,'source_validation_sha256':checkpoint_digest(validation),'source_validation_record':str(validation)}
        fit(subset,prototypes,FAMILY/fold/'1234',current,device,logger,fold=fold)
        verify_completion(FAMILY/fold/'1234',fold)
        logger.info('%s verified complete; next fold may start fresh',fold.upper())
        del subset;gc.collect()
        if torch.device(device).type=='cuda':torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds',nargs='+',choices=tuple(FOLDS),default=['f1'])
    args=parser.parse_args()
    if len(set(args.folds))!=len(args.folds):parser.error('Do not repeat folds')
    if args.folds!=sorted(args.folds):parser.error('Use canonical sequential fold order')
    launch=LAUNCH if args.folds==['f1'] else FAMILY/('launch_'+'_'.join(args.folds))
    for fold in args.folds:
        if (FAMILY/fold/'1234').exists():raise FileExistsError('Existing run preserved; no overwrite or resume: '+fold)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');logger=logging.getLogger('lacf.moe4')
    provenance=json.loads((launch/'provenance.json').read_text());check_guards(provenance)
    cache=Path(provenance['cache'])
    for name,key in (('complete.json','cache_completion_sha256'),('config.json','cache_config_sha256'),('execution_accelerated.json','cache_execution_sha256')):
        if checkpoint_digest(cache/name)!=provenance[key]:raise ValueError('Frozen cache identity changed')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):raise RuntimeError('Native CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.nnpack.set_flags(False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    bank=load_sources(cache,Path('/mnt/drive/audio-deepfake-cache'),args.folds,logger)
    prototypes,evidence=fixed_prototypes();provenance['prototypes']=evidence
    run_folds(bank,prototypes,args.folds,provenance,'cuda',logger,launch)
    save_json(launch/'complete.json',{'folds':args.folds,'seed':1234,'target_access':False,
        'completion_sha256':{f:checkpoint_digest(FAMILY/f/'1234/training_complete.json') for f in args.folds}})
    if all((FAMILY/f/'1234/training_complete.json').exists() for f in FOLDS):
        rows=[]
        for fold in FOLDS:
            verify_completion(FAMILY/fold/'1234',fold)
            metrics=json.loads((FAMILY/fold/'1234/metrics.json').read_text())
            rows.append({'fold':fold,'seed':1234,'source_dev':metrics['source_dev'],
                         'best_epoch':metrics['best_epoch'],'final_epoch':metrics['final_epoch']})
        save_json(FAMILY/'source_summary.json',{'scope':'four independently trained source-selected folds; no target performance',
                  'seed':1234,'folds':rows,'target_access':False})


if __name__=='__main__':main()
