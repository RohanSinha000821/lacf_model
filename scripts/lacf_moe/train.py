"""Main cached three-expert LACF-MoE: preserve F1, train fresh F2→F3→F4."""

import gc
import itertools
import json
import logging
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from audio_deepfake_detection.lacf_moe.model import GROUPS, LACFMoE, TRAINABLE_PARAMETERS
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_bank, score_probe
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, canonical_score_labels, checkpoint_digest, save_score_manifest, verify_completed_run, verify_source_macro
from scripts.lacf.train_probes import save_json, seed_all
from scripts.lacf.train_ablations import fixed_prototypes

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / 'outputs/lacf_moe'
CACHE = ROOT / 'outputs/experiments/lacf_diagnostics/frozen_features'
DATA = Path('/mnt/drive/audio-deepfake-cache')
NAME = 'LACF-MoE'


def read(path):
    return json.loads(path.read_text())


def guards(provenance):
    for path, digest in provenance['code_sha256'].items():
        if checkpoint_digest(ROOT / path) != digest:
            raise ValueError('Execution source changed: ' + path)
    for filename, key in (('config.json','cache_config_sha256'), ('complete.json','cache_completion_sha256'),
                          ('execution_accelerated.json','cache_execution_sha256')):
        if key in provenance and checkpoint_digest(Path(provenance['cache']) / filename) != provenance[key]:
            raise ValueError('Frozen cache identity changed')


def assert_bank(bank, fold):
    if set(bank) != {(domain, split) for domain in FOLDS[fold]['sources'] for split in ('train','dev')}:
        raise ValueError('Only this fold source train/dev partitions may enter fitting')


def checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    if temporary.exists():
        raise FileExistsError('Interrupted checkpoint preserved')
    torch.save(value, temporary)
    temporary.replace(path)


def score_sources(model, bank, fold, device, logger=None):
    scores, metrics = {}, {}
    for domain in FOLDS[fold]['sources']:
        scores[domain], metrics[domain] = score_probe(model, FeatureBatches(
            [bank[domain,'dev']], 'concat', device, training=False, seed=1234))
        if logger:
            logger.info('%s DEV %s | EER %.4f%% AUROC %.6f', fold.upper(), domain,
                        metrics[domain]['eer']*100, metrics[domain]['auroc'])
    return scores, {'by_domain':metrics, 'macro_eer':statistics.mean(v['eer'] for v in metrics.values()),
                    'macro_auroc':statistics.mean(v['auroc'] for v in metrics.values())}


@torch.inference_mode()
def routing_summary(model, bank, fold, device):
    model.eval()
    result = {'groups_half_open':[list(g) for g in GROUPS], 'expert_count':3, 'domains':{},
              'scope':'source development only; descriptive, never model selection',
              'interpretation':'Argmax ties select expert0; inspect soft weights/entropy, not top counts alone'}
    for domain in FOLDS[fold]['sources']:
        totals = torch.zeros(3,11,dtype=torch.float64,device=device)
        for batch in FeatureBatches([bank[domain,'dev']], 'weighted', device, training=False, seed=1234):
            with torch.autocast(torch.device(device).type,dtype=torch.bfloat16):
                weights = model.layer_fusion.routing_weights(batch['wavlm'])
            entropy = -(weights*weights.clamp_min(1e-8).log()).sum(-1,keepdim=True)
            weights64 = weights.double()
            values = torch.cat((torch.ones_like(entropy).double(),weights64,weights64.square(),entropy.double(),
                                torch.nn.functional.one_hot(weights.argmax(-1),3).double()),-1)
            masks = torch.stack((torch.ones_like(batch['labels']),batch['labels']==0,batch['labels']==1)).double()
            totals += masks @ values
        groups = {}
        for name,row in zip(('all','bonafide','spoof'),totals.cpu()):
            count = int(row[0]);groups[name] = {'count':count}
            if count:
                mean = row[1:4]/count
                groups[name].update(mean_weights=mean.tolist(),std_weights=(row[4:7]/count-mean.square()).clamp_min(0).sqrt().tolist(),
                                    mean_entropy_nats=float(row[7]/count),top_expert_counts=row[8:11].long().tolist())
        result['domains'][domain] = groups
    return result


def verify_completion(run, fold, *, data_root=DATA, expected_labels=None):
    done, digest = verify_completed_run(run,fold,1234)
    config = read(run/'config.json')
    if (config['sources'] != list(FOLDS[fold]['sources']) or config['seed_plan'] != [1234]
            or config['target_access'] is not False or config['model']['trainable_parameters'] != TRAINABLE_PARAMETERS
            or config['model']['expert_groups_half_open'] != [list(g) for g in GROUPS]
            or config['model']['objective'] != 'detection BCE only' or done['best_checkpoint_sha256'] != digest
            or not done['selected_checkpoint_reloaded_and_source_verified']):
        raise ValueError('LACF-MoE completion identity differs')
    for filename,key in (('config.json','config_sha256'),('metrics.json','metrics_sha256'),
                         ('routing.json','routing_sha256'),('source_calibration.json','calibration_sha256')):
        if checkpoint_digest(run/filename) != done[key]:
            raise ValueError('Completion-bound artifact changed: ' + filename)
    manifest = read(run/'score_manifest.json')
    if (manifest['fold'] != fold or manifest['seed'] != 1234 or manifest['sources'] != list(FOLDS[fold]['sources'])
            or manifest['checkpoint_sha256'] != digest
            or manifest['training_completion_sha256'] != checkpoint_digest(run/'training_complete.json')
            or manifest['score_sha256'] != done['source_scores_sha256']):
        raise ValueError('Source manifest differs')
    inputs = {}
    for domain in FOLDS[fold]['sources']:
        filename = f'source_dev_{domain}.csv'
        if checkpoint_digest(run/filename) != done['source_scores_sha256'][filename]:
            raise ValueError('Source export changed')
        labels = expected_labels[domain] if expected_labels is not None else canonical_score_labels(domain,'dev',data_root)
        inputs[domain] = read_score_csv(run/filename,domain,expected_labels=labels)
    calibration = calibrate_source_scores(inputs)
    if calibration != read(run/'source_calibration.json'):
        raise ValueError('Frozen source calibration differs')
    verify_source_macro(calibration['source_dev']['macro_eer'],done['best_macro_source_dev_eer'])
    state = torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    if (state['config_sha256'] != checkpoint_digest(run/'config.json') or state['epoch'] != done['best_epoch']
            or state['source_dev'] != read(run/'metrics.json')['source_dev']):
        raise ValueError('Selected checkpoint/config/source result differs')
    return done


def fit(bank, prototypes, output, fold, provenance, device, logger, *, max_epochs=30, patience=5):
    assert_bank(bank,fold);guards(provenance)
    output.mkdir(parents=True,exist_ok=False)
    handler = logging.FileHandler(output/'training.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));logger.addHandler(handler)
    try:
        seed_all(1234);model=LACFMoE(prototypes).to(device);optimizer=make_optimizer(model)
        sources=FOLDS[fold]['sources']
        train=FeatureBatches([bank[d,'train'] for d in sources],'concat',device,training=True,seed=1234)
        config={'model_name':NAME,'fold':fold,'seed':1234,'seed_plan':[1234],'sources':list(sources),
                'excluded_target':FOLDS[fold]['target'],'target_access':False,'model':model.configuration(),
                'batch_size':132,'accumulation':1,'optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'scheduler':'constant',
                'max_epochs':max_epochs,'patience':patience,'precision':'BF16 mixed; sensitive FP32',
                'sampling':'equal domain/class mass; replacement; recording count per epoch',
                'training_views':'common up-to10s; centered plus two fixed random views, uniform per draw',
                'development_views':'centered up-to10s; cached native unpadded encoder features; head batch132',
                'selection':'strict equal-domain macro source-dev EER; ties earlier','resume':False,'feature_workers':0,
                'train_records':train.count,'dev_records':{d:len(bank[d,'dev'].labels) for d in sources},
                'scope':'full source folds, seed1234; exploratory after prior target exposure, not inner-source selection',
                'provenance':provenance}
        save_json(output/'config.json',config);digest=checkpoint_digest(output/'config.json')
        best,best_epoch,stale,selected_scores=float('inf'),0,0,None
        logger.info('START %s %s | %d records | %d trainable | original 3 experts; frozen cached encoders',NAME,fold.upper(),train.count,TRAINABLE_PARAMETERS)
        with (output/'history.jsonl').open('x') as history:
            for epoch in range(1,max_epochs+1):
                guards(provenance);start=time.monotonic()
                losses,updates=train_epoch(model,train,optimizer,device,accumulation_steps=1,logger=logger,phase=f'{fold.upper()} epoch{epoch}')
                scores,metrics=score_sources(model,bank,fold,device,logger)
                if metrics['macro_eer']<best:
                    best,best_epoch,stale,selected_scores=metrics['macro_eer'],epoch,0,scores
                    checkpoint(output/'best.pt',{'model_state_dict':model.state_dict(),'epoch':epoch,'source_dev':metrics,'config_sha256':digest})
                else:stale+=1
                checkpoint(output/'last.pt',{'model_state_dict':model.state_dict(),'epoch':epoch,'source_dev':metrics,
                           'optimizer_state_dict':optimizer.state_dict(),'best_epoch':best_epoch,'patience':stale,'config_sha256':digest})
                row={'epoch':epoch,'losses':losses,'updates':updates,'source_dev':metrics,'best_epoch':best_epoch,
                     'patience':stale,'seconds':time.monotonic()-start}
                history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
                logger.info('EPOCH %d | source EER %.4f%% AUROC %.6f | best%d patience%d/%d | %.1fs',epoch,
                            metrics['macro_eer']*100,metrics['macro_auroc'],best_epoch,stale,patience,row['seconds'])
                if stale>=patience:break
        state=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        del model,optimizer;gc.collect()
        model=LACFMoE(prototypes).to(device);model.load_state_dict(state['model_state_dict'],strict=True)
        scores,metrics=score_sources(model,bank,fold,device,logger)
        if state['config_sha256']!=digest or checkpoint_digest(output/'config.json')!=digest or metrics!=state['source_dev'] or any(not torch.equal(scores[d],selected_scores[d]) for d in sources):
            raise ValueError('Fresh selected-checkpoint source reload differs')
        inputs,hashes={},{}
        labels={d:dict(zip(bank[d,'dev'].ids,bank[d,'dev'].labels.tolist())) for d in sources}
        for domain in sources:
            data=bank[domain,'dev'];path=output/f'source_dev_{domain}.csv'
            write_score_csv(path,zip(data.ids,itertools.repeat(domain),data.labels.tolist(),scores[domain].tolist()))
            inputs[domain]=read_score_csv(path,domain,expected_labels=labels[domain])
            order=sorted(range(len(data.ids)),key=data.ids.__getitem__)
            if not np.array_equal(inputs[domain][1],scores[domain].numpy()[order]):
                raise ValueError('Canonical source export differs')
            hashes[path.name]=checkpoint_digest(path)
        calibration=calibrate_source_scores(inputs);verify_source_macro(calibration['source_dev']['macro_eer'],best)
        save_json(output/'source_calibration.json',calibration)
        save_json(output/'routing.json',routing_summary(model,bank,fold,device))
        save_json(output/'metrics.json',{'model_name':NAME,'fold':fold,'seed':1234,'source_dev':metrics,
                                       'best_epoch':best_epoch,'final_epoch':epoch,'target_access':False})
        done={'model_name':NAME,'fold':fold,'seed':1234,'sources':list(sources),'best_epoch':best_epoch,'final_epoch':epoch,
              'best_macro_source_dev_eer':best,'selected_checkpoint_reloaded_and_source_verified':True,
              'best_checkpoint_sha256':checkpoint_digest(output/'best.pt'),'config_sha256':digest,
              'metrics_sha256':checkpoint_digest(output/'metrics.json'),'source_scores_sha256':hashes,
              'calibration_sha256':checkpoint_digest(output/'source_calibration.json'),
              'routing_sha256':checkpoint_digest(output/'routing.json'),'target_access':False}
        save_json(output/'training_complete.json',done)
        save_score_manifest(output,{'fold':fold,'seed':1234,'sources':list(sources),'checkpoint_sha256':done['best_checkpoint_sha256'],
                            'training_completion_sha256':checkpoint_digest(output/'training_complete.json'),'score_sha256':hashes,'target_access':False})
        guards(provenance);verify_completion(output,fold,expected_labels=labels)
        logger.info('COMPLETE %s | source EER %.4f%% | reload/export/calibration passed; target unopened',fold.upper(),best*100)
        return done
    finally:
        logger.removeHandler(handler);handler.close()


def preflight(bank, prototypes, fold, provenance, device, logger):
    guards(provenance);assert_bank(bank,fold);seed_all(1234)
    model=LACFMoE(prototypes).to(device);optimizer=make_optimizer(model)
    loader=FeatureBatches([bank[d,'train'] for d in FOLDS[fold]['sources']],'concat',device,training=True,seed=1234,max_batches=32)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    losses,updates=train_epoch(model,loader,optimizer,device,accumulation_steps=1)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    expected,_=score_sources(model,bank,fold,device,logger)
    fresh=LACFMoE(prototypes).to(device);fresh.load_state_dict(state,strict=True)
    start=time.monotonic();actual,_=score_sources(fresh,bank,fold,device,logger);torch.cuda.synchronize()
    dev_seconds=time.monotonic()-start
    if any(not torch.equal(expected[d],actual[d]) for d in expected):raise ValueError('Source preflight reload differs')
    report={'passed':True,'fold':fold,'losses':losses,'updates':updates,'source_reload_bit_exact':True,
            'train_examples_per_second':loader.count/elapsed,'source_dev_seconds':dev_seconds,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'probe_weights_used_for_final_training':False,'target_access':False}
    report['rough_epoch_seconds']=sum(len(bank[d,'train'].labels) for d in expected)/report['train_examples_per_second']+dev_seconds
    logger.info('PREFLIGHT %s passed | %.1f train audio/s | %.1fs estimated epoch',fold.upper(),report['train_examples_per_second'],report['rough_epoch_seconds'])
    del model,optimizer,fresh;gc.collect();torch.cuda.empty_cache()
    return report


def run_folds(bank, prototypes, folds, provenance, device, logger, output, launch):
    for fold in folds:
        subset={(d,s):bank[d,s] for d in FOLDS[fold]['sources'] for s in ('train','dev')}
        validation=launch/f'source_validation_{fold}.json'
        save_json(validation,preflight(subset,prototypes,fold,provenance,device,logger))
        current={**provenance,'source_validation_record':str(validation),'source_validation_sha256':checkpoint_digest(validation)}
        fit(subset,prototypes,output/fold/'1234',fold,current,device,logger)
        verify_completion(output/fold/'1234',fold,expected_labels={d:dict(zip(bank[d,'dev'].ids,bank[d,'dev'].labels.tolist())) for d in FOLDS[fold]['sources']})
        logger.info('%s completion verified; advance only now',fold.upper())
        del subset;gc.collect()
        if torch.device(device).type=='cuda':torch.cuda.empty_cache()


def main():
    launch=FAMILY/'launch'
    if (launch/'started.json').exists() or any((FAMILY/f/'1234').exists() for f in ('f2','f3','f4')):
        raise FileExistsError('Prior launch/run preserved; no automatic resume or retry')
    provenance=read(launch/'provenance.json');guards(provenance)
    original=verify_completion(FAMILY/'f1/1234','f1')
    if checkpoint_digest(FAMILY/'f1/1234/training_complete.json')!=provenance['preserved_f1_completion_sha256']:
        raise ValueError('Preserved F1 completion changed')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.nnpack.set_flags(False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');logger=logging.getLogger('lacf_moe')
    save_json(launch/'started.json',{'phase':'cache verification/loading','preserved_f1_best_epoch':original['best_epoch'],'target_access':False})
    bank=load_bank(CACHE,DATA,logger=logger)
    if any(not torch.isfinite(data.wavlm).all() or not torch.isfinite(data.clap).all() for data in bank.values()):
        raise ValueError('Nonfinite source features')
    # Validate the main-folder implementation on every preserved F1 source score.
    state=torch.load(FAMILY/'f1/1234/best.pt',map_location='cpu',weights_only=True)
    model=LACFMoE(state['model_state_dict']['backend.prototypes']).to('cuda')
    model.load_state_dict(state['model_state_dict'],strict=True)
    scores,metrics=score_sources(model,bank,'f1','cuda',logger)
    if metrics!=state['source_dev']:raise ValueError('Preserved F1 source metrics differ')
    for domain in FOLDS['f1']['sources']:
        ds=bank[domain,'dev'];_,saved=read_score_csv(FAMILY/f'f1/1234/source_dev_{domain}.csv',domain,
                                                  expected_labels=dict(zip(ds.ids,ds.labels.tolist())))
        order=sorted(range(len(ds.ids)),key=ds.ids.__getitem__)
        if not np.array_equal(scores[domain].numpy()[order],saved):raise ValueError('Preserved F1 source reload differs')
    save_json(launch/'f1_reuse_verified.json',{'selected_checkpoint_reloaded_and_source_verified':True,
              'full_source_score_bit_exact':True,'training_rerun':False,'target_access':False})
    del model,state,scores;gc.collect();torch.cuda.empty_cache()
    prototypes,evidence=fixed_prototypes();provenance['prototypes']=evidence
    run_folds(bank,prototypes,('f2','f3','f4'),provenance,'cuda',logger,FAMILY,launch)
    records=[verify_completion(FAMILY/f/'1234',f) for f in FOLDS]
    save_json(FAMILY/'source_summary.json',{'model_name':NAME,'seed':1234,'folds':records,'target_access':False})
    save_json(launch/'complete.json',{'folds':list(FOLDS),'preserved_f1':True,'target_access':False,
              'completion_sha256':{f:checkpoint_digest(FAMILY/f/'1234/training_complete.json') for f in FOLDS}})


if __name__=='__main__':main()
