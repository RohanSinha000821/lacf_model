"""Fresh F1--F4 three-expert LACF-MoE in FP32, reusing verified frozen features.

Isolated precision variant. Historical BF16 code/artifacts remain unchanged.
Only source train/dev data are read; no automatic target evaluation or retry.
"""

import gc
import hashlib
import io
import itertools
import json
import logging
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from audio_deepfake_detection.lacf_moe.model import GROUPS, LACFMoE, TRAINABLE_PARAMETERS
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_bank
from audio_deepfake_detection.lacf.training import make_optimizer, device_batch, report_progress
from audio_deepfake_detection.metrics import calibrate_source_scores, compute_auroc, compute_eer, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, save_score_manifest, verify_source_macro
from scripts.lacf.train_probes import save_json, seed_all
from scripts.lacf.train_ablations import fixed_prototypes
from scripts.lacf_moe.train import assert_bank, checkpoint, guards, read, verify_completion as verify_original_completion

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / 'outputs/lacf_moe/fp32'
CACHE = ROOT / 'outputs/experiments/lacf_diagnostics/frozen_features'
DATA = Path('/mnt/drive/audio-deepfake-cache')
NAME = 'LACF-MoE (FP32)'
PRECISION = 'FP32 trainable model, source selection, calibration and scoring; TF32 disabled'


def assert_fp32(model):
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise TypeError('All trainable model parameters must remain FP32')
    if any(b.is_floating_point() and b.dtype != torch.float32 for b in model.buffers()):
        raise TypeError('All floating-point model buffers must remain FP32')


def train_epoch(model, loader, optimizer, device, *, accumulation_steps=1, logger=None, phase='train'):
    if accumulation_steps != 1 or len(loader) < 1:
        raise ValueError('Preserve batch132 / accumulation1 and nonempty training data')
    assert_fp32(model)
    model.train()
    names = ('total', 'detection', 'semantic', 'bonafide')
    sums = torch.zeros(len(names), device=device, dtype=torch.float32)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    count = 0
    started = time.monotonic()
    for index, batch in enumerate(loader):
        batch = device_batch(batch, device)
        if any(v.dtype != torch.float32 for k, v in batch.items() if k != 'labels'):
            raise TypeError('Cached model inputs must be FP32')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(torch.device(device).type, enabled=False):
            output = model(batch)
            if output['logit'].dtype != torch.float32:
                raise TypeError('Final scoring layer did not execute in FP32')
            losses = model.objective(output, batch['labels'])
        values = torch.stack([losses[name].detach() for name in names])
        if not torch.isfinite(values).all():
            raise RuntimeError('Nonfinite FP32 loss; no retry or precision fallback')
        losses['total'].backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
            raise RuntimeError('Nonfinite/missing FP32 gradients; optimizer update refused')
        optimizer.step()
        count += len(batch['labels'])
        sums += values * len(batch['labels'])
        if (index + 1) % 100 == 0 or index + 1 == len(loader):
            report_progress(logger, phase, index + 1, len(loader), started,
                            loss=float(sums[0] / count) if logger is not None else None, examples=count)
    return dict(zip(names, (sums / count).cpu().tolist())), len(loader)


@torch.inference_mode()
def score_probe(model, loader):
    assert_fp32(model)
    model.eval()
    scores = torch.empty(loader.count, dtype=torch.float32, device=loader.device)
    offset = 0
    for batch in loader:
        if any(v.dtype != torch.float32 for k, v in batch.items() if k != 'labels'):
            raise TypeError('Cached scoring inputs must be FP32')
        with torch.autocast(loader.device.type, enabled=False):
            values = model(batch)['logit']
        if values.dtype != torch.float32 or not torch.isfinite(values).all():
            raise RuntimeError('Final scores must be finite, computed in FP32')
        scores[offset:offset + len(values)] = values
        offset += len(values)
    if offset != loader.count:
        raise ValueError('Incomplete source score vector')
    scores = scores.cpu()
    labels = loader.datasets[0].labels[:loader.count]
    return scores, {'eer': compute_eer(labels.numpy(), scores.numpy()),
                    'auroc': compute_auroc(labels.numpy(), scores.numpy())}


def verify_completion(run, fold, *, expected_labels=None):
    config = read(run / 'config.json')
    if config['model_name'] != NAME or config['precision'] != PRECISION or config['provenance']['target_access'] is not False:
        raise ValueError('FP32 completion precision/identity differs')
    if config['provenance']['environment']['cuda_matmul_tf32'] or config['provenance']['environment']['cudnn_tf32']:
        raise ValueError('TF32 must remain disabled')
    return verify_original_completion(run, fold, expected_labels=expected_labels)



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
            with torch.autocast(torch.device(device).type, enabled=False):
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
                'max_epochs':max_epochs,'patience':patience,'precision':PRECISION,
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
    linear_dtypes = set()
    hooks = [m.register_forward_hook(lambda m, args, out: linear_dtypes.add(str(out.dtype)))
             for m in model.modules() if isinstance(m, torch.nn.Linear)]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    losses,updates=train_epoch(model,loader,optimizer,device,accumulation_steps=1)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    expected,_=score_sources(model,bank,fold,device,logger)
    serialized = io.BytesIO()
    torch.save(state, serialized)
    serialized.seek(0)
    reloaded = torch.load(serialized, map_location='cpu', weights_only=True)
    fresh=LACFMoE(prototypes).to(device);fresh.load_state_dict(reloaded,strict=True)
    for hook in hooks: hook.remove()
    if linear_dtypes != {'torch.float32'}:
        raise TypeError('Source GPU linear layers did not all execute in FP32')
    start=time.monotonic();actual,_=score_sources(fresh,bank,fold,device,logger);torch.cuda.synchronize()
    dev_seconds=time.monotonic()-start
    if any(not torch.equal(expected[d],actual[d]) for d in expected):raise ValueError('Source preflight reload differs')
    report={'passed':True,'fold':fold,'losses':losses,'updates':updates,'source_reload_bit_exact':True,
            'train_examples_per_second':loader.count/elapsed,'source_dev_seconds':dev_seconds,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'probe_weights_used_for_final_training':False,'target_access':False,'precision':PRECISION,
            'linear_output_dtypes':sorted(linear_dtypes),'serialized_source_reload_bit_exact':True,
            'serialized_probe_sha256':hashlib.sha256(serialized.getvalue()).hexdigest(),
            'full_source_dev_recordings':sum(len(bank[d,'dev'].labels) for d in expected)}
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
    launch = FAMILY / 'launch'
    if (launch / 'started.json').exists() or any((FAMILY / f / '1234').exists() for f in FOLDS):
        raise FileExistsError('Prior FP32 run preserved; no automatic resume, retry or overwrite')
    provenance = read(launch / 'provenance.json')
    guards(provenance)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.nnpack.set_flags(False)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        raise RuntimeError('Need CUDA and at least 2GiB free; preserve other jobs')
    torch.cuda.set_per_process_memory_fraction(1024**3 / torch.cuda.get_device_properties(0).total_memory)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('lacf_moe_fp32')
    save_json(launch / 'started.json', {'phase':'source cache verification/loading', 'precision':PRECISION,
                                      'target_access':False, 'folds':list(FOLDS)})
    bank = load_bank(CACHE, DATA, logger=logger)
    if any(not torch.isfinite(data.wavlm).all() or not torch.isfinite(data.clap).all() for data in bank.values()):
        raise ValueError('Nonfinite source features')
    prototypes, evidence = fixed_prototypes()
    if evidence != provenance['prototypes']:
        raise ValueError('Original prototype identity changed')
    # Each bounded source preflight is discarded. fit() resets seed and initializes fresh weights.
    run_folds(bank, prototypes, tuple(FOLDS), provenance, 'cuda', logger, FAMILY, launch)
    records = [verify_completion(FAMILY / f / '1234', f) for f in FOLDS]
    save_json(FAMILY / 'source_summary.json', {'model_name':NAME, 'precision':PRECISION,
              'seed':1234, 'folds':records, 'target_access':False})
    save_json(launch / 'complete.json', {'folds':list(FOLDS), 'precision':PRECISION,
              'target_access':False, 'completion_sha256':{
                  f:checkpoint_digest(FAMILY / f / '1234/training_complete.json') for f in FOLDS}})


if __name__ == '__main__':
    main()
