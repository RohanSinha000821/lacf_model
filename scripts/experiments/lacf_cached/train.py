"""Owner-approved cached-feature LACF F4; original backend and three losses."""

import gc
import itertools
import json
import logging
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from audio_deepfake_detection.lacf.ablations import RelationAblation
from audio_deepfake_detection.lacf.probes import FeatureBatches, load_split
from audio_deepfake_detection.lacf.training import make_optimizer, train_epoch
from audio_deepfake_detection.metrics import calibrate_source_scores, compute_auroc, compute_eer, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, save_score_manifest, verify_completed_run, verify_source_macro
from scripts.experiments.layer_fusion.train_full_f1 import checkpoint
from scripts.lacf.train_ablations import fixed_prototypes
from scripts.lacf.train_probes import save_json, seed_all


ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / 'outputs/experiments/lacf_diagnostics/frozen_features'
RUN = ROOT / 'outputs/lacf10s_optimized_20261009/f4/1234'
LAUNCH = ROOT / 'outputs/lacf10s_optimized_20261009/launch_f4_cached'
SOURCES = FOLDS['f4']['sources']
MODE = 'cached_fixed_views_native_unpadded'


def check_guards(provenance):
    for name, expected in provenance['source_sha256'].items():
        if checkpoint_digest(ROOT / name) != expected:
            raise ValueError('Preserved execution source changed: ' + name)


def load_sources(cache, data_root, logger):
    manifest = json.loads((cache / 'complete.json').read_text())
    if (manifest.get('step') != 2 or checkpoint_digest(cache / 'config.json') != manifest['config_sha256']
            or checkpoint_digest(cache / 'execution_accelerated.json') != manifest['accelerated_execution_sha256']):
        raise ValueError('Frozen feature-cache identity changed')
    bank = {}
    for domain, split in itertools.product(SOURCES, ('train', 'dev')):
        logger.info('CACHE %s/%s | verify once and load last WavLM layer + CLAP', domain, split)
        data = load_split(cache, domain, split, manifest['splits'][domain + '/' + split], data_root,
                          logger=logger, last_layer_only=True)
        if not torch.isfinite(data.wavlm).all() or not torch.isfinite(data.clap).all():
            raise ValueError('Nonfinite frozen source features')
        bank[domain, split] = data
    return bank


@torch.inference_mode()
def score_sources(model, bank, device, logger):
    model.eval()
    scores, domains = {}, {}
    for domain in SOURCES:
        data = bank[domain, 'dev']
        audio = {'wavlm': data.wavlm.to(device), 'clap': data.clap.to(device)}
        values = torch.empty(len(data.labels), device=device)
        for index in range(len(values)):
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                values[index] = model({k: v[index:index+1] for k,v in audio.items()})['logit'].float()[0]
            if (index+1) % 50000 == 0:
                logger.info('DEV %s | %d/%d cached recordings; original batch-one head', domain,index+1,len(values))
        scores[domain] = values.cpu()
        if not torch.isfinite(scores[domain]).all():
            raise ValueError('Nonfinite source development scores')
        domains[domain] = {'eer': compute_eer(data.labels.numpy(), scores[domain].numpy()),
                           'auroc': compute_auroc(data.labels.numpy(), scores[domain].numpy())}
        del audio, values
        logger.info('DEV %s | EER %.4f%% AUROC %.6f', domain, domains[domain]['eer']*100, domains[domain]['auroc'])
    return scores, {'by_domain': domains, 'eer_by_domain': {d: v['eer'] for d, v in domains.items()},
                    'macro_eer': statistics.mean(v['eer'] for v in domains.values()),
                    'macro_auroc': statistics.mean(v['auroc'] for v in domains.values())}


def new_model(prototypes, device):
    model = RelationAblation('A9', prototypes).to(device)
    config = model.configuration()
    if (config['trainable_parameters'] != 661479 or config['classifier'] != [35,64,16,1]
            or not config['semantic_loss'] or not config['bonafide_loss']
            or config['semantic_weight'] != .25 or config['bonafide_weight'] != .10):
        raise ValueError('Original primary LACF architecture and all three losses required')
    return model


def verify_completion(run):
    done, digest = verify_completed_run(run, 'f4', 1234)
    config = json.loads((run / 'config.json').read_text())
    if (config['execution_mode'] != MODE or config['sources'] != list(SOURCES)
            or config['trainable_parameters'] != 661479 or done['best_checkpoint_sha256'] != digest
            or not done['selected_checkpoint_reloaded_and_source_verified']):
        raise ValueError('Cached F4 completion identity differs')
    for filename, key in (('config.json','config_sha256'),('metrics.json','metrics_sha256'),
                          ('source_calibration.json','calibration_sha256')):
        if checkpoint_digest(run / filename) != done[key]:
            raise ValueError('Completion-bound artifact changed: ' + filename)
    source = {}
    for domain in SOURCES:
        filename = f'source_dev_{domain}.csv'
        if checkpoint_digest(run / filename) != done['source_scores_sha256'][filename]:
            raise ValueError('Source export changed')
        source[domain] = read_score_csv(run / filename, domain)
    calibration = calibrate_source_scores(source)
    if calibration != json.loads((run / 'source_calibration.json').read_text()):
        raise ValueError('Source calibration changed')
    verify_source_macro(calibration['source_dev']['macro_eer'], done['best_macro_source_dev_eer'])
    return done


def fit(bank, prototypes, output, provenance, device, logger, *, epochs=30, patience=5):
    if set(bank) != {(d,s) for d in SOURCES for s in ('train','dev')}:
        raise ValueError('Only the six canonical F4 source partitions may enter training')
    check_guards(provenance)
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / 'training.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler)
    try:
        seed_all(1234)
        model = new_model(prototypes, device)
        optimizer = make_optimizer(model)
        train = FeatureBatches([bank[d,'train'] for d in SOURCES], 'concat', device, training=True,
                               seed=1234, last_layer_only=True)
        config = {'model_name':'LACF-Frozen', 'run_name':'lacf', 'fold':'f4', 'seed':1234,
            'sources':list(SOURCES), 'target':'asv2019', 'target_split':'eval', 'target_access':False,
            'execution_mode':MODE, 'checkpoint_format':'frozen_encoder_backend_only',
            'components':model.configuration(), 'trainable_parameters':661479, 'final_seed_plan':[1234],
            'segment_seconds':10, 'batch_size':132, 'gradient_accumulation_steps':1,'effective_batch_size':132,
            'precision':'bfloat16_mixed','sensitive_dtype':'float32','optimizer':'AdamW',
            'learning_rate':3e-4,'weight_decay':1e-4,'scheduler':'constant','max_epochs':epochs,'patience':patience,
            'sampling':'equal domain/class mass; replacement; recording count per epoch',
            'training_views':'uniform choice among centered plus two fixed random common up-to10s views',
            'wavlm_training_padding':'none; native frozen diagnostic encoder features',
            'development_views':'centered native unpadded cached encoder features; original head batch1',
            'eval_batch_size':1,
            'comparison_disclosure':'F1-F3 original primary runs use fresh random padded training inputs; F4 uses fixed native unpadded cached training views. Batch-one source head scoring is retained.',
            'selection':'strict equal-weight macro source-dev EER; ties retain earlier',
            'score':'scalar_spoof_logit','label_order':['bona_fide','spoof'],'feature_workers':0,
            'num_workers':0,'eval_workers':0,
            'foundation_encoders_loaded':False,'resume':False,'train_records':train.count,
            'dev_records':{d:len(bank[d,'dev'].labels) for d in SOURCES},
            'dataset_counts':{d:{s:len(bank[d,s].labels) for s in ('train','dev')} for d in SOURCES},
            'provenance':provenance}
        save_json(output / 'config.json', config)
        digest = checkpoint_digest(output / 'config.json')
        best, best_epoch, stale, selected_scores = float('inf'), 0, 0, None
        logger.info('START LACF F4 | cached encoders | original 661479-parameter backend and all three losses | %d train', train.count)
        with (output / 'history.jsonl').open('x') as history:
            for epoch in range(1, epochs+1):
                check_guards(provenance); tick=time.monotonic()
                losses, updates = train_epoch(model,train,optimizer,device,accumulation_steps=1,logger=logger,phase=f'epoch{epoch} train')
                scores, metrics = score_sources(model,bank,device,logger)
                if metrics['macro_eer'] < best:
                    best,best_epoch,stale,selected_scores=metrics['macro_eer'],epoch,0,scores
                    checkpoint(output/'best.pt',{'epoch':epoch,'config':config,'config_sha256':digest,
                        'source_dev':metrics,'model_state_dict':model.state_dict()})
                else: stale+=1
                checkpoint(output/'last.pt',{'epoch':epoch,'config_sha256':digest,'source_dev':metrics,
                    'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                    'best_epoch':best_epoch,'patience':stale})
                row={'epoch':epoch,'losses':losses,'updates':updates,'source_dev':metrics,
                     'best_epoch':best_epoch,'patience':stale,'seconds':time.monotonic()-tick}
                history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
                logger.info('EPOCH %d DONE | source EER %.4f%% AUROC %.6f | best%d patience%d/%d | %.1fs',
                            epoch,metrics['macro_eer']*100,metrics['macro_auroc'],best_epoch,stale,patience,row['seconds'])
                if stale>=patience:break
        state=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        if state['config_sha256']!=digest or checkpoint_digest(output/'config.json')!=digest:
            raise ValueError('Selected checkpoint/config changed')
        del model,optimizer;gc.collect()
        reloaded=new_model(prototypes,device)
        reloaded.load_state_dict(state['model_state_dict'],strict=True)
        scores,metrics=score_sources(reloaded,bank,device,logger)
        if metrics!=state['source_dev'] or any(not torch.equal(scores[d],selected_scores[d]) for d in SOURCES):
            raise ValueError('Fresh selected-checkpoint full source reload differs')
        source,hashes={},{}
        for domain in SOURCES:
            data=bank[domain,'dev'];filename=f'source_dev_{domain}.csv';path=output/filename
            write_score_csv(path,zip(data.ids,itertools.repeat(domain),data.labels.tolist(),scores[domain].tolist()))
            source[domain]=read_score_csv(path,domain,expected_labels=dict(zip(data.ids,data.labels.tolist())))
            order=sorted(range(len(data.ids)),key=data.ids.__getitem__)
            if not np.array_equal(source[domain][1],scores[domain].numpy()[order]):
                raise ValueError('Canonical source export changed scores')
            hashes[filename]=checkpoint_digest(path)
        calibration=calibrate_source_scores(source)
        verify_source_macro(calibration['source_dev']['macro_eer'],best)
        save_json(output/'source_calibration.json',calibration)
        save_json(output/'metrics.json',{'model_name':'LACF-Frozen','fold':'f4','seed':1234,
            'execution_mode':MODE,'source_dev':metrics,'best_epoch':best_epoch,'final_epoch':epoch,'target_access':False})
        done={'fold':'f4','seed':1234,'best_epoch':best_epoch,'final_epoch':epoch,
            'execution_mode':MODE,'best_macro_source_dev_eer':best,'reloaded_dev_eers':metrics['eer_by_domain'],
            'selected_checkpoint_reloaded_and_source_verified':True,'best_checkpoint_sha256':checkpoint_digest(output/'best.pt'),
            'config_sha256':digest,'metrics_sha256':checkpoint_digest(output/'metrics.json'),
            'source_scores_sha256':hashes,'calibration_sha256':checkpoint_digest(output/'source_calibration.json'),'target_access':False}
        save_json(output/'training_complete.json',done)
        save_score_manifest(output,{'fold':'f4','seed':1234,'sources':list(SOURCES),'target_access':False,
            'checkpoint_sha256':done['best_checkpoint_sha256'],'training_completion_sha256':checkpoint_digest(output/'training_complete.json'),
            'score_sha256':hashes,'execution_mode':MODE})
        check_guards(provenance);verify_completion(output)
        logger.info('COMPLETE | selected epoch%d source EER%.4f%%; full reload/export/calibration passed; target unopened',best_epoch,best*100)
        return done
    finally:
        logger.removeHandler(handler);handler.close()


def preflight(bank,prototypes,device,logger):
    seed_all(1234);model=new_model(prototypes,device);optimizer=make_optimizer(model)
    loader=FeatureBatches([bank[d,'train'] for d in SOURCES],'concat',device,training=True,seed=1234,
                          last_layer_only=True,max_batches=32)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
    losses,updates=train_epoch(model,loader,optimizer,device,accumulation_steps=1)
    torch.cuda.synchronize();seconds=time.monotonic()-tick
    scores,metrics=score_sources(model,bank,device,logger)
    checkpoint(LAUNCH/'preflight.pt',{'model_state_dict':model.state_dict()})
    reloaded=new_model(prototypes,device)
    reloaded.load_state_dict(torch.load(LAUNCH/'preflight.pt',map_location='cpu',weights_only=True)['model_state_dict'])
    other,other_metrics=score_sources(reloaded,bank,device,logger)
    if other_metrics!=metrics or any(not torch.equal(scores[d],other[d]) for d in SOURCES):
        raise ValueError('Preflight full source reload differs')
    record={'passed':True,'target_access':False,'losses':losses,'updates':updates,'training_examples':loader.count,
            'training_seconds':seconds,'audio_per_second':loader.count/seconds,'full_source_dev_records':sum(len(bank[d,'dev'].labels) for d in SOURCES),
            'fresh_reload_bit_exact':True,'probe_weights_discarded':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    del model,optimizer,reloaded;gc.collect();torch.cuda.empty_cache()
    return record


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logger=logging.getLogger('lacf.f4.cached')
    provenance=json.loads((LAUNCH/'provenance.json').read_text());check_guards(provenance)
    if RUN.exists():raise FileExistsError('Existing results preserved; no overwrite/resume')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2);torch.backends.nnpack.set_flags(False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    bank=load_sources(CACHE,Path('/mnt/drive/audio-deepfake-cache'),logger)
    prototypes,evidence=fixed_prototypes();provenance['fixed_prototypes']=evidence
    record=preflight(bank,prototypes,'cuda',logger)
    save_json(LAUNCH/'source_validation.json',record)
    provenance['source_validation_sha256']=checkpoint_digest(LAUNCH/'source_validation.json')
    fit(bank,prototypes,RUN,provenance,'cuda',logger)


if __name__=='__main__':main()
