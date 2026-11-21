"""Full-fold BF16 MoE replication: retain eight seed1234 cases, add sixteen fits.

Seed reaches initialization, dropout, sampler and logical crop choice. Physical
cached crops stay identical across seeds. No original guarded source is edited.
"""
import gc
import io
import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import torch

from audio_deepfake_detection.lacf_moe.model import LACFMoE
from audio_deepfake_detection.experiments.layer_fusion.four_experts import FourExpertDetector
from audio_deepfake_detection.lacf.probes import FeatureBatches
from audio_deepfake_detection.lacf.training import make_optimizer,train_epoch
from audio_deepfake_detection.metrics import read_score_csv,write_score_csv,calibrate_source_scores,compute_eer,compute_auroc
from audio_deepfake_detection.protocol import FOLDS,checkpoint_digest,canonical_score_labels,verify_completed_run,save_score_manifest,load_score_manifest
from scripts.lacf.train_probes import seed_all
from scripts.lacf.train_ablations import fixed_prototypes
from scripts.lacf_moe import train as original3
from scripts.experiments.layer_fusion import train_four_experts as original4
from scripts.experiments.lacf_moe_embedding_paths import train as cache_loader
from scripts.experiments.lacf_moe_ablations import evaluate as target_helpers
from scripts.experiments.layer_fusion.evaluate_four_experts_cached import target_chunks as fold_chunks
from scripts.experiments.layer_fusion.evaluate_lacf_layer_moe import target_chunks as f1_chunks
from scripts.evaluate_scores import evaluate_run
from scripts.summarize_results import flat_metrics
from scripts.token_fusion.train import save_json,replace_json

ROOT,CACHE,DATA=original3.ROOT,original3.CACHE,original3.DATA
STUDY=ROOT/'outputs/experiments/lacf_moe_multiseed'
LAUNCH=STUDY/'launch'
SEEDS=(1234,1235,1236)
NEW_SEEDS=SEEDS[1:]
MODELS=('three_experts','four_experts')
FACTORIES={'three_experts':LACFMoE,'four_experts':FourExpertDetector}
FAMILIES={'three_experts':ROOT/'outputs/lacf_moe',
          'four_experts':ROOT/'outputs/experiments/layer_fusion/lacf_layer_moe4'}
EVALUATIONS={'three_experts':ROOT/'outputs/lacf_moe/evaluations',
             'four_experts':ROOT/'outputs/experiments/layer_fusion/evaluations/lacf_layer_moe4'}
PRECISION='BF16 mixed; sensitive FP32'
read=original3.read
guards=original3.guards


def paths(model,fold,seed):
    if model not in MODELS or fold not in FOLDS or seed not in SEEDS:
        raise ValueError('Only authorized two models/four folds/three seeds')
    return FAMILIES[model]/fold/str(seed),EVALUATIONS[model]/fold/str(seed)


def progress(**status):
    replace_json(LAUNCH/'progress.json',{'at_unix':time.time(),**status})


def actual_bf16(module,args,output):
    if output.dtype!=torch.bfloat16:
        raise TypeError('Final classifier must compute BF16 logits before float32 export')


def build_model(model,prototypes,device):
    result=FACTORIES[model](prototypes).to(device)
    if any(p.dtype!=torch.float32 for p in result.parameters()):raise TypeError('Historical FP32 parameter storage required')
    result.backend.classifier[-1].register_forward_hook(actual_bf16)
    return result


def loader(bank,fold,seed,device,*,training,domain=None,max_batches=None):
    datasets=[bank[d,'train']for d in FOLDS[fold]['sources']]if training else[bank[domain,'dev']]
    return FeatureBatches(datasets,'concat',device,training=training,seed=seed,max_batches=max_batches)


@torch.inference_mode()
def score_sources(model,bank,fold,seed,device,logger=None):
    model.eval();scores={};metrics={}
    for domain in FOLDS[fold]['sources']:
        data=bank[domain,'dev'];batches=loader(bank,fold,seed,device,training=False,domain=domain)
        values=[]
        for batch in batches:
            with torch.autocast(torch.device(device).type,dtype=torch.bfloat16):logits=model(batch)['logit']
            if logits.dtype!=torch.bfloat16 or not torch.isfinite(logits).all():raise ValueError('Invalid BF16 source logits')
            values.append(logits.float().cpu())
        scores[domain]=torch.cat(values)
        if len(scores[domain])!=len(data.labels):raise ValueError('Incomplete source scoring')
        metrics[domain]={'eer':compute_eer(data.labels.numpy(),scores[domain].numpy()),'auroc':compute_auroc(data.labels.numpy(),scores[domain].numpy())}
        if logger:logger.info('%s DEV %s EER%.4f%% AUROC%.6f',fold.upper(),domain,100*metrics[domain]['eer'],metrics[domain]['auroc'])
    return scores,{'by_domain':metrics,'macro_eer':statistics.mean(v['eer']for v in metrics.values()),
        'macro_auroc':statistics.mean(v['auroc']for v in metrics.values())}


def labels_for(bank,fold):
    return {d:dict(zip(bank[d,'dev'].ids,bank[d,'dev'].labels.tolist()))for d in FOLDS[fold]['sources']}


def verify_training(run,model,fold,seed,expected_labels):
    done,digest=verify_completed_run(run,fold,seed);cfg=read(run/'config.json');guards(cfg['provenance'])
    if (cfg['seed']!=seed or cfg['seed_plan']!=list(SEEDS) or cfg['sources']!=list(FOLDS[fold]['sources'])
            or cfg['precision']!=PRECISION or cfg['target_access'] is not False or done['target_access'] is not False
            or done['model_key']!=model or done['best_checkpoint_sha256']!=digest
            or not done['selected_checkpoint_reloaded_and_source_verified']
            or cfg['provenance']['cache_completion_sha256']!=checkpoint_digest(CACHE/'complete.json')):
        raise ValueError('Multi-seed training identity differs')
    for filename,key in [('config.json','config_sha256'),('metrics.json','metrics_sha256'),
            ('source_calibration.json','calibration_sha256'),('routing.json','routing_sha256')]:
        if checkpoint_digest(run/filename)!=done[key]:raise ValueError('Training artifact changed: '+filename)
    state=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    with torch.random.fork_rng(devices=[]):
        expected=FACTORIES[model](state['model_state_dict']['backend.prototypes'])
    if (json.loads(json.dumps(expected.configuration()))!=cfg['model'] or state['epoch']!=done['best_epoch']
            or state['config_sha256']!=done['config_sha256'] or state['source_dev']!=read(run/'metrics.json')['source_dev']):
        raise ValueError('Selected model/config/source result differs')
    expected.load_state_dict(state['model_state_dict'],strict=True)
    inputs={}
    for domain in FOLDS[fold]['sources']:
        name=f'source_dev_{domain}.csv'
        if checkpoint_digest(run/name)!=done['source_scores_sha256'][name]:raise ValueError('Source export changed')
        inputs[domain]=read_score_csv(run/name,domain,expected_labels=expected_labels[domain])
    calibration=calibrate_source_scores(inputs)
    if calibration!=read(run/'source_calibration.json') or abs(calibration['source_dev']['macro_eer']-done['best_macro_source_dev_eer'])>1e-12:
        raise ValueError('Frozen canonical source calibration differs')
    manifest=read(run/'score_manifest.json')
    if (manifest['fold']!=fold or manifest['seed']!=seed or manifest['checkpoint_sha256']!=digest
            or manifest['training_completion_sha256']!=checkpoint_digest(run/'training_complete.json')
            or manifest['score_sha256']!=done['source_scores_sha256']):raise ValueError('Source manifest differs')
    return cfg,done,state


def compare_source_scores(scores,run,bank,fold):
    checks=[]
    for domain in FOLDS[fold]['sources']:
        data=bank[domain,'dev'];_,saved=read_score_csv(run/f'source_dev_{domain}.csv',domain,
            expected_labels=dict(zip(data.ids,data.labels.tolist())))
        order=sorted(range(len(data.ids)),key=data.ids.__getitem__)
        if not np.array_equal(saved,scores[domain].numpy()[order]):raise ValueError('Full source score parity failed; target held')
        checks.append({'domain':domain,'records':len(saved),'score_bit_exact':True})
    return checks


def preflight(bank,prototypes,model,fold,seed,provenance,device,logger):
    original3.assert_bank(bank,fold);guards(provenance);seed_all(seed)
    net=build_model(model,prototypes,device);optimizer=make_optimizer(net)
    for batches in(8,16):
        stream=loader(bank,fold,seed,device,training=True,max_batches=batches)
        if torch.device(device).type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        tick=time.monotonic();losses,updates=train_epoch(net,stream,optimizer,device,accumulation_steps=1)
        if torch.device(device).type=='cuda':torch.cuda.synchronize()
        elapsed=time.monotonic()-tick
    expected,metrics=score_sources(net,bank,fold,seed,device)
    serial=io.BytesIO();torch.save(net.state_dict(),serial);serial.seek(0)
    fresh=build_model(model,prototypes,device);fresh.load_state_dict(torch.load(serial,map_location='cpu',weights_only=True),strict=True)
    actual,other=score_sources(fresh,bank,fold,seed,device)
    if metrics!=other or any(not torch.equal(expected[d],actual[d])for d in expected):raise ValueError('Source serialized reload failed')
    result={'passed':True,'model_key':model,'fold':fold,'seed':seed,'device':torch.device(device).type,
        'actual_logit_dtype':'torch.bfloat16','serialized_full_source_reload_bit_exact':True,
        'probe_weights_used_for_final_training':False,'target_access':False,'losses':losses,'updates':updates,
        'examples_per_second':stream.count/elapsed,'code_sha256':provenance['code_sha256'],
        'cache_completion_sha256':provenance['cache_completion_sha256'],
        'peak_reserved_bytes':torch.cuda.max_memory_reserved()if torch.device(device).type=='cuda'else None}
    logger.info('PREFLIGHT %s %s seed%d passed %.1fexamples/s',model,fold,seed,result['examples_per_second'])
    del net,fresh,optimizer,serial;gc.collect()
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    return result


def fit(bank,prototypes,run,model,fold,seed,provenance,device,logger,*,max_epochs=30,patience=5):
    if seed not in NEW_SEEDS:raise ValueError('Seed1234 must be reused, never retrained')
    original3.assert_bank(bank,fold);guards(provenance)
    proof=Path(provenance['source_validation_record'])
    if checkpoint_digest(proof)!=provenance['source_validation_sha256']:raise ValueError('Source proof changed')
    evidence=read(proof)
    if (not evidence['passed'] or evidence['model_key']!=model or evidence['fold']!=fold or evidence['seed']!=seed
            or evidence['device']!=torch.device(device).type or not evidence['serialized_full_source_reload_bit_exact']
            or evidence['actual_logit_dtype']!='torch.bfloat16' or evidence['probe_weights_used_for_final_training'] is not False
            or evidence['code_sha256']!=provenance['code_sha256'] or evidence['cache_completion_sha256']!=provenance['cache_completion_sha256']):
        raise ValueError('Source proof does not match final fit')
    run.mkdir(parents=True,exist_ok=False)
    handler=logging.FileHandler(run/'training.log');logger.addHandler(handler)
    try:
        seed_all(seed);net=build_model(model,prototypes,device);optimizer=make_optimizer(net)
        stream=loader(bank,fold,seed,device,training=True)
        cfg={'model_name':'LACF-MoE (3 experts)'if model=='three_experts'else'LACF-MoE (4 experts)',
            'model_key':model,'fold':fold,'seed':seed,'seed_plan':list(SEEDS),'sources':list(FOLDS[fold]['sources']),
            'excluded_target':FOLDS[fold]['target'],'target_access':False,'model':net.configuration(),
            'batch_size':132,'accumulation':1,'optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'scheduler':'constant',
            'max_epochs':max_epochs,'patience':patience,'precision':PRECISION,'actual_logit_dtype':'torch.bfloat16',
            'sampling':'equal domain/class mass; replacement; recording count per epoch',
            'training_views':'same cached center+two fixed physical crops; logical-view sampler seed follows training seed',
            'physical_cache_crop_seed':1234,'selection':'strict equal-domain macro source-dev EER; ties earlier',
            'scope':'full-source fold, additional-seed replication after prior target exposure','feature_workers':0,'resume':False,
            'train_records':stream.count,'dev_records':{d:len(bank[d,'dev'].labels)for d in FOLDS[fold]['sources']},'provenance':provenance}
        save_json(run/'config.json',cfg);digest=checkpoint_digest(run/'config.json')
        best,best_epoch,stale,selected=float('inf'),0,0,None
        with(run/'history.jsonl').open('x')as history:
            for epoch in range(1,max_epochs+1):
                guards(provenance)
                if device=='cuda':progress(phase='training',model=model,fold=fold,seed=seed,epoch=epoch,best_epoch=best_epoch,target_access=False)
                tick=time.monotonic();losses,updates=train_epoch(net,stream,optimizer,device,accumulation_steps=1,logger=logger,phase=f'{model} {fold} seed{seed} epoch{epoch}')
                scores,metrics=score_sources(net,bank,fold,seed,device,logger)
                if metrics['macro_eer']<best:best,best_epoch,stale,selected=metrics['macro_eer'],epoch,0,scores
                else:stale+=1
                state={'model_state_dict':net.state_dict(),'epoch':epoch,'source_dev':metrics,'config_sha256':digest}
                if stale==0:original3.checkpoint(run/'best.pt',state)
                original3.checkpoint(run/'last.pt',{**state,'optimizer_state_dict':optimizer.state_dict(),'best_epoch':best_epoch,'patience':stale})
                row={'epoch':epoch,'losses':losses,'updates':updates,'source_dev':metrics,'best_epoch':best_epoch,'patience':stale,'seconds':time.monotonic()-tick}
                history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
                logger.info('EPOCH %s %s seed%d #%d EER%.4f%% AUROC%.6f best%d patience%d/%d %.1fs',model,fold,seed,epoch,100*metrics['macro_eer'],metrics['macro_auroc'],best_epoch,stale,patience,row['seconds'])
                if stale>=patience:break
        state=torch.load(run/'best.pt',map_location='cpu',weights_only=True);del net,optimizer
        net=build_model(model,prototypes,device);net.load_state_dict(state['model_state_dict'],strict=True)
        scores,metrics=score_sources(net,bank,fold,seed,device,logger)
        if metrics!=state['source_dev']or any(not torch.equal(scores[d],selected[d])for d in scores):raise ValueError('Selected full-source reload failed')
        inputs={};hashes={};expected=labels_for(bank,fold)
        for domain in FOLDS[fold]['sources']:
            data=bank[domain,'dev'];name=f'source_dev_{domain}.csv'
            write_score_csv(run/name,zip(data.ids,itertools.repeat(domain),data.labels.tolist(),scores[domain].tolist()))
            inputs[domain]=read_score_csv(run/name,domain,expected_labels=expected[domain]);hashes[name]=checkpoint_digest(run/name)
        compare_source_scores(scores,run,bank,fold);calibration=calibrate_source_scores(inputs)
        if abs(calibration['source_dev']['macro_eer']-best)>1e-12:raise ValueError('Source calibration EER differs')
        save_json(run/'source_calibration.json',calibration)
        routes=original3.routing_summary(net,bank,fold,device)if model=='three_experts'else original4.routing_summary(net,bank,device,fold=fold)
        save_json(run/'routing.json',routes)
        save_json(run/'metrics.json',{'model_key':model,'fold':fold,'seed':seed,'source_dev':metrics,'best_epoch':best_epoch,'final_epoch':epoch,'target_access':False})
        done={'model_key':model,'fold':fold,'seed':seed,'sources':list(FOLDS[fold]['sources']),'best_epoch':best_epoch,'final_epoch':epoch,
            'best_macro_source_dev_eer':best,'selected_checkpoint_reloaded_and_source_verified':True,
            'best_checkpoint_sha256':checkpoint_digest(run/'best.pt'),'config_sha256':digest,
            'source_scores_sha256':hashes,'metrics_sha256':checkpoint_digest(run/'metrics.json'),
            'routing_sha256':checkpoint_digest(run/'routing.json'),'calibration_sha256':checkpoint_digest(run/'source_calibration.json'),'target_access':False}
        save_json(run/'training_complete.json',done)
        save_score_manifest(run,{'fold':fold,'seed':seed,'sources':list(FOLDS[fold]['sources']),'target_access':False,
            'checkpoint_sha256':done['best_checkpoint_sha256'],'training_completion_sha256':checkpoint_digest(run/'training_complete.json'),'score_sha256':hashes})
        verify_training(run,model,fold,seed,expected)
        logger.info('COMPLETE FIT %s %s seed%d selected%d EER%.4f%%',model,fold,seed,best_epoch,100*best)
        return done
    finally:logger.removeHandler(handler);handler.close()


def evaluate(run,output,model,fold,seed,bank,provenance,device,logger,*,bootstrap_resamples=1000):
    if output.exists():raise FileExistsError('Existing evaluation preserved; no retry/overwrite')
    cfg,done,state=verify_training(run,model,fold,seed,labels_for(bank,fold))
    net=build_model(model,state['model_state_dict']['backend.prototypes'],device)
    net.load_state_dict(state['model_state_dict'],strict=True)
    scores,metrics=score_sources(net,bank,fold,seed,device,logger)
    if metrics!=state['source_dev']:raise ValueError('Fresh source metrics differ; target held')
    checks=compare_source_scores(scores,run,bank,fold)
    validation={'passed':True,'full_source_score_bit_exact':True,'checks':checks,'precision':PRECISION,
        'actual_logit_dtype':'torch.bfloat16','scorer_sha256':provenance['code_sha256'],'script_sha256':checkpoint_digest(Path(__file__)),
        'encoders_loaded_or_executed':False,'target_access_before_source_gate':False}
    guards(provenance)
    producer,features=target_helpers.verify_producer(fold,provenance)
    output.mkdir(parents=True,exist_ok=False)
    for filename in('best.pt','config.json','training_complete.json','source_calibration.json'):shutil.copy2(run/filename,output/filename)
    shutil.copy2(run/'metrics.json',output/'source_metrics.json');save_json(output/'source_validation.json',validation)
    manifest=load_score_manifest(output,fold,seed,done['best_checkpoint_sha256'],create=True)
    for domain in FOLDS[fold]['sources']:
        name=f'source_dev_{domain}.csv';shutil.copy2(run/name,output/name);manifest['score_sha256'][name]=checkpoint_digest(output/name)
    inference={'input':'same verified centered up-to10s frozen cache','head_batch':132,'precision':PRECISION,
        'actual_logit_dtype':'torch.bfloat16','encoder_execution':False,'score':'scalar spoof logit'}
    save_json(output/'evaluation_config.json',{'model_key':model,'fold':fold,'seed':seed,'training_run':str(run),
        'training_completion_sha256':checkpoint_digest(run/'training_complete.json'),'target_features':str(features),
        'inference':inference,'provenance':provenance,'bootstrap_resamples':bootstrap_resamples,'bootstrap_seed':2026,
        'no_target_tuning':True,'exploratory_after_prior_target_exposure':True})
    if device=='cuda':progress(phase='cached target evaluation and bootstrap',model=model,fold=fold,seed=seed,target_access=True)
    chunks=f1_chunks(features,CACHE,DATA)if fold=='f1'else fold_chunks(producer,CACHE,fold,DATA)
    identifiers=[];values=[]
    for ids,arrays in chunks:
        guards(provenance);identifiers.extend(ids)
        values.extend(target_helpers.score_arrays(net,arrays,device,'concat').tolist())
        replace_json(output/'progress.json',{'phase':'cached target scoring','scored':len(identifiers)})
        if len(identifiers)%10240==0:logger.info('TARGET %s %s seed%d %d',model,fold,seed,len(identifiers))
    spec=FOLDS[fold];expected=canonical_score_labels(spec['target'],spec['target_split'],DATA)
    if identifiers!=list(expected)or len(values)!=len(expected):raise ValueError('Incomplete/reordered target export')
    write_score_csv(output/'target_scores.csv',zip(identifiers,itertools.repeat(spec['target']),expected.values(),values))
    manifest['score_sha256']['target_scores.csv']=checkpoint_digest(output/'target_scores.csv')
    manifest.update(model_family='lacf_moe'if model=='three_experts'else'lacf_layer_moe4',inference=inference,
        training_run=str(run),training_metrics_copy='source_metrics.json',source_validation_sha256=checkpoint_digest(output/'source_validation.json'),
        target_features_sha256=checkpoint_digest(features/'complete.json'),scorer_sha256=provenance['code_sha256'])
    save_score_manifest(output,manifest)
    del net,arrays;gc.collect()
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    report=evaluate_run(output,fold,seed=seed,source_data_root=DATA,target_data_root=DATA,bootstrap_resamples=bootstrap_resamples)
    if read(output/'source_calibration.json')!=read(run/'source_calibration.json'):raise ValueError('Frozen calibration changed')
    guards(provenance)
    completion={'model_key':model,'fold':fold,'seed':seed,'checkpoint_sha256':done['best_checkpoint_sha256'],
        'metrics_sha256':checkpoint_digest(output/'metrics.json'),'source_validation_sha256':manifest['source_validation_sha256'],
        'score_sha256':manifest['score_sha256'],'target_features_sha256':manifest['target_features_sha256'],
        'source_selected_checkpoint_only':True,'no_target_tuning':True,'exploratory_after_prior_target_exposure':True}
    save_json(output/'evaluation_complete.json',completion)
    for filename,digest in completion['score_sha256'].items():
        if checkpoint_digest(output/filename)!=digest:raise ValueError('Final score export changed')
    logger.info('COMPLETE EVAL %s %s seed%d EER%.4f%% AUROC%.6f',model,fold,seed,100*report['target']['eer'],report['target']['auroc'])
    return report


def verify_reuse(provenance):
    """Check all old seed1234 bindings; never retrain or rescore them."""
    result=[]
    for model in MODELS:
        for fold in FOLDS:
            run,ev=paths(model,fold,1234)
            for part,root in [('training',run),('evaluation',ev)]:
                for name,digest in provenance['reuse_sha256'][model][fold][part].items():
                    if checkpoint_digest(root/name)!=digest:raise ValueError('Historical seed1234 artifact changed')
            cfg=read(run/'config.json');m=read(ev/'metrics.json');manifest=read(ev/'score_manifest.json')
            done,digest=verify_completed_run(run,fold,1234)
            with torch.random.fork_rng(devices=[]):expected=FACTORIES[model](torch.eye(8,512)).configuration()
            if (cfg['model']!=json.loads(json.dumps(expected))or cfg['seed']!=1234 or 'BF16'not in cfg['precision']
                    or cfg['sources']!=list(FOLDS[fold]['sources'])or m['seed']!=1234 or m['fold']!=fold
                    or m['checkpoint_sha256']!=digest or done['best_checkpoint_sha256']!=digest
                    or m['checkpoint_epoch']!=done['best_epoch'] or manifest['score_sha256']!=m['score_sha256']):
                raise ValueError('Seed1234 model/recipe/evaluation differs')
            for name,value in [('batch_size',132),('accumulation',1),('lr',3e-4),('weight_decay',1e-4),('max_epochs',30),('patience',5)]:
                if cfg[name]!=value:raise ValueError('Historical recipe differs: '+name)
            source={d:read_score_csv(run/f'source_dev_{d}.csv',d,
                expected_labels=canonical_score_labels(d,'dev',DATA))for d in FOLDS[fold]['sources']}
            if calibrate_source_scores(source)!=read(run/'source_calibration.json'):raise ValueError('Seed1234 calibration differs')
            for name,d in m['score_sha256'].items():
                if checkpoint_digest(ev/name)!=d:raise ValueError('Seed1234 score export changed')
            result.append({'model':model,'fold':fold,'seed':1234,'training':str(run),'evaluation':str(ev),
                'training_rerun':False,'metrics_sha256':checkpoint_digest(ev/'metrics.json')})
    return result


def summarize():
    output={'seeds':list(SEEDS),'folds':list(FOLDS),'precision':PRECISION,'training_seed_count':3,
        'std_definition':'sample standard deviation across training seeds, ddof=1; distinct from conditional target bootstrap CI',
        'models':{},'target_datasets_pooled':False,'exploratory_after_prior_target_exposure':True}
    for model in MODELS:
        reports={(f,s):read(paths(model,f,s)[1]/'metrics.json')for f in FOLDS for s in SEEDS}
        for (fold,seed),m in reports.items():
            if m['fold']!=fold or m['seed']!=seed:raise ValueError('Summary fold/seed differs')
        fold_results={}
        for fold in FOLDS:
            values=[flat_metrics(reports[fold,seed])for seed in SEEDS]
            fold_results[fold]={metric:{'mean':statistics.mean(v[metric]for v in values),
                'sample_std':statistics.stdev(v[metric]for v in values),'seed_values':{str(seed):values[i][metric]for i,seed in enumerate(SEEDS)}}for metric in values[0]}
        per_seed={str(seed):{metric:statistics.mean(flat_metrics(reports[f,seed])[metric]for f in FOLDS)
            for metric in flat_metrics(reports['f1',seed])}for seed in SEEDS}
        output['models'][model]={'by_fold':fold_results,'equal_fold_mean_by_seed':per_seed,
            'equal_fold_mean':{metric:{'mean':statistics.mean(v[metric]for v in per_seed.values()),
                'sample_std':statistics.stdev(v[metric]for v in per_seed.values())}for metric in next(iter(per_seed.values()))}}
    return output


def main():
    if(LAUNCH/'started.json').exists()or any(paths(m,f,s)[0].exists()for m in MODELS for f in FOLDS for s in NEW_SEEDS):
        raise FileExistsError('Prior multi-seed launch/runs preserved; no retry/resume/overwrite')
    provenance=read(LAUNCH/'provenance.json');guards(provenance)
    save_json(LAUNCH/'started.json',{'pid':os.getpid(),'fresh_fits':16,'reused_seed1234_fits':8,'seed_plan':list(SEEDS)})
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');logger=logging.getLogger('moe_multiseed')
    progress(phase='verifying eight preserved seed1234 references',target_access=False)
    reused=verify_reuse(provenance);save_json(LAUNCH/'reference_reuse.json',{'runs':reused,'retrained':False,'rescored':False})
    if not torch.cuda.is_available()or not torch.cuda.is_bf16_supported(including_emulation=False):raise RuntimeError('Native CUDA BF16 required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.nnpack.set_flags(False)
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    prototypes,evidence=fixed_prototypes()
    if evidence!=provenance['prototypes']:raise ValueError('Original text prototypes changed')
    completed=[];bank={};save_json(LAUNCH/'queue_status.json',{'completed_fits':0,'completed_evaluations':0,'reused_fits':8,'runs':[]})
    old=cache_loader.LAUNCH;cache_loader.LAUNCH=LAUNCH
    try:
        for fold in FOLDS:
            cache_loader.load_fold_bank(bank,fold,provenance,logger)
            for seed in NEW_SEEDS:
                for model in MODELS:
                    cache_loader.wait_resources(provenance,logger,ram_bytes=2*1024**3,phase='waiting for multi-seed resources')
                    progress(phase='source-only BF16 preflight',model=model,fold=fold,seed=seed,target_access=False)
                    proof=LAUNCH/f'preflight_{model}_{fold}_{seed}.json';save_json(proof,preflight(bank,prototypes,model,fold,seed,provenance,'cuda',logger))
                    current={**provenance,'source_validation_record':str(proof),'source_validation_sha256':checkpoint_digest(proof)}
                    run,ev=paths(model,fold,seed);fit(bank,prototypes,run,model,fold,seed,current,'cuda',logger)
                    completed.append({'model':model,'fold':fold,'seed':seed,'completion_sha256':checkpoint_digest(run/'training_complete.json')})
                    replace_json(LAUNCH/'queue_status.json',{'completed_fits':len(completed),'completed_evaluations':len(completed)-1,'reused_fits':8,'runs':completed})
                    evaluate(run,ev,model,fold,seed,bank,provenance,'cuda',logger)
                    completed[-1]['evaluation_completion_sha256']=checkpoint_digest(ev/'evaluation_complete.json')
                    replace_json(LAUNCH/'queue_status.json',{'completed_fits':len(completed),'completed_evaluations':len(completed),'reused_fits':8,'runs':completed})
                    gc.collect();torch.cuda.empty_cache()
    finally:cache_loader.LAUNCH=old
    guards(provenance);verify_reuse(provenance)
    for case in completed:
        run,ev=paths(case['model'],case['fold'],case['seed'])
        if (checkpoint_digest(run/'training_complete.json')!=case['completion_sha256']
                or checkpoint_digest(ev/'evaluation_complete.json')!=case['evaluation_completion_sha256']
                or checkpoint_digest(ev/'metrics.json')!=read(ev/'evaluation_complete.json')['metrics_sha256']):
            raise ValueError('Completed replication artifacts changed before summary')
    save_json(STUDY/'summary.json',summarize())
    save_json(LAUNCH/'complete.json',{'fresh_fits':16,'fresh_evaluations':16,'reused_fits':8,'total_model_fold_seed_results':24,'runs':completed,'precision':PRECISION})
    progress(phase='all24 model/fold/seed results complete;16fresh plus8reused',completed_fits=16,completed_evaluations=16,target_access=True)


if __name__=='__main__':
    try:main()
    except Exception:
        if LAUNCH.is_dir():progress(phase='FAILED; stopped without retry')
        logging.exception('Multi-seed queue stopped; preserve all evidence')
        raise
