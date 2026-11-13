"""Source-gated native F1 evaluation of the selected adapted TF2 checkpoint."""

import gc
import itertools
import json
import logging
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.lacf.training import device_batch, report_progress
from audio_deepfake_detection.metrics import calibrate_source_scores, evaluate_transfer, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, canonical_score_labels, checkpoint_digest, read_dataset, save_score_manifest
from audio_deepfake_detection.token_fusion.adaptation import AdaptationConfig
from audio_deepfake_detection.token_fusion.adaptation_data import SelectedCropDataset, make_audio_loader, source_split, verify_source_membership
from scripts.token_fusion import train_adaptation as trainer
from scripts.token_fusion.train import ROOT, check_guards, replace_json, save_json

TRAINING = ROOT / 'outputs/experiments/token_fusion/adaptation_restart/f1/seed_1234'
PRODUCER_LAUNCH = ROOT / 'outputs/experiments/token_fusion/adaptation_restart/launch'
FAMILY = ROOT / 'outputs/experiments/token_fusion/evaluations/tf2_adapted'
OUTPUT = FAMILY / 'f1/1234'
LAUNCH = FAMILY / 'launch'
DATA = Path('/mnt/drive/audio-deepfake-cache')
SOURCES = FOLDS['f1']['sources']
PARAMETERS = 20_049_185


def read(path):
    return json.loads(path.read_text())


def wait_for_training(logger, *, timeout=48*3600):
    start = time.monotonic()
    while not (PRODUCER_LAUNCH/'exit_status.json').is_file():
        pid = read(LAUNCH/'authorization.json')['training_pid']
        if not Path(f'/proc/{pid}').exists() and not (TRAINING/'complete.json').is_file():
            raise RuntimeError('Adapted training exited before source completion; target held')
        if time.monotonic()-start > timeout:
            raise TimeoutError('Training verification did not finish within48h')
        replace_json(LAUNCH/'progress.json',{'phase':'waiting for adapted TF2 final source reload/export verification'})
        logger.info('WAIT: adapted TF2 final source verification is running; no target access')
        time.sleep(30)
    status = read(PRODUCER_LAUNCH/'exit_status.json')
    if not status['full_f1_source_completion_verified'] or status['status'] != 'complete':
        raise RuntimeError('Adapted source completion failed; no evaluation/retry')


def verify_training(run, data_root):
    config,done = read(run/'config.json'),read(run/'complete.json')
    if (config['mode']!='train' or config['condition']!='wavlm_lora_clap' or not config['full_f1']
            or config['held_source'] is not None or config['fold']!='f1' or config['seed']!=1234
            or config['training_sources']!=list(SOURCES) or config['development_sources']!=list(SOURCES)
            or config['model']['trainable_parameters']!=PARAMETERS or not config['model']['clap_final_stage']
            or config['model']['objective']!='TF2 detection BCE only' or config['target_access'] is not False
            or done['fold']!='f1' or done['seed']!=1234 or not done['full_f1'] or done['held_source'] is not None
            or done['sources']!=list(SOURCES) or done['condition']!='wavlm_lora_clap'
            or not done['selected_checkpoint_reloaded_and_source_verified'] or done['target_access'] is not False
            or not 1<=done['best_epoch']<=done['final_epoch']):
        raise ValueError('Adapted full-F1 source/model completion identity differs')
    check_guards(config['provenance'])
    for name,key in (('config.json','config_sha256'),('best_adaptation.pt','checkpoint_sha256'),
                     ('metrics.json','metrics_sha256'),('source_calibration.json','calibration_sha256')):
        if checkpoint_digest(run/name)!=done[key]:
            raise ValueError('Completion-bound adapted artifact changed: '+name)
    state=torch.load(run/'best_adaptation.pt',map_location='cpu',weights_only=True)
    metrics=read(run/'metrics.json')
    if (state['run_config_sha256']!=done['config_sha256'] or state['epoch']!=done['best_epoch']
            or state['source_dev']!=metrics['source_dev']):
        raise ValueError('Sparse checkpoint/config/source selection differs')
    bank={d:source_split(data_root,d,'dev') for d in SOURCES}
    marker=read(trainer.CACHE/'complete.json')
    if checkpoint_digest(trainer.CACHE/'config.json')!=config['cache_config_sha256']:
        raise ValueError('Pinned processor/source cache configuration changed')
    membership=verify_source_membership(list(bank.values()),marker)
    if any(membership[k]!=config['source_partitions'][k] for k in membership):
        raise ValueError('Canonical source-dev membership changed')
    expected_files={f'source_dev_{d}.csv' for d in SOURCES}
    if set(done['source_scores_sha256'])!=expected_files:
        raise ValueError('Adapted source score file set differs')
    source={}
    for domain in SOURCES:
        name=f'source_dev_{domain}.csv'
        if checkpoint_digest(run/name)!=done['source_scores_sha256'][name]:
            raise ValueError('Adapted source score export changed')
        source[domain]=read_score_csv(run/name,domain,expected_labels=canonical_score_labels(domain,'dev',data_root))
    calibration=calibrate_source_scores(source)
    if calibration!=read(run/'source_calibration.json') or not math.isclose(
            calibration['source_dev']['macro_eer'],done['best_macro_source_dev_eer'],abs_tol=1e-14,rel_tol=0):
        raise ValueError('Frozen adapted source calibration differs')
    return config,done,source,bank


def reconstruct(run,config,device):
    fields=('clap_final_stage','rank','alpha','lora_lr','clap_lr','fusion_lr','weight_decay')
    adaptation=AdaptationConfig(**{k:config['model'][k] for k in fields})
    model,collator,revisions=trainer.load_model(adaptation,config['seed'],device)
    if model.configuration()!=config['model'] or revisions!=config['pretrained']:
        raise ValueError('Reconstructed adapted architecture/base revisions differ')
    state=torch.load(run/'best_adaptation.pt',map_location='cpu',weights_only=True)
    trainer.restore_adaptation(model,state['trainable_state'])
    return model.eval(),collator


def source_memory_check(model,collator,bank,config,device):
    directory=trainer.CACHE/'asv5/dev/chunk_00000'
    row=next((r for r in read(directory/'records.json') if r['frames']>=10*r['rate']),None)
    if row is None or bank['asv5'].identifiers[row['index']]!=row['id']:
        raise ValueError('Canonical maximum-duration source example unavailable')
    example=bank['asv5'][row['index']]
    batch=device_batch(collator([example]*config['physical_batch']),device)
    if batch['wavlm']['input_values'].shape[-1]!=160000:
        raise ValueError('Expected real common10s source stress')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        scores=model(batch)['logit'].float()
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite maximum-duration source logits; target held')
    torch.cuda.synchronize()
    return {'passed':True,'records':len(scores),'seconds':10,'batch_size':config['physical_batch'],
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def source_check(run,config,done,source,bank,data_root,device,logger):
    model,collator=reconstruct(run,config,device)
    stress=source_memory_check(model,collator,bank,config,device)
    checks=[]
    for domain,dataset in bank.items():
        size=config['physical_batch'];count=len(dataset)
        indices=list(range(min(size,count)))
        tail=list(range(((count-1)//size)*size,count))
        indices+= [i for i in tail if i not in indices]
        selected=SelectedCropDataset([dataset.records[i] for i in indices],[dataset.identifiers[i] for i in indices],training=False)
        loader=make_audio_loader(selected,collator,batch_size=size,workers=config['development_workers'],seed=config['seed'])
        actual,labels=trainer.score_audio(model,loader,device,logger=logger)
        expected_map=dict(zip(sorted(canonical_score_labels(domain,'dev',data_root)),source[domain][1]))
        expected=np.array([expected_map[dataset.identifiers[i].split('/',2)[2]] for i in indices])
        if not np.array_equal(actual.numpy(),expected) or labels.tolist()!=[dataset.records[i][1] for i in indices]:
            raise ValueError('Fresh adapted evaluator source scores differ; target held')
        checks.append({'domain':domain,'records':len(actual),'score_bit_exact':True,'indices':indices})
    validation={'passed':True,'target_access':False,'source_memory_check':stress,'fresh_evaluator_source_subset':checks,
        'full_source_reload':{'performed_by':'trainer final fresh pinned-base reconstruction before complete.json',
          'completion_sha256':checkpoint_digest(run/'complete.json'),'selected_checkpoint_sha256':done['checkpoint_sha256'],
          'source_score_sha256':done['source_scores_sha256'],'records':{d:len(ds) for d,ds in bank.items()},
          'source_dev':read(run/'metrics.json')['source_dev']},
        'scope':'reuse bound full source reload plus fresh evaluator first/tail source-batch parity; not a second all-source audio pass'}
    logger.info('SOURCE CHECK PASSED: bound full reload and %d fresh evaluator scores bit-exact',sum(x['records'] for x in checks))
    return model,collator,validation


@torch.inference_mode()
def target_scores(model,dataset,collator,config,device,logger,execution):
    loader=make_audio_loader(dataset,collator,batch_size=config['physical_batch'],workers=config['development_workers'],seed=config['seed'])
    count=len(dataset);scores=np.empty(count,np.float32);labels=np.empty(count,np.int64)
    offset=0;started=last_log=time.monotonic()
    model.eval()
    for batch in loader:
        size=len(batch['labels']);labels[offset:offset+size]=batch['labels'].numpy()
        with torch.autocast(torch.device(device).type,dtype=torch.bfloat16):
            values=model(device_batch(batch,device))['logit'].float()
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite adapted target scores; no examples may be skipped')
        scores[offset:offset+size]=values.cpu().numpy();offset+=size
        if time.monotonic()-last_log>=30 or offset==count:
            check_guards(execution)
            report_progress(logger,'adapted TF2 F1 target',offset,count,started)
            elapsed=time.monotonic()-started
            replace_json(LAUNCH/'progress.json',{'phase':'fresh adapted TF2 SpeechFake scoring','scored':offset,'total':count,
                'examples_per_second':offset/elapsed,'eta_hours':(count-offset)/(offset/elapsed)/3600})
            last_log=time.monotonic()
    if offset!=count or labels.tolist()!=[label for _,label in dataset.records]:
        raise ValueError('Incomplete/reordered adapted target export')
    return scores,labels


def evaluate(run,output,data_root,device,logger,*,bootstrap_resamples=1000):
    if output.exists():
        raise FileExistsError('Existing adapted evaluation preserved; no overwrite/retry')
    execution=read(LAUNCH/'provenance.json');check_guards(execution)
    config,done,source,bank=verify_training(run,data_root)
    if (done['config_sha256']!=execution['expected_config_sha256']
            or done['checkpoint_sha256']!=execution['expected_checkpoint_sha256']):
        raise ValueError('Selected adapted run differs from the authorized evaluation snapshot')
    model,collator,validation=source_check(run,config,done,source,bank,data_root,device,logger)
    check_guards(execution)
    output.mkdir(parents=True,exist_ok=False)
    for name in ('config.json','best_adaptation.pt','source_calibration.json'):
        shutil.copy2(run/name,output/name)
    shutil.copy2(run/'complete.json',output/'source_training_complete.json')
    shutil.copy2(run/'metrics.json',output/'source_metrics.json')
    save_json(output/'source_validation.json',validation)
    inference={'input':'centered common physical interval up to10s; native unpadded WavLM groups; CLAP repeatpad',
        'batch_size':config['physical_batch'],'workers':config['development_workers'],'prefetch':2,
        'precision':config['precision'],'score':'scalar spoof logit','encoders':'recompute BOTH adapted audio branches; no frozen feature reuse'}
    save_json(output/'evaluation_config.json',{'model_name':'TF2 with WavLM LoRA and CLAP adaptation','fold':'f1','seed':1234,
        'inference':inference,'training_run':str(run.resolve()),'source_completion_sha256':checkpoint_digest(run/'complete.json'),
        'checkpoint_format':'sparse trainable weights plus exact pinned base models; not a standalone full checkpoint',
        'pretrained':config['pretrained'],'model':config['model'],'recovery':config.get('recovery'),
        'exploratory_after_prior_target_exposure':True,'target_tuning':False,'bootstrap_resamples':bootstrap_resamples,'bootstrap_seed':2026,
        'target_copy_status':'source SHA256 verified; destination presence/sizes checked; full destination checksums owner-waived'})
    manifest={'model_family':'tf2_adapted','fold':'f1','seed':1234,'checkpoint_sha256':done['checkpoint_sha256'],
        'checkpoint_file':'best_adaptation.pt','training_completion_sha256':checkpoint_digest(run/'complete.json'),
        'training_completion_file':'source_training_complete.json','source_validation_sha256':checkpoint_digest(output/'source_validation.json'),
        'score_sha256':{},'scorer_sha256':execution['code_sha256'],'inference':inference}
    for domain in SOURCES:
        name=f'source_dev_{domain}.csv';shutil.copy2(run/name,output/name);manifest['score_sha256'][name]=checkpoint_digest(output/name)
    # FIRST target read, after complete source gates and fresh reconstruction parity.
    records=read_dataset('speechfake','test',data_root)
    expected=canonical_score_labels('speechfake','test',data_root,records=records)
    dataset=SelectedCropDataset(records,[f'speechfake/test/{name}' for name in expected],training=False)
    scores,labels=target_scores(model,dataset,collator,config,device,logger,execution)
    path=output/'target_scores.csv'
    write_score_csv(path,zip(expected,itertools.repeat('speechfake'),labels.tolist(),scores.tolist()))
    exported=read_score_csv(path,'speechfake',expected_labels=expected)
    manifest['score_sha256'][path.name]=checkpoint_digest(path);save_score_manifest(output,manifest)
    del model,collator;gc.collect()
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    replace_json(LAUNCH/'progress.json',{'phase':'metrics and conditional bootstrap','scored':len(scores),'total':len(scores)})
    report=evaluate_transfer(source,*exported,bootstrap_resamples=bootstrap_resamples,bootstrap_seed=2026)
    report.update(model='TF2 with WavLM LoRA and CLAP adaptation',model_family='tf2_adapted',fold='f1',seed=1234,
        sources=list(SOURCES),target_domain='speechfake',target_split='test',checkpoint_epoch=done['best_epoch'],
        checkpoint_sha256=done['checkpoint_sha256'],score_sha256=manifest['score_sha256'],inference=inference,
        source_validation_sha256=manifest['source_validation_sha256'],scorer_sha256=execution['code_sha256'],
        source_data_root=str(data_root),target_data_root=str(data_root),recovery=config.get('recovery'),
        exploratory_after_prior_target_exposure=True)
    save_json(output/'metrics.json',report);check_guards(execution)
    save_json(output/'evaluation_complete.json',{'fold':'f1','seed':1234,'model_family':'tf2_adapted',
        'checkpoint_sha256':done['checkpoint_sha256'],'checkpoint_file':'best_adaptation.pt','source_completion_sha256':checkpoint_digest(run/'complete.json'),
        'source_validation_sha256':manifest['source_validation_sha256'],'score_sha256':manifest['score_sha256'],
        'metrics_sha256':checkpoint_digest(output/'metrics.json'),'no_target_tuning':True,'source_selected_checkpoint_only':True})
    replace_json(LAUNCH/'progress.json',{'phase':'complete','scored':len(scores),'total':len(scores)})
    logger.info('COMPLETE adapted TF2 F1 | target EER %.4f%% AUROC %.6f',report['target']['eer']*100,report['target']['auroc'])
    return report


def main():
    if (LAUNCH/'started.json').exists() or OUTPUT.exists():
        raise FileExistsError('Prior adapted evaluation preserved; no retry/overwrite')
    check_guards(read(LAUNCH/'provenance.json'))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');logger=logging.getLogger('tf2_adapted_eval')
    save_json(LAUNCH/'started.json',{'phase':'waiting for adapted TF2 final source completion','fold':'f1','seed':1234})
    wait_for_training(logger)
    verify_training(TRAINING,DATA)
    started=time.monotonic()
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) < 10*1024:
        if time.monotonic()-started>48*3600:raise TimeoutError('No safe GPU headroom within48h')
        logger.info('WAIT: require10GiB free for7GiB-capped adapted evaluation; other jobs unchanged');time.sleep(30)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required')
    torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2);torch.backends.nnpack.set_flags(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    evaluate(TRAINING,OUTPUT,DATA,'cuda',logger)


if __name__=='__main__':main()
