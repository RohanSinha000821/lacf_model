"""Eight fresh cached BF16 fits and their fixed-checkpoint native evaluations."""
from contextlib import contextmanager
import gc
import json
import logging
import os
from pathlib import Path

import torch

from audio_deepfake_detection.experiments.lacf_moe_embedding_paths import model_bf16 as models
from scripts.experiments.lacf_moe_ablations import train as core
from scripts.experiments.lacf_moe_ablations import evaluate as evaluator
from scripts.experiments.lacf_moe_embedding_paths import train as cache_loader

ROOT=core.ROOT
FAMILY=ROOT/'outputs/experiments/lacf_moe_embedding_paths/bf16'
LAUNCH=FAMILY/'launch'
OUTPUT=FAMILY/'evaluations'
STUDY='lacf_moe_embedding_paths_bf16'
_original_verify=core.verify_completion


def verify_completion(run,fold,variant,expected_labels):
    done=_original_verify(run,fold,variant,expected_labels)
    config=core.read(run/'config.json')
    if config['precision']!=models.PRECISION or done['study']!=STUDY:
        raise ValueError('BF16 study/scoring precision differs')
    state=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    prototypes=state['model_state_dict']['backend.prototypes']
    with torch.random.fork_rng(devices=[]):expected=models.build_model(variant,prototypes).configuration()
    if json.loads(json.dumps(expected))!=config['model']:
        raise ValueError('Embedding path architecture/objective/precision configuration differs')
    if any(p.dtype!=torch.float32 for p in state['model_state_dict'].values() if p.is_floating_point()):
        raise ValueError('Historical FP32 parameter storage differs')
    if (Path(config['provenance']['cache']).resolve()!=core.CACHE.resolve()
            or config['provenance']['cache_completion_sha256']!=core.checkpoint_digest(core.CACHE/'complete.json')):
        raise ValueError('Frozen feature cache identity differs')
    return done


@contextmanager
def bindings():
    """Use tested BF16 fit/export loops only in this process; source bytes unchanged."""
    replacements={'CORE':('reference',*models.VARIANTS),'NAMES':models.NAMES,
        'PARAMETERS':models.PARAMETERS,'FEATURE_INPUTS':models.FEATURE_INPUTS,
        'build_model':models.build_model,'STUDY':STUDY,'FAMILY':FAMILY,'verify_completion':verify_completion}
    old={key:getattr(core,key)for key in replacements}
    eval_changes={'CORE':('reference',*models.VARIANTS),'NAMES':models.NAMES,
        'FEATURE_INPUTS':models.FEATURE_INPUTS,'build_model':models.build_model,'OUTPUT':OUTPUT}
    previous={key:getattr(evaluator,key)for key in eval_changes}
    old_launch=cache_loader.LAUNCH
    try:
        for key,value in replacements.items():setattr(core,key,value)
        for key,value in eval_changes.items():setattr(evaluator,key,value)
        cache_loader.LAUNCH=LAUNCH
        yield core
    finally:
        for key,value in old.items():setattr(core,key,value)
        for key,value in previous.items():setattr(evaluator,key,value)
        cache_loader.LAUNCH=old_launch


def preflight(bank,prototypes,fold,variant,provenance,device,logger):
    report=core.preflight(bank,prototypes,fold,variant,provenance,device,logger)
    # The model's forward asserts actual BF16 logits on every update and source pass.
    report.update(precision=models.PRECISION,actual_logit_dtype='torch.bfloat16',
        sensitive_operations_dtype='torch.float32',weights_storage_dtype='torch.float32')
    return report


def verify_reference(bank,fold,provenance):
    run=core.REFERENCE/fold/'1234'
    for name,digest in provenance['reference_sha256'][fold].items():
        if core.checkpoint_digest(run/name)!=digest:raise ValueError('Historical BF16 reference changed')
    core.verify_reference(run,fold,expected_labels=core.labels_for(bank,fold))
    link=FAMILY/'reference'/fold/'1234';link.parent.mkdir(parents=True,exist_ok=False)
    link.symlink_to(os.path.relpath(run,link.parent),target_is_directory=True)
    core.save_json(LAUNCH/f'reference_{fold}.json',{'fold':fold,'reference':str(run),
        'training_rerun':False,'learned_parameters_used_for_initialization':False,
        'reference_sha256':provenance['reference_sha256'][fold]})


def verify_evaluation(run,output,fold,done):
    report=core.read(output/'metrics.json');finished=core.read(output/'evaluation_complete.json')
    manifest=core.read(output/'score_manifest.json');validation=core.read(output/'source_validation.json')
    if (finished['fold']!=fold or finished['seed']!=1234
            or finished['checkpoint_sha256']!=done['best_checkpoint_sha256']
            or finished['metrics_sha256']!=core.checkpoint_digest(output/'metrics.json')
            or finished['source_validation_sha256']!=core.checkpoint_digest(output/'source_validation.json')
            or not finished['source_selected_checkpoint_only'] or not finished['no_target_tuning']
            or not validation['passed'] or not validation['full_source_score_bit_exact']
            or not all(r['score_bit_exact']for r in validation['checks'])
            or {r['domain']for r in validation['checks']}!=set(core.FOLDS[fold]['sources'])
            or report['inference']['precision']!=models.PRECISION
            or report['model_family']!='lacf_moe_embedding_paths_bf16'
            or manifest['training_completion_sha256']!=core.checkpoint_digest(run/'training_complete.json')
            or report['confidence_intervals']['resamples']!=1000 or report['confidence_intervals']['seed']!=2026):
        raise ValueError('Completed BF16 evaluation identity/source proof differs')
    for filename in ('best.pt','config.json','training_complete.json','source_calibration.json'):
        if core.checkpoint_digest(output/filename)!=core.checkpoint_digest(run/filename):
            raise ValueError('Copied training/calibration artifact changed')
    if finished['score_sha256']!=manifest['score_sha256'] or report['score_sha256']!=manifest['score_sha256']:
        raise ValueError('Score export bindings differ')
    expected={f'source_dev_{d}.csv'for d in core.FOLDS[fold]['sources']}|{'target_scores.csv'}
    if set(manifest['score_sha256'])!=expected:raise ValueError('Incomplete source/target exports')
    for filename,digest in manifest['score_sha256'].items():
        if core.checkpoint_digest(output/filename)!=digest:raise ValueError('Score export changed')
        if filename!='target_scores.csv'and done['source_scores_sha256'][filename]!=digest:
            raise ValueError('Evaluation source scores differ')
    return finished


def evaluate_fit(run,output,fold,variant,provenance,bank,device,logger,*,bootstrap_resamples=1000):
    if output.exists():raise FileExistsError('Preserve previous evaluation; no retry/overwrite')
    config,done,model=evaluator.verify_training(run,fold,variant,core.DATA)
    model=model.to(device)
    validation=evaluator.source_reload(model,run,config,fold,variant,device,logger,bank)
    core.guards(config['provenance'])
    validation.update(actual_logit_dtype='torch.bfloat16',precision=models.PRECISION,
        queue_script_sha256=core.checkpoint_digest(Path(__file__)))
    if device=='cuda':
        evaluator.replace_json(LAUNCH/'progress.json',{'phase':'cached BF16 target scoring and bootstrap',
            'variant':variant,'fold':fold,'target_access':True})
    current={**provenance,'selected_checkpoint_sha256':done['best_checkpoint_sha256'],
        'selected_training_completion_sha256':core.checkpoint_digest(run/'training_complete.json')}
    report=evaluator.finish_evaluation(model,run,output,fold,
        f'LACF-MoE + {models.NAMES[variant]}','lacf_moe_embedding_paths_bf16',
        config,done,validation,current,device,logger,inputs='concat',bootstrap_resamples=bootstrap_resamples)
    verify_evaluation(run,output,fold,done)
    del model;gc.collect()
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    return report


def main():
    if (LAUNCH/'started.json').exists()or any((FAMILY/v).exists()for v in (*models.VARIANTS,'reference')):
        raise FileExistsError('Preserve existing launch/runs; no retry/resume/overwrite')
    provenance=core.read(LAUNCH/'provenance.json');core.guards(provenance)
    if not torch.cuda.is_available()or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native CUDA BF16 required; no fallback')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False;torch.backends.nnpack.set_flags(False)
    torch.cuda.set_per_process_memory_fraction(1024**3/torch.cuda.get_device_properties(0).total_memory)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');logger=logging.getLogger(STUDY)
    core.save_json(LAUNCH/'started.json',{'pid':os.getpid(),'precision':models.PRECISION,'expected_fits':8,'expected_evaluations':8})
    core.save_json(LAUNCH/'queue_status.json',{'completed_fits':0,'completed_evaluations':0,'runs':[]})
    prototypes,evidence=core.fixed_prototypes()
    if evidence!=provenance['prototypes']:raise ValueError('Original prototype bank changed')
    bank,completed={},[]
    with bindings():
        for fold in core.FOLDS:
            cache_loader.load_fold_bank(bank,fold,provenance,logger)
            verify_reference(bank,fold,provenance)
            for variant in models.VARIANTS:
                cache_loader.wait_resources(provenance,logger,ram_bytes=2*1024**3,phase='waiting for fit resources')
                core.progress(LAUNCH,phase='BF16 source preflight',variant=variant,fold=fold,completed_fits=len(completed))
                proof=LAUNCH/f'source_validation_{variant}_{fold}.json'
                core.save_json(proof,preflight(bank,prototypes,fold,variant,provenance,'cuda',logger))
                current={**provenance,'source_validation_record':str(proof),'source_validation_sha256':core.checkpoint_digest(proof)}
                run=FAMILY/variant/fold/'1234'
                core.fit(bank,prototypes,run,fold,variant,current,'cuda',logger,launch=LAUNCH)
                done=verify_completion(run,fold,variant,core.labels_for(bank,fold))
                completed.append({'variant':variant,'fold':fold,'completion_sha256':core.checkpoint_digest(run/'training_complete.json')})
                evaluator.replace_json(LAUNCH/'queue_status.json',{'completed_fits':len(completed),'completed_evaluations':len(completed)-1,'runs':completed})
                core.progress(LAUNCH,phase='fresh BF16 source parity then cached target evaluation',variant=variant,fold=fold,completed_fits=len(completed))
                output=OUTPUT/variant/fold/'1234'
                evaluate_fit(run,output,fold,variant,provenance,bank,'cuda',logger)
                completed[-1]['evaluation_completion_sha256']=core.checkpoint_digest(output/'evaluation_complete.json')
                evaluator.replace_json(LAUNCH/'queue_status.json',{'completed_fits':len(completed),'completed_evaluations':len(completed),'runs':completed})
                gc.collect();torch.cuda.empty_cache()
    if len(completed)!=8:raise ValueError('Incomplete BF16 queue')
    core.guards(provenance)
    core.save_json(LAUNCH/'complete.json',{'completed_fits':8,'completed_evaluations':8,'runs':completed,
        'precision':models.PRECISION,'seed':1234,'no_target_tuning':True,'encoder_execution':False})
    evaluator.replace_json(LAUNCH/'progress.json',{'phase':'all eight BF16 fits and evaluations verified complete',
        'completed_fits':8,'completed_evaluations':8,'target_access':True})


if __name__=='__main__':
    try:main()
    except Exception:
        if LAUNCH.is_dir():core.progress(LAUNCH,phase='FAILED; stopped without retry')
        logging.exception('BF16 embedding paths stopped; preserve all evidence')
        raise
