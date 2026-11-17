"""Original cached LACF F4 evaluation; retain the trained batch-one scoring."""

import logging
import json

import numpy as np
import torch

from audio_deepfake_detection.metrics import read_score_csv, calibrate_source_scores
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest, canonical_score_labels, verify_completed_run, verify_source_macro
from scripts.experiments.lacf_cached import train
from scripts.experiments.lacf_moe_ablations import evaluate as cached

OUTPUT = train.RUN.parents[1] / 'evaluations/f4/1234'
LAUNCH = OUTPUT.parents[1] / 'launch'


def verify_training(run):
    done,digest=verify_completed_run(run,'f4',1234);config=cached.read(run / 'config.json')
    train.check_guards(config['provenance'])
    if (config['fold']!='f4' or config['seed']!=1234 or config['eval_batch_size']!=1
            or config['precision']!='bfloat16_mixed' or config['target_access'] is not False):
        raise ValueError('Original cached F4 recipe differs')
    if (config['execution_mode']!=train.MODE or config['sources']!=list(train.SOURCES)
            or config['trainable_parameters']!=661479 or done['best_checkpoint_sha256']!=digest
            or not done['selected_checkpoint_reloaded_and_source_verified']):
        raise ValueError('Original cached F4 completion differs')
    for filename,key in [('config.json','config_sha256'),('metrics.json','metrics_sha256'),
                         ('source_calibration.json','calibration_sha256')]:
        if checkpoint_digest(run / filename)!=done[key]:
            raise ValueError('Completion-bound artifact changed: '+filename)
    source={}
    for domain in train.SOURCES:
        filename=f'source_dev_{domain}.csv'
        if checkpoint_digest(run / filename)!=done['source_scores_sha256'][filename]:
            raise ValueError('Source export changed')
        # Match the exact canonical ID order used at training export/calibration.
        source[domain]=read_score_csv(run / filename,domain,
            expected_labels=canonical_score_labels(domain,'dev',cached.DATA))
    calibration=calibrate_source_scores(source)
    if calibration!=cached.read(run / 'source_calibration.json'):
        raise ValueError('Canonical frozen source calibration differs')
    verify_source_macro(calibration['source_dev']['macro_eer'],done['best_macro_source_dev_eer'])
    for name,key in [('complete.json','cache_completion_sha256'),('config.json','cache_config_sha256'),
                      ('execution_accelerated.json','cache_execution_sha256')]:
        if checkpoint_digest(train.CACHE / name)!=config['provenance'][key]:
            raise ValueError('Training feature cache changed')
    state=torch.load(run / 'best.pt',map_location='cpu',weights_only=True)
    if (state['config_sha256']!=done['config_sha256'] or state['epoch']!=done['best_epoch']
            or state['source_dev']!=cached.read(run / 'metrics.json')['source_dev']):
        raise ValueError('Selected checkpoint/config/metrics differ')
    model=train.new_model(state['model_state_dict']['prototypes'],'cpu')
    model.load_state_dict(state['model_state_dict'],strict=True)
    if cached.read(run / 'config.json')['components']!=json.loads(json.dumps(model.configuration())):
        raise ValueError('Original backend architecture differs')
    return config,done,model.requires_grad_(False).eval()


def evaluate(run,output,provenance,device,logger,*,bootstrap_resamples=1000):
    if output.exists():raise FileExistsError('Existing original F4 evaluation preserved')
    if (checkpoint_digest(run / 'training_complete.json')!=provenance['training_completion_sha256']['primary/f4']
            or checkpoint_digest(run / 'best.pt')!=provenance['checkpoint_sha256']['primary/f4']):
        raise ValueError('Launch-bound original F4 changed')
    config,done,model=verify_training(run);model.to(device)
    bank=cached.source_bank(train.CACHE,'f4',cached.DATA,logger,{},last_layer_only=True)
    scores,metrics=train.score_sources(model,bank,device,logger)
    if metrics!=cached.read(run / 'metrics.json')['source_dev']:
        raise ValueError('Fresh original F4 source metrics differ; target held')
    checks=[]
    for d in FOLDS['f4']['sources']:
        ds=bank[d,'dev'];_,exported=read_score_csv(run / f'source_dev_{d}.csv',d,
            expected_labels=dict(zip(ds.ids,ds.labels.tolist())))
        order=sorted(range(len(ds.ids)),key=ds.ids.__getitem__)
        if not np.array_equal(scores[d].numpy()[order],exported):
            raise ValueError('Fresh original F4 source scores differ; target held')
        checks.append({'domain':d,'records':len(exported),'score_bit_exact':True})
    del bank,scores
    validation={'passed':True,'full_source_score_bit_exact':True,'checks':checks,
                'target_access':False,'encoders_loaded_or_executed':False,'head_batch':1}
    train.check_guards(config['provenance'])
    return cached.finish_evaluation(model,run,output,'f4','Original LACF (cached F4)','lacf',
        config,done,validation,provenance,device,logger,batch_size=1,last_layer_only=True,
        bootstrap_resamples=bootstrap_resamples)


def main():
    if (LAUNCH / 'started.json').exists():raise FileExistsError('Prior launch preserved')
    provenance=cached.read(LAUNCH / 'provenance.json');cached.check_guards(provenance)
    free=cached.runtime();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    cached.save_json(LAUNCH / 'started.json',{'fold':'f4','seed':1234,'gpu_free_mib':free,'encoder_execution':False})
    evaluate(train.RUN,OUTPUT,provenance,'cuda',logging.getLogger('original_lacf_f4_evaluation'))
    cached.save_json(LAUNCH / 'complete.json',{'fold':'f4','seed':1234,
        'evaluation_completion_sha256':checkpoint_digest(OUTPUT / 'evaluation_complete.json')})


if __name__=='__main__':main()
