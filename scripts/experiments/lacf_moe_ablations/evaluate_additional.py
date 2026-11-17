"""Evaluate each B1–B6 checkpoint only after its source training completes."""

from contextlib import contextmanager
import logging
from pathlib import Path
import time

from audio_deepfake_detection.experiments.lacf_moe_ablations import additional_model as models
from scripts.experiments.lacf_moe_ablations import evaluate as original
from scripts.experiments.lacf_moe_ablations.train_additional import FAMILY,LAUNCH as TRAIN_LAUNCH,additional_controls

OUTPUT=FAMILY/'evaluations'
LAUNCH=OUTPUT/'launch'


@contextmanager
def additional_evaluation():
    replacements={'CORE':('A0',*models.VARIANTS),'NAMES':models.NAMES,'build_model':models.build_model,
                  'FEATURE_INPUTS':models.FEATURE_INPUTS,'OUTPUT':OUTPUT,'LAUNCH':LAUNCH}
    before={name:getattr(original,name) for name in replacements}
    try:
        with additional_controls():
            for name,value in replacements.items():setattr(original,name,value)
            yield original
    finally:
        for name,value in before.items():setattr(original,name,value)


def wait_for_run(run,key,logger):
    """No target read while waiting; a failed producer stops this queue."""
    variant,fold=key.split('/')
    while True:
        status=TRAIN_LAUNCH/'queue_status.json'
        if status.exists():
            selected=[r for r in original.read(status)['runs'] if r['variant']==variant and r['fold']==fold]
            if selected:
                if original.checkpoint_digest(run/'training_complete.json')!=selected[0]['completion_sha256']:
                    raise ValueError('Producer verified completion changed')
                return
        if (TRAIN_LAUNCH/'exit.json').exists():
            raise RuntimeError('Additional producer exited before '+key+' verified; no retry')
        if (TRAIN_LAUNCH/'launch_verified.json').exists():
            pid=original.read(TRAIN_LAUNCH/'launch_verified.json')['pid']
            if not Path(f'/proc/{pid}').exists():raise RuntimeError('Additional producer process absent')
        logger.info('WAIT: source-only training for %s; target remains unopened',key)
        time.sleep(30)


def main():
    if (LAUNCH/'started.json').exists():raise FileExistsError('Prior evaluation launch preserved')
    provenance=original.read(LAUNCH/'provenance.json');original.check_guards(provenance)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');logger=logging.getLogger('additional_evaluation')
    original.save_json(LAUNCH/'started.json',{'phase':'waiting for completed source fits','evaluations':24,'seed':1234})
    completed=[];retained={};gpu_ready=False
    with additional_evaluation() as evaluator:
        for variant in models.VARIANTS:
            for fold in evaluator.FOLDS:
                key=f'{variant}/{fold}';folder=f'{variant}_{models.NAMES[variant]}';run=FAMILY/folder/fold/'1234'
                original.replace_json(LAUNCH/'progress.json',{'variant':variant,'fold':fold,'completed':len(completed),'phase':'waiting for source completion'})
                wait_for_run(run,key,logger)
                evaluator.check_guards(provenance)
                if not gpu_ready:evaluator.runtime();gpu_ready=True
                # Bind the newly completed source selection now; model design/recipe
                # were fixed at launch, before any B-series target evaluation.
                current={**provenance,'checkpoint_sha256':{key:evaluator.checkpoint_digest(run/'best.pt')},
                    'training_completion_sha256':{key:evaluator.checkpoint_digest(run/'training_complete.json')}}
                evaluator.save_json(LAUNCH/f'selected_{variant}_{fold}.json',current)
                evaluator.evaluate(run,OUTPUT/folder/fold/'1234',fold,variant,current,'cuda',logger,retained)
                completed.append({'variant':variant,'fold':fold,'completion_sha256':evaluator.checkpoint_digest(OUTPUT/folder/fold/'1234/evaluation_complete.json')})
                evaluator.replace_json(LAUNCH/'queue_status.json',{'completed':len(completed),'runs':completed})
    original.save_json(LAUNCH/'complete.json',{'evaluations':completed,'seed':1234,'target_tuning':False})
    original.replace_json(LAUNCH/'progress.json',{'phase':'all24 additional evaluations complete','completed':24})


if __name__=='__main__':main()
