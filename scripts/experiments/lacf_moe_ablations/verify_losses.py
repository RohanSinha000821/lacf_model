"""Strict completion check for C1–C3, preserving the guarded A/B verifier.

Same artifact/canonical-score/calibration gates as the core verifier, with the
actual per-condition objective and complete loss-model configuration checked.
"""
import json
import torch
from audio_deepfake_detection.experiments.lacf_moe_ablations import loss_model as models
from scripts.experiments.lacf_moe_ablations import train as core

read,STUDY,FOLDS = core.read,core.STUDY,core.FOLDS
checkpoint_digest = core.checkpoint_digest
verify_completed_run,verify_source_macro = core.verify_completed_run,core.verify_source_macro
read_score_csv,calibrate_source_scores = core.read_score_csv,core.calibrate_source_scores
PARAMETERS = models.PARAMETERS

def verify_completion(run, fold, variant, expected_labels):
    done, digest = verify_completed_run(run, fold, 1234)
    config = read(run / 'config.json')
    if (done['study'] != STUDY or done['variant'] != variant or config['variant'] != variant
            or config['seed_plan'] != [1234] or config['sources'] != list(FOLDS[fold]['sources'])
            or config['target_access'] is not False or done['target_access'] is not False
            or config['model']['trainable_parameters'] != PARAMETERS[variant]
            or config['model']['objective'] != models.OBJECTIVES[variant]
            or done['best_checkpoint_sha256'] != digest
            or done['selected_checkpoint_reloaded_and_source_verified'] is not True):
        raise ValueError('Ablation completion identity differs')
    for filename, key in (('config.json', 'config_sha256'), ('metrics.json', 'metrics_sha256'),
                          ('routing.json', 'routing_sha256'), ('source_calibration.json', 'calibration_sha256')):
        if checkpoint_digest(run / filename) != done[key]:
            raise ValueError('Completion artifact changed: ' + filename)
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if (state['epoch'] != done['best_epoch'] or state['config_sha256'] != done['config_sha256']
            or state['source_dev'] != read(run / 'metrics.json')['source_dev']):
        raise ValueError('Selected checkpoint/config/source result differs')
    with torch.random.fork_rng(devices=[]):
        expected = models.build_model(variant, state['model_state_dict']['backend.prototypes']).configuration()
    if json.loads(json.dumps(expected)) != config['model']:
        raise ValueError('Loss ablation architecture or weights differ')
    inputs = {}
    for domain in FOLDS[fold]['sources']:
        filename = f'source_dev_{domain}.csv'
        if checkpoint_digest(run / filename) != done['source_scores_sha256'][filename]:
            raise ValueError('Source export changed')
        inputs[domain] = read_score_csv(run / filename, domain, expected_labels=expected_labels[domain])
    calibration = calibrate_source_scores(inputs)
    if calibration != read(run / 'source_calibration.json'):
        raise ValueError('Canonical source calibration differs')
    verify_source_macro(calibration['source_dev']['macro_eer'], done['best_macro_source_dev_eer'])
    manifest = read(run / 'score_manifest.json')
    if (manifest['study'] != STUDY or manifest['variant'] != variant or manifest['fold'] != fold
            or manifest['seed'] != 1234 or manifest['sources'] != list(FOLDS[fold]['sources'])
            or manifest['target_access'] is not False or manifest['checkpoint_sha256'] != digest
            or manifest['training_completion_sha256'] != checkpoint_digest(run / 'training_complete.json')
            or manifest['score_sha256'] != done['source_scores_sha256']):
        raise ValueError('Source provenance manifest differs')
    return done
