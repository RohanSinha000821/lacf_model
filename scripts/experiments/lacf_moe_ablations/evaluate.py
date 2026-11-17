"""Evaluate completed MoE-A1–A6 on verified cached native P1 features."""

import gc
import hashlib
import itertools
import json
import logging
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from audio_deepfake_detection.experiments.lacf_moe_ablations.model import CORE, FEATURE_INPUTS, NAMES, build_model
from audio_deepfake_detection.lacf.probes import load_split
from audio_deepfake_detection.metrics import read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, save_score_manifest
from scripts.evaluate_scores import evaluate_run
from scripts.experiments.lacf_moe_ablations import train
from scripts.experiments.layer_fusion.evaluate_four_experts_cached import target_chunks as fold_chunks
from scripts.experiments.layer_fusion.evaluate_lacf_layer_moe import TARGET_FEATURES, target_chunks as f1_chunks
from scripts.token_fusion.train import check_guards, replace_json, save_json

ROOT, DATA, CACHE = train.ROOT, train.DATA, train.CACHE
OUTPUT = train.FAMILY / 'evaluations'
LAUNCH = OUTPUT / 'launch'
PRODUCERS = ROOT / 'outputs/lacf_moe/evaluations'


def read(path):
    return json.loads(Path(path).read_text())


def runtime():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native BF16 CUDA required; no fallback')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits', '-i', '0'], text=True).strip())
    if free < 2048:
        raise RuntimeError('Need2GiB free; preserve active jobs')
    torch.cuda.set_per_process_memory_fraction(1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return free


def source_bank(cache, fold, data_root, logger, retained, *, last_layer_only=False):
    marker = read(cache / 'complete.json')
    if (marker['step'] != 2 or checkpoint_digest(cache / 'config.json') != marker['config_sha256']
            or checkpoint_digest(cache / 'execution_accelerated.json') != marker['accelerated_execution_sha256']):
        raise ValueError('Source cache completion/config/execution changed')
    for domain in FOLDS[fold]['sources']:
        if (domain, 'dev') not in retained:
            retained[domain, 'dev'] = load_split(cache, domain, 'dev', marker['splits'][f'{domain}/dev'],
                data_root, logger=logger, last_layer_only=last_layer_only)
    return {(d, 'dev'): retained[d, 'dev'] for d in FOLDS[fold]['sources']}


def verify_training(run, fold, variant, data_root):
    if variant not in CORE[1:]:
        raise ValueError('Only completed A1–A6; A0 results already exist, additional controls held')
    labels = {d: canonical_score_labels(d, 'dev', data_root) for d in FOLDS[fold]['sources']}
    done = train.verify_completion(run, fold, variant, labels)
    config = read(run / 'config.json')
    train.guards(config['provenance'])
    if (Path(config['provenance']['cache']).resolve() != CACHE.resolve()
            or config['provenance']['cache_completion_sha256'] != checkpoint_digest(CACHE / 'complete.json')):
        raise ValueError('Training source cache identity changed')
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    # A1/A2 contain no prototypes; recover only the original fixed buffer for construction.
    prototypes = state['model_state_dict'].get('prototypes')
    if prototypes is None:
        reference = torch.load(train.REFERENCE / 'f1/1234/best.pt', map_location='cpu', weights_only=True, mmap=True)
        prototypes = reference['model_state_dict']['backend.prototypes']
    if hashlib.sha256(prototypes.numpy().tobytes()).hexdigest() != config['provenance']['prototypes']['prototype_bytes_sha256']:
        raise ValueError('Original fixed prototype bytes changed')
    model = build_model(variant, prototypes)
    model.load_state_dict(state['model_state_dict'], strict=True)
    if json.loads(json.dumps(model.configuration())) != config['model']:
        raise ValueError('Selected architecture/configuration differs')
    return config, done, model.requires_grad_(False).eval()


def source_reload(model, run, config, fold, variant, device, logger, retained):
    bank = source_bank(CACHE, fold, DATA, logger, retained)
    scores, metrics, _ = train.score_sources(model, bank, fold, variant, device, logger)
    if metrics != read(run / 'metrics.json')['source_dev']:
        raise ValueError('Fresh source metrics differ; target held')
    checks = []
    for d in FOLDS[fold]['sources']:
        ds = bank[d, 'dev']
        _, exported = read_score_csv(run / f'source_dev_{d}.csv', d,
                                     expected_labels=dict(zip(ds.ids, ds.labels.tolist())))
        order = sorted(range(len(ds.ids)), key=ds.ids.__getitem__)
        if not np.array_equal(scores[d].numpy()[order], exported):
            raise ValueError('Fresh full source score reload differs; target held')
        checks.append({'domain': d, 'records': len(exported), 'score_bit_exact': True})
    return {'passed': True, 'full_source_score_bit_exact': True, 'checks': checks,
            'target_access': False, 'encoders_loaded_or_executed': False,
            'calibration': 'exact original canonical source calibration checked before scoring'}


def verify_producer(fold, provenance):
    """Called only after this detector passes its source gates; no re-extraction."""
    producer = PRODUCERS / fold / '1234'
    done, manifest = read(producer / 'evaluation_complete.json'), read(producer / 'score_manifest.json')
    if (checkpoint_digest(producer / 'evaluation_complete.json') != provenance['producer_completion_sha256'][fold]
            or done['fold'] != fold or done['seed'] != 1234 or not done['source_selected_checkpoint_only']
            or done['score_sha256'] != manifest['score_sha256']
            or done['checkpoint_sha256'] != manifest['checkpoint_sha256']):
        raise ValueError('Completed target producer identity changed')
    for name, key in [('metrics.json','metrics_sha256'),('best.pt','checkpoint_sha256'),
                      ('source_validation.json','source_validation_sha256')]:
        if checkpoint_digest(producer / name) != done[key]:
            raise ValueError('Target producer artifact changed: ' + name)
    if read(producer / 'source_validation.json')['passed'] is not True:
        raise ValueError('Producer source validation failed')
    for name, digest in done['score_sha256'].items():
        if checkpoint_digest(producer / name) != digest:
            raise ValueError('Producer score export changed')
    features = TARGET_FEATURES if fold == 'f1' else producer / 'target_features'
    if checkpoint_digest(features / 'complete.json') != manifest['target_features_sha256']:
        raise ValueError('Target feature completion changed')
    return producer, features


@torch.inference_mode()
def score_arrays(model, arrays, device, inputs='concat', *, batch_size=132, last_layer_only=False):
    if batch_size not in (1, 132):
        raise ValueError('Keep prescribed native head batch')
    count = len(arrays['wavlm_layers']); values = []
    for start in range(0, count, batch_size):
        batch = {}
        if inputs != 'clap':
            wave = arrays['wavlm_layers'][start:start+batch_size]
            if last_layer_only: wave = wave[:, -1]
            batch['wavlm'] = torch.from_numpy(wave).to(device)
        if inputs in ('clap', 'concat'):
            batch['clap'] = torch.from_numpy(arrays['clap_projected'][start:start+batch_size]).to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            scores = model(batch)['logit'].float()
        if scores.shape != (len(next(iter(batch.values()))),) or not torch.isfinite(scores).all():
            raise ValueError('Invalid/nonfinite target scores; no skipping')
        values.append(scores.cpu())
    return torch.cat(values).numpy()


def finish_evaluation(model, run, output, fold, name, family, config, done, validation, provenance,
                      device, logger, *, inputs='concat', batch_size=132, last_layer_only=False, bootstrap_resamples=1000):
    """Shared cached export for this fixed study and the original cached F4."""
    if output.exists():
        raise FileExistsError('Existing evaluation preserved; no overwrite/retry')
    if not validation.get('passed') or not validation.get('full_source_score_bit_exact'):
        raise ValueError('Full source verification required before target access')
    check_guards(provenance)
    validation.update(scorer_sha256=provenance['code_sha256'],script_sha256=checkpoint_digest(Path(__file__)))
    output.mkdir(parents=True, exist_ok=False)
    for filename in ('best.pt','config.json','training_complete.json','source_calibration.json'):
        shutil.copy2(run / filename, output / filename)
    shutil.copy2(run / 'metrics.json', output / 'source_metrics.json')
    save_json(output / 'source_validation.json', validation)
    producer, features = verify_producer(fold, provenance)
    manifest = load_score_manifest(output, fold, 1234, done['best_checkpoint_sha256'], create=True)
    for d in FOLDS[fold]['sources']:
        filename = f'source_dev_{d}.csv'; shutil.copy2(run / filename, output / filename)
        manifest['score_sha256'][filename] = checkpoint_digest(output / filename)
    inference = {'input':'centered common interval up to10s; native-unpadded frozen WavLM; CLAP repeatpad',
                 'head_batch':batch_size, 'precision':config['precision'], 'score':'scalar spoof logit',
                 'encoder_execution':False, 'feature_inputs':inputs, 'last_layer_only':last_layer_only}
    save_json(output / 'evaluation_config.json', {'model_name':name, 'fold':fold, 'seed':1234,
        'training_run':str(run.resolve()),'training_completion_sha256':checkpoint_digest(run / 'training_complete.json'),
        'target_feature_producer':str(producer.resolve()),'target_features':str(features.resolve()),
        'inference':inference,'provenance':provenance,'target_tuning':False,
        'exploratory_after_prior_target_exposure':True,'bootstrap_resamples':bootstrap_resamples,'bootstrap_seed':2026,
        'comparison_disclosure':config.get('comparison_disclosure','single-seed frozen-cache ablation; historical target exposure')})
    chunks = f1_chunks(features, CACHE, DATA) if fold == 'f1' else fold_chunks(producer, CACHE, fold, DATA)
    ids, values = [], []
    for identifiers, arrays in chunks:
        check_guards(provenance)
        ids.extend(identifiers)
        values.extend(score_arrays(model, arrays, device, inputs, batch_size=batch_size,
                                   last_layer_only=last_layer_only).tolist())
        replace_json(output / 'progress.json',{'phase':'cached target scoring','scored':len(ids)})
        if len(ids) % 10240 == 0:
            logger.info('%s %s TARGET %d cached recordings',name,fold.upper(),len(ids))
    spec = FOLDS[fold]; expected = canonical_score_labels(spec['target'],spec['target_split'],DATA)
    if ids != list(expected) or len(values) != len(expected):
        raise ValueError('Incomplete/reordered target export')
    write_score_csv(output / 'target_scores.csv',zip(ids,itertools.repeat(spec['target']),expected.values(),values))
    manifest['score_sha256']['target_scores.csv'] = checkpoint_digest(output / 'target_scores.csv')
    manifest.update(model_family=family,inference=inference,training_run=str(run.resolve()),
        training_metrics_copy='source_metrics.json',source_validation_sha256=checkpoint_digest(output / 'source_validation.json'),
        target_features_sha256=checkpoint_digest(features / 'complete.json'),target_features_path=str(features.resolve()),
        scorer_sha256=provenance['code_sha256'],exploratory_after_prior_target_exposure=True)
    save_score_manifest(output,manifest)
    del model,arrays; gc.collect()
    if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    replace_json(output / 'progress.json',{'phase':'metrics and conditional bootstrap','scored':len(ids),'total':len(ids)})
    logger.info('%s %s METRICS | %d conditional bootstrap resamples',name,fold.upper(),bootstrap_resamples)
    report=evaluate_run(output,fold,seed=1234,source_data_root=DATA,target_data_root=DATA,bootstrap_resamples=bootstrap_resamples)
    check_guards(provenance)
    if read(output / 'source_calibration.json') != read(run / 'source_calibration.json'):
        raise ValueError('Original calibration changed')
    save_json(output / 'evaluation_complete.json',{'model_name':name,'fold':fold,'seed':1234,
        'checkpoint_sha256':done['best_checkpoint_sha256'],'metrics_sha256':checkpoint_digest(output / 'metrics.json'),
        'source_validation_sha256':manifest['source_validation_sha256'],'score_sha256':manifest['score_sha256'],
        'target_features_sha256':manifest['target_features_sha256'],
        'source_selected_checkpoint_only':True,'no_target_tuning':True,'exploratory_after_prior_target_exposure':True})
    replace_json(output / 'progress.json',{'phase':'complete','scored':len(ids),'total':len(ids)})
    logger.info('COMPLETE %s %s | EER %.4f%% AUROC %.6f',name,fold.upper(),100*report['target']['eer'],report['target']['auroc'])
    return report


def evaluate(run, output, fold, variant, provenance, device, logger, retained, *, bootstrap_resamples=1000):
    if output.exists():raise FileExistsError('Existing evaluation preserved')
    key=f'{variant}/{fold}'
    if (checkpoint_digest(run / 'training_complete.json') != provenance['training_completion_sha256'][key]
            or checkpoint_digest(run / 'best.pt') != provenance['checkpoint_sha256'][key]):
        raise ValueError('Launch-bound training changed')
    config,done,model=verify_training(run,fold,variant,DATA);model.to(device)
    validation=source_reload(model,run,config,fold,variant,device,logger,retained)
    train.guards(config['provenance'])
    return finish_evaluation(model,run,output,fold,f'MoE-{variant} {NAMES[variant]}','lacf_moe_ablations',
        config,done,validation,provenance,device,logger,inputs=FEATURE_INPUTS[variant],bootstrap_resamples=bootstrap_resamples)


def main():
    if (LAUNCH / 'started.json').exists():raise FileExistsError('Prior launch preserved; no retry')
    provenance=read(LAUNCH / 'provenance.json');check_guards(provenance)
    if read(train.FAMILY / 'launch/complete.json')['completed_fits']!=24:
        raise ValueError('Training study incomplete')
    free=runtime();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logger=logging.getLogger('moe_ablations_evaluation');retained={}
    save_json(LAUNCH / 'started.json',{'fits':24,'seed':1234,'gpu_free_mib':free,'encoder_execution':False})
    completed=[]
    for variant in CORE[1:]:
        for fold in FOLDS:
            folder=f'{variant}_{NAMES[variant]}'
            replace_json(LAUNCH / 'progress.json',{'variant':variant,'fold':fold,'completed':len(completed),'phase':'source verification'})
            evaluate(train.FAMILY / folder / fold / '1234',OUTPUT / folder / fold / '1234',
                     fold,variant,provenance,'cuda',logger,retained)
            completed.append({'variant':variant,'fold':fold,'completion_sha256':checkpoint_digest(OUTPUT / folder / fold / '1234/evaluation_complete.json')})
    save_json(LAUNCH / 'complete.json',{'evaluations':completed,'seed':1234,'target_tuning':False})
    replace_json(LAUNCH / 'progress.json',{'phase':'all24 evaluations complete','completed':24})


if __name__=='__main__':main()
