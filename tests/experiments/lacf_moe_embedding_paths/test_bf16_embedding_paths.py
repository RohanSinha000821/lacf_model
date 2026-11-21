"""Historical BF16 arithmetic, fresh source completion and registry isolation."""
import json
import logging

import numpy as np
import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches,FeatureSplit
from audio_deepfake_detection.experiments.lacf_moe_embedding_paths import model as fp32_models
from scripts.experiments.lacf_moe_embedding_paths import train_bf16 as runner


def bank():
    return {(d,s):FeatureSplit([(str(i),i%2)for i in range(8)],
        [f'{d}_{s}_{i}'for i in range(8)],torch.tensor([0,1]*4),
        torch.arange(8)[:,None].repeat(1,3 if s=='train'else 1),
        torch.randn(8,13,768),torch.randn(8,512))
        for d in runner.core.FOLDS['f1']['sources']for s in('train','dev')}


@pytest.mark.parametrize('variant',runner.models.VARIANTS)
def test_bf16_logits_and_unchanged_weights_and_sensitive_math(variant):
    p=unit_vectors(torch.randn(8,512));torch.manual_seed(1234)
    original=fp32_models.build_model(variant,p).eval();torch.manual_seed(1234)
    model=runner.models.build_model(variant,p).eval()
    assert all(torch.equal(v,original.state_dict()[k])for k,v in model.state_dict().items())
    assert sum(p.numel()for p in model.parameters())==1005418
    inputs={'wavlm':torch.randn(135,13,768),'clap':torch.randn(135,512)}
    seen=[];hook=model.backend.classifier[-1].register_forward_hook(lambda m,a,out:seen.append(out.dtype))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        a,b=model(inputs),original(inputs)
    hook.remove()
    assert seen==[torch.bfloat16] and torch.equal(a['logit'],b['logit'])
    assert a['relations'].dtype==torch.float32 and a['p_w'].dtype==torch.float32
    assert a['features'].shape==(135,99)
    with pytest.raises(ValueError,match='BF16 autocast required'):model(inputs)
    arrays={'wavlm_layers':inputs['wavlm'].numpy(),'clap_projected':inputs['clap'].numpy()}
    scores=runner.evaluator.score_arrays(model,arrays,'cpu')
    assert scores.dtype==np.float32 and np.array_equal(scores,torch.from_numpy(scores).bfloat16().float().numpy())
    assert model.configuration()['score_precision']==runner.models.PRECISION


@pytest.mark.parametrize('variant',runner.models.VARIANTS)
def test_fresh_bf16_fit_export_and_source_reload(tmp_path,monkeypatch,variant):
    b=bank();p=unit_vectors(torch.randn(8,512));logger=logging.getLogger('bf16_test')
    cache=tmp_path/'cache';cache.mkdir();(cache/'complete.json').write_text('{}')
    monkeypatch.setattr(runner.core,'CACHE',cache)
    evidence={'code_sha256':{},'cache':str(cache),'cache_completion_sha256':runner.core.checkpoint_digest(cache/'complete.json'),'target_access':False}
    with runner.bindings()as core:
        proof=runner.preflight(b,p,'f1',variant,evidence,'cpu',logger)
        assert proof['actual_logit_dtype']=='torch.bfloat16' and proof['full_source_dev_reload_bit_exact']
        path=tmp_path/'proof.json';core.save_json(path,proof)
        provenance={**evidence,'source_validation_record':str(path),'source_validation_sha256':core.checkpoint_digest(path)}
        run=tmp_path/variant/'f1'/'1234'
        done=core.fit(b,p,run,'f1',variant,provenance,'cpu',logger,max_epochs=1)
        assert done['study']==runner.STUDY and done['selected_checkpoint_reloaded_and_source_verified']
        assert runner.verify_completion(run,'f1',variant,core.labels_for(b,'f1'))==done
        cfg=core.read(run/'config.json');assert cfg['precision']==runner.models.PRECISION
        assert cfg['model']['score_precision']==runner.models.PRECISION
        assert cfg['model']['objective']=='detection BCE only' and cfg['model']['classifier']==[99,64,16,1]
        m=runner.models.build_model(variant,p)
        state=torch.load(run/'best.pt',weights_only=True);m.load_state_dict(state['model_state_dict'])
        monkeypatch.setattr(runner.evaluator,'source_bank',lambda *a,**k:{(d,'dev'):b[d,'dev']for d in core.FOLDS['f1']['sources']})
        validation=runner.evaluator.source_reload(m,run,cfg,'f1',variant,'cpu',logger,{})
        assert validation['full_source_score_bit_exact']
        assert not(run/'target_scores.csv').exists()
        with pytest.raises(FileExistsError):core.fit(b,p,run,'f1',variant,provenance,'cpu',logger,max_epochs=1)


def test_source_failure_prevents_target_access(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.evaluator,'verify_training',lambda *a:({}, {},torch.nn.Linear(1,1)))
    def fail(*a):raise ValueError('source mismatch')
    monkeypatch.setattr(runner.evaluator,'source_reload',fail)
    monkeypatch.setattr(runner.evaluator,'finish_evaluation',lambda *a,**k:pytest.fail('Target accessed'))
    with runner.bindings(),pytest.raises(ValueError,match='source mismatch'):
        runner.evaluate_fit(tmp_path,tmp_path/'new','f1',runner.models.VARIANTS[0],{}, {},'cpu',logging.getLogger('test'))
    assert not(tmp_path/'new').exists()


def test_bindings_restore_all_modules_after_failure():
    original=(runner.core.build_model,runner.core.verify_completion,runner.core.FAMILY,
              runner.evaluator.build_model,runner.evaluator.CORE,runner.cache_loader.LAUNCH)
    with pytest.raises(RuntimeError):
        with runner.bindings():
            assert runner.core.CORE[1:]==runner.models.VARIANTS
            assert runner.cache_loader.LAUNCH==runner.LAUNCH
            raise RuntimeError('discarded test')
    assert original==(runner.core.build_model,runner.core.verify_completion,runner.core.FAMILY,
                      runner.evaluator.build_model,runner.evaluator.CORE,runner.cache_loader.LAUNCH)
