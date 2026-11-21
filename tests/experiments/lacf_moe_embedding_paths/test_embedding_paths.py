"""Capacity, initialization, FP32 gradients and source-only fit/cache gates."""
import json
import logging

import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureBatches, FeatureSplit
from audio_deepfake_detection.lacf_moe.model import LACFMoE
from audio_deepfake_detection.experiments.lacf_moe_embedding_paths import model as models
from scripts.experiments.lacf_moe_embedding_paths import train as runner


def prototypes():return unit_vectors(torch.randn(8,512))


def split(domain,kind):
    return FeatureSplit([(str(i),i%2)for i in range(8)],
        [f'{domain}_{kind}_{i}'for i in range(8)],torch.tensor([0,1]*4),
        torch.arange(8)[:,None].repeat(1,3 if kind=='train' else 1),
        torch.randn(8,13,768),torch.randn(8,512))


def bank():
    return {(d,s):split(d,s)for d in runner.core.FOLDS['f1']['sources']for s in ('train','dev')}


def test_matched_initialization_and_preserved_reference_weights_rng():
    p=prototypes()
    torch.manual_seed(1234);reference=LACFMoE(p);rng=torch.random.get_rng_state()
    built=[]
    for variant in models.VARIANTS:
        torch.manual_seed(1234);m=models.build_model(variant,p)
        assert torch.equal(rng,torch.random.get_rng_state())
        assert sum(x.numel()for x in m.parameters())==models.PARAMETERS[variant]==1005418
        for name in ('adapter','prototypes'):
            if name=='prototypes':assert torch.equal(reference.backend.prototypes,m.backend.prototypes)
            else:assert all(torch.equal(v,m.backend.adapter.state_dict()[k])for k,v in reference.backend.adapter.state_dict().items())
        assert all(torch.equal(v,m.layer_fusion.state_dict()[k])for k,v in reference.layer_fusion.state_dict().items())
        for i in range(4,len(reference.backend.classifier)):
            assert all(torch.equal(v,m.backend.classifier[i].state_dict()[k])for k,v in reference.backend.classifier[i].state_dict().items())
        assert not m.backend.config.semantic_loss and not m.backend.config.bonafide_loss
        built.append(m)
    assert all(torch.equal(v,built[1].state_dict()[k])for k,v in built[0].state_dict().items())
    batch={'wavlm':torch.randn(6,13,768),'clap':torch.randn(6,512)}
    a,b=[m.eval()(batch)for m in built]
    for output in [a,b]:
        assert output['relations'].shape==(6,35)
        assert output['embedding_features'].shape==(6,64)
        assert output['features'].shape==(6,99) and output['logit'].shape==(6,)
    assert torch.equal(a['relations'],b['relations']) and torch.equal(a['routing'],b['routing'])
    assert not torch.equal(a['embedding_features'],b['embedding_features'])


@pytest.mark.parametrize('variant',models.VARIANTS)
def test_fp32_training_and_full_source_reload_export(tmp_path,variant):
    b=bank();p=prototypes();m=models.build_model(variant,p)
    observed=[]
    handles=[x.register_forward_hook(lambda m,args,out:observed.append(out.dtype))for x in m.modules()if isinstance(x,torch.nn.Linear)]
    loader=FeatureBatches([b[d,'train']for d in runner.core.FOLDS['f1']['sources']],
        'concat','cpu',training=True,seed=1234,batch_size=4,max_batches=2)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        losses,updates=runner.core.train_epoch(m,loader,runner.core.make_optimizer(m),'cpu')
    for h in handles:h.remove()
    assert observed and set(observed)=={torch.float32} and updates==2
    assert losses['semantic']==losses['bonafide']==0
    assert all(x.grad is not None and torch.isfinite(x.grad).all()for x in m.embedding_path.parameters())
    provenance={'code_sha256':{},'target_access':False}
    with runner.bindings() as core:
        proof=core.preflight(b,p,'f1',variant,provenance,'cpu',logging.getLogger('test'))
        assert proof['linear_output_dtypes']==['torch.float32'] and proof['serialized_source_reload_bit_exact']
        path=tmp_path/'proof.json';core.save_json(path,proof)
        current={**provenance,'source_validation_record':str(path),'source_validation_sha256':core.checkpoint_digest(path)}
        output=tmp_path/variant/'f1'/'1234'
        done=core.fit(b,p,output,'f1',variant,current,'cpu',logging.getLogger('test'),max_epochs=1)
        assert done['selected_checkpoint_reloaded_and_source_verified']
        cfg=core.read(output/'config.json')
        assert cfg['study']=='lacf_moe_embedding_paths' and cfg['precision']==core.PRECISION
        assert cfg['model']['classifier']==[99,64,16,1]
        assert cfg['model']['backend']['classifier']==[99,64,16,1]
        assert cfg['model']['objective']=='detection BCE only'
        assert cfg['model']['trainable_parameters']==1005418
        assert core.verify_completion(output,'f1',variant,core.labels_for(b,'f1'))==done
        assert not (output/'target_scores.csv').exists()
        with pytest.raises(FileExistsError):
            core.fit(b,p,output,'f1',variant,current,'cpu',logging.getLogger('test'),max_epochs=1)


def test_registry_restores_on_error():
    before={key:getattr(runner.core,key)for key in ['CORE','NAMES','PARAMETERS','build_model','STUDY','FAMILY']}
    with pytest.raises(RuntimeError):
        with runner.bindings():
            assert runner.core.CORE[1:]==models.VARIANTS
            raise RuntimeError('discarded test')
    assert all(getattr(runner.core,key) is value for key,value in before.items())


def test_only_current_source_fold_loaded_and_common_partitions_reused(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.core,'CACHE',tmp_path)
    monkeypatch.setattr(runner,'LAUNCH',tmp_path)
    descriptor={'records':8,'unique_rows':8}
    manifest={'step':2,'splits':{f'{d}/{s}':descriptor for d in ['asv2019','asv5','cfad','speechfake']for s in ['train','dev']}}
    (tmp_path/'complete.json').write_text(json.dumps(manifest))
    calls=[]
    def load(root,d,s,*a,**k):calls.append((d,s));return split(d,s)
    monkeypatch.setattr(runner,'load_split',load)
    monkeypatch.setattr(runner,'wait_resources',lambda *a,**k:None)
    provenance={'code_sha256':{}}
    b={}
    runner.load_fold_bank(b,'f1',provenance,logging.getLogger('test'))
    assert set(b)=={(d,s)for d in runner.core.FOLDS['f1']['sources']for s in ('train','dev')}
    assert not any(d=='speechfake'for d,s in calls)
    old=b['asv2019','train']
    runner.load_fold_bank(b,'f2',provenance,logging.getLogger('test'))
    assert b['asv2019','train'] is old
    assert len(calls)==8 and ('cfad','train')not in b


def test_refuses_target_partition_before_creating_fit(tmp_path):
    b=bank();b['speechfake','dev']=split('speechfake','dev')
    with runner.bindings(),pytest.raises(ValueError,match='Only this fold source'):
        runner.core.fit(b,prototypes(),tmp_path/'f1'/'1234','f1',models.VARIANTS[0],
            {'code_sha256':{}},'cpu',logging.getLogger('test'))


def test_unknown_variant_rejected():
    with pytest.raises(ValueError,match='fixed CLAP path'):models.build_model('extra_candidate',prototypes())
