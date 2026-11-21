"""Bounded source fitting/export checks for both prepared cached-head trainers."""
import json

import pytest
import torch

from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.probes import FeatureSplit
from scripts.feature_cache import train_heads_4s as runner


def bank():
    torch.manual_seed(1234)
    return {(domain,split):FeatureSplit([(f'{i}.wav',i%2)for i in range(8)],
        [f'{i}.wav'for i in range(8)],torch.tensor([0,1]*4),torch.arange(8)[:,None],
        torch.randn(8,13,768),torch.randn(8,512))
        for domain in runner.FOLDS['f1']['sources']for split in ('train','dev')}


@pytest.mark.parametrize('name',['wavlm_wa','lacf_moe'])
def test_single_epoch_selected_source_reload_export_and_no_target(tmp_path,name):
    b=bank();prototypes=unit_vectors(torch.randn(8,512));output=tmp_path/name/'f1'/'1234'
    done=runner.fit(b,prototypes,name,'f1',1234,output,{'cache':'synthetic4s'},'cpu',max_epochs=1)
    assert done['best_epoch']==done['final_epoch']==1
    assert done['full_source_scores_bit_exact']and done['selected_checkpoint_reloaded_and_source_verified']
    assert done['target_access']is False
    assert len(done['source_scores_sha256'])==3
    config=json.loads((output/'config.json').read_text())
    assert config['segment_seconds']==4 and config['train_views']=='one fixed random common interval'
    assert config['model']['trainable_parameters']==(1551 if name=='wavlm_wa'else 967338)
    assert (output/'source_calibration.json').is_file()and not(output/'target_scores.csv').exists()
    with pytest.raises(FileExistsError):runner.fit(b,prototypes,name,'f1',1234,output,{},'cpu',max_epochs=1)


def test_held_target_partition_blocks_fit_before_output(tmp_path):
    b=bank();b['speechfake','dev']=next(iter(b.values()))
    with pytest.raises(ValueError,match='Only current fold source'):
        runner.fit(b,unit_vectors(torch.randn(8,512)),'lacf_moe','f1',1234,tmp_path/'new',{},'cpu',max_epochs=1)
    assert not(tmp_path/'new').exists()


def test_prepared_native_recipes_are_explicit():
    w,m=runner.recipe('wavlm_wa'),runner.recipe('lacf_moe')
    assert w['sampling']=='domain'and w['loss']=='weighted CE [9,1]'and w['batch_size']==96
    assert m['sampling']=='domain_class'and m['loss']=='detection BCE only'and m['batch_size']==132
    assert w['max_epochs']==100 and m['max_epochs']==30 and w['patience']==m['patience']==5
    with pytest.raises(ValueError):runner.recipe('unknown')
