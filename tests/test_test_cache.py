from pathlib import Path

import pytest

from scripts.prepare_test_cache import copy_verified, validate_roots, cache_target


def test_copy_and_reuse_are_byte_identical_without_overwriting(tmp_path):
    source, cache = tmp_path/'raw', tmp_path/'cache'
    path = source/'audio/a.wav'
    path.parent.mkdir(parents=True);path.write_bytes(b'full recording bytes')
    first=copy_verified(path,source,cache)
    destination=cache/'audio/a.wav';before=destination.stat().st_mtime_ns
    second=copy_verified(path,source,cache)
    assert first['copied'] and not second['copied']
    assert first['sha256']==second['sha256'] and destination.read_bytes()==path.read_bytes()
    assert destination.stat().st_mtime_ns==before


@pytest.mark.parametrize('existing',[b'bad',b'full recording byteX'])
def test_mismatched_existing_cache_preserved(tmp_path,existing):
    source,cache=tmp_path/'raw',tmp_path/'cache'
    source.mkdir();cache.mkdir();(source/'a.wav').write_bytes(b'full recording bytes')
    destination=cache/'a.wav';destination.write_bytes(existing)
    with pytest.raises(ValueError,match='mismatch'):
        copy_verified(source/'a.wav',source,cache)
    assert destination.read_bytes()==existing


def test_cache_symlink_cannot_change_raw_file(tmp_path):
    source,cache=tmp_path/'raw',tmp_path/'cache'
    source.mkdir();cache.mkdir();raw=source/'a.wav';raw.write_bytes(b'preserve')
    (cache/'a.wav').symlink_to(raw)
    with pytest.raises(ValueError,match='Unsafe'):
        copy_verified(raw,source,cache)
    assert raw.read_bytes()==b'preserve'


@pytest.mark.parametrize('nested',['same','inside_source','source_inside_cache'])
def test_root_overlap_rejected(tmp_path,nested):
    raw=tmp_path/'raw'
    args={'same':(raw,raw),'inside_source':(raw,raw/'cache'),'source_inside_cache':(raw/'data',raw)}[nested]
    with pytest.raises(ValueError):validate_roots(*args)


def test_cached_membership_and_labels_checked_not_just_count(tmp_path,monkeypatch):
    import scripts.prepare_test_cache as cache_script
    source,cache,state=tmp_path/'source',tmp_path/'cache',tmp_path/'state'
    source.mkdir();cache.mkdir();state.mkdir()
    raw=cache_script.dataset_root(source,'cfad')/'audio/a.wav';raw.parent.mkdir(parents=True);raw.write_bytes(b'bytes')
    monkeypatch.setattr(cache_script,'read_dataset',lambda *args:[(str(raw),0)])
    monkeypatch.setattr(cache_script,'canonical_score_labels',lambda domain,split,root,**kwargs:{'audio/a.wav':0 if root==source else 1})
    with pytest.raises(ValueError,match='membership/labels'):
        cache_target('cfad',source,cache,1,state,1)
    assert not (state/'cfad_test_unseen_complete.json').exists()
    assert (cache_script.dataset_root(cache,'cfad')/'audio/a.wav').read_bytes()==b'bytes'


def test_complete_target_cache_preserves_canonical_cfad_paths_and_labels(tmp_path):
    import json
    from audio_deepfake_detection.protocol import dataset_root,canonical_score_labels
    source,cache,state=tmp_path/'source',tmp_path/'cache',tmp_path/'state'
    source.mkdir();cache.mkdir();state.mkdir()
    base=dataset_root(source,'cfad')/'clean_version/test_unseen_clean'
    for relative,data in [('real_clean/a.wav',b'genuine bytes'),('fake_clean/b.wav',b'spoof bytes')]:
        path=base/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    result=cache_target('cfad',source,cache,1,state,1)
    assert result['phase']=='complete' and result['expected_records']==2
    assert (result['bona_fide'],result['spoof'])==(1,1)
    assert canonical_score_labels('cfad','test_unseen',source)==canonical_score_labels('cfad','test_unseen',cache)
    rows=[json.loads(line) for line in (state/'cfad_test_unseen_files.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(row['copied'] for row in rows)


def test_insufficient_space_creates_no_partial_destination(tmp_path,monkeypatch):
    import scripts.prepare_test_cache as cache_script
    from collections import namedtuple
    source,cache=tmp_path/'raw',tmp_path/'cache'
    source.mkdir();cache.mkdir();raw=source/'a.wav';raw.write_bytes(b'preserve full source')
    Usage=namedtuple('Usage','total used free')
    monkeypatch.setattr(cache_script.shutil,'disk_usage',lambda root:Usage(1,1,0))
    with pytest.raises(RuntimeError,match='space'):
        copy_verified(raw,source,cache)
    assert not (cache/'a.wav').exists() and raw.read_bytes()==b'preserve full source'
