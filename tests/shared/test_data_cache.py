import csv
import sys

import pytest

from audio_deepfake_detection.data import _index_speechfake_rows, read_speechfake
from scripts.prepare_cache import parse_args


def row(file="audio/a.wav", **updates):
    return {"file": file, "label": "bonafide", "generator": "human", "speaker": "s1",
            "language": "en", "extra_metadata": "original", **updates}


def write_metadata(root, split, rows):
    path = root / "metadata" / "experiments" / "baseline" / f"{split}_all.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_speechfake_deduplicates_normalized_paths_and_all_equal_metadata():
    indexed = _index_speechfake_rows([row(), row("./audio/sub/../a.wav", label=" BONAFIDE ")], split="dev")
    assert list(indexed) == ["audio/a.wav"]


@pytest.mark.parametrize("field", ["label", "generator", "speaker", "language", "extra_metadata"])
def test_speechfake_conflicting_duplicate_field(field):
    with pytest.raises(ValueError, match="Conflicting"):
        _index_speechfake_rows([row(), row(**{field: "changed"})], split="dev")


def test_speechfake_identical_train_dev_overlap_removed(tmp_path):
    write_metadata(tmp_path, "train", [row(), row("audio/b.wav", label="spoof")])
    write_metadata(tmp_path, "dev", [row("./audio/a.wav")])
    assert read_speechfake(tmp_path, "train") == [(str(tmp_path / "audio/b.wav"), 1)]
    assert read_speechfake(tmp_path, "dev") == [(str(tmp_path / "audio/a.wav"), 0)]


def test_speechfake_conflicting_overlap_not_hidden(tmp_path):
    write_metadata(tmp_path, "train", [row()])
    write_metadata(tmp_path, "dev", [row(extra_metadata="changed")])
    with pytest.raises(ValueError, match="train/dev"):
        read_speechfake(tmp_path, "train")


@pytest.mark.parametrize("file", ["", "../outside.wav", "/outside.wav", "audio/../../outside.wav"])
def test_speechfake_invalid_relative_path(file):
    with pytest.raises(ValueError, match="relative path"):
        _index_speechfake_rows([row(file)], split="dev")


@pytest.mark.parametrize("workers", ["0", "-1"])
def test_cache_rejects_nonpositive_workers(monkeypatch, workers):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["prepare_cache.py", "--datasets", "speechfake", "--workers", workers])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def test_cache_defaults_without_environment(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["prepare_cache.py", "--datasets", "speechfake", "--workers", "1"])
    args = parse_args()
    assert args.workers == 1
    assert str(args.cache_root) == "/mnt/drive/audio-deepfake-cache"
