"""Small, shared LODO dataset contract. No target is read during training."""

from pathlib import Path

from audio_deepfake_detection.data import (
    Record,
    read_asvspoof2019,
    read_asvspoof5,
    read_cfad,
    read_speechfake,
)


FOLDS = {
    "f1": {"sources": ("asv2019", "asv5", "cfad"), "target": "speechfake", "target_split": "test"},
    "f2": {"sources": ("asv2019", "asv5", "speechfake"), "target": "cfad", "target_split": "test_unseen"},
    "f3": {"sources": ("asv2019", "cfad", "speechfake"), "target": "asv5", "target_split": "eval"},
    "f4": {"sources": ("asv5", "cfad", "speechfake"), "target": "asv2019", "target_split": "eval"},
}

DISPLAY_NAMES = {
    "asv2019": "ASVspoof2019",
    "asv5": "ASVspoof5",
    "cfad": "CFAD",
    "speechfake": "SpeechFake",
}

DATASET_FOLDERS = {
    "asv2019": "asvspoof2019",
    "asv5": "asvspoof5",
    "cfad": "cfad",
    "speechfake": "speechfake",
}

READERS = {
    "asv2019": read_asvspoof2019,
    "asv5": read_asvspoof5,
    "cfad": read_cfad,
    "speechfake": read_speechfake,
}


def dataset_root(data_root: str | Path, dataset_name: str) -> Path:
    if dataset_name not in READERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return Path(data_root) / DATASET_FOLDERS[dataset_name] / "extracted"


def read_dataset(dataset_name: str, split: str, data_root: str | Path) -> list[Record]:
    return READERS[dataset_name](dataset_root(data_root, dataset_name), split)


def utterance_id(path: str | Path, dataset_name: str, data_root: str | Path) -> str:
    """Stable ID within a dataset, identical for source and cache copies."""
    return Path(path).relative_to(dataset_root(data_root, dataset_name)).as_posix()
