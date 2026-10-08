"""Small, shared LODO dataset contract. No target is read during training."""

import hashlib
import json
import math
import re
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

FINAL_SEEDS = (1234, 2345, 3456)


def final_seeds_for(model: str) -> tuple[int, ...]:
    """Owner single-seed decisions: WavLM, AASIST and initial LACF study."""
    return (1234,) if model in ("wavlm_bs96", "aasist", "lacf") else FINAL_SEEDS


def validate_run_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None:
        raise ValueError("Run name must contain only letters, digits, underscores, or hyphens")
    return value

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


def canonical_score_labels(dataset_name, split, data_root, *, records=None) -> dict[str, int]:
    """Canonical split membership, independent of the physical cache location."""
    if records is None:
        records = read_dataset(dataset_name, split, data_root)
    expected = {}
    for path, label in records:
        identifier = utterance_id(path, dataset_name, data_root)
        if identifier in expected or label not in (0, 1):
            raise ValueError(f"Invalid canonical record for {dataset_name}/{split}: {identifier}")
        expected[identifier] = label
    if not expected:
        raise ValueError(f"No canonical records for {dataset_name}/{split}")
    return expected


def checkpoint_digest(path: Path) -> str:
    """Also used to bind exported CSV bytes to their checkpoint manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_completed_run(run_dir: Path, fold_name: str, seed: int) -> tuple[dict, str]:
    best_path = run_dir / "best.pt"
    completion_path = run_dir / "training_complete.json"
    if not best_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"Training has not completed for {run_dir}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("fold") != fold_name or completion.get("seed") != seed:
        raise ValueError("Training completion record does not match requested fold/seed")
    if run_dir.name != str(seed) or run_dir.parent.name != fold_name:
        raise ValueError("Run directory does not match requested fold/seed")
    best_epoch = completion.get("best_epoch")
    final_epoch = completion.get("final_epoch")
    if (type(best_epoch) is not int or type(final_epoch) is not int
            or not 1 <= best_epoch <= final_epoch):
        raise ValueError("Invalid best/final epoch in training completion record")
    verify_source_macro(completion.get("best_macro_source_dev_eer"), completion.get("best_macro_source_dev_eer"))
    return completion, checkpoint_digest(best_path)


def verify_source_macro(observed, expected) -> None:
    if (not isinstance(observed, (int, float)) or not isinstance(expected, (int, float))
            or not math.isfinite(observed) or not math.isfinite(expected)
            or not 0 <= observed <= 1 or not 0 <= expected <= 1
            or abs(observed - expected) > 1e-5):
        raise ValueError("Source-dev macro EER differs from the completion record or is invalid; target remains unopened")


def load_score_manifest(run_dir: Path, fold_name: str, seed: int, digest: str, *, create=False) -> dict:
    fold = FOLDS[fold_name]
    identity = {
        "model": run_dir.parent.parent.name,
        "fold": fold_name,
        "seed": seed,
        "checkpoint_sha256": digest,
        "training_completion_sha256": checkpoint_digest(run_dir / "training_complete.json"),
        "splits": {
            **{f"source_dev_{domain}.csv": {"dataset": domain, "split": "dev"} for domain in fold["sources"]},
            "target_scores.csv": {"dataset": fold["target"], "split": fold["target_split"]},
        },
    }
    path = run_dir / "score_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("Score manifest has a stale checkpoint, wrong fold/seed, or wrong split identity")
        if not isinstance(manifest.get("score_sha256"), dict):
            raise ValueError("Score manifest lacks score-file checksums")
        return manifest
    if not create or any(run_dir.glob("source_dev_*.csv")) or (run_dir / "target_scores.csv").exists():
        raise ValueError("Scores cannot be used without a checkpoint manifest")
    manifest = {**identity, "score_sha256": {}}
    save_score_manifest(run_dir, manifest)
    return manifest


def save_score_manifest(run_dir: Path, manifest: dict) -> None:
    path = run_dir / "score_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_score_digest(path: Path, manifest: dict) -> None:
    expected = manifest["score_sha256"].get(path.name)
    if expected is None or checkpoint_digest(path) != expected:
        raise ValueError(f"Score file is unregistered or changed since export: {path}")
