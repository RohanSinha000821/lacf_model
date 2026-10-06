"""Read-only P2–P4 metadata adapters; training readers remain unchanged."""

from dataclasses import dataclass, field
from pathlib import Path
import csv
import re

from .data import ASV2019_SPLITS, ASV5_SPLITS, _read_speechfake_metadata, _index_speechfake_rows
from .protocol import dataset_root, read_dataset, utterance_id


@dataclass(frozen=True)
class EvaluationExample:
    identifier: str
    path: str
    label: int
    groups: dict[str, str] = field(default_factory=dict)
    pair_id: str = ""


def relative_path(value):
    """Reject escaping paths from third-party metadata before resolving audio."""
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative metadata path: {value!r}")
    return path


def validate_examples(examples):
    if not examples:
        raise ValueError("Evaluation partition is empty")
    seen = set()
    for item in examples:
        if not item.identifier or item.identifier in seen or item.label not in (0, 1):
            raise ValueError(f"Duplicate/invalid evaluation record: {item.identifier}")
        seen.add(item.identifier)
    return examples


def protocol_rows(path, width):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != width:
                raise ValueError(f"{path}:{number}: expected {width} fields, got {len(parts)}")
            yield parts


def binary_label(value):
    if value not in ("bonafide", "spoof"):
        raise ValueError(f"Unknown official label: {value}")
    return int(value == "spoof")


def native_examples(dataset, split, data_root):
    """Exact P1 membership, enriched with official generator/attack/codec fields."""
    root = dataset_root(data_root, dataset)
    records = read_dataset(dataset, split, data_root)
    metadata = {}
    if dataset == "speechfake":
        rows = _read_speechfake_metadata(root, split)
        indexed = _index_speechfake_rows(rows, split=split)
        for identifier, signature in indexed.items():
            row = dict(signature)
            label = binary_label(row["label"])
            generator = row.get("generator", "")
            if label and generator not in ("TTS", "VC", "NV"):
                raise ValueError(f"Unknown SpeechFake generator annotation: {generator}")
            metadata[identifier] = (label, {"generator": generator if label else "bonafide",
                                          "model": row.get("model", "unknown"),
                                          "language": row.get("language", "unknown")}, "")
    elif dataset in ("asv2019", "asv5"):
        info = (ASV2019_SPLITS if dataset == "asv2019" else ASV5_SPLITS)[split]
        protocol = (root / "ASVspoof2019_LA_cm_protocols" / info["protocol"]
                    if dataset == "asv2019" else root / "protocols" / info["protocol"])
        for parts in protocol_rows(protocol, 5 if dataset == "asv2019" else 10):
            identifier = f"{info['audio_dir']}/{parts[1]}.flac"
            label = binary_label(parts[4] if dataset == "asv2019" else parts[8])
            attack = parts[3] if dataset == "asv2019" else parts[7]
            groups = {"attack": attack if label else "bonafide"}
            pair = ""
            if dataset == "asv5":
                # '-' and C00 both mean uncoded; CODEC_SEED names the original waveform.
                groups["codec"] = "C00" if parts[3] == "-" else parts[3]
                if not re.fullmatch(r"C(?:0[0-9]|1[01])", groups["codec"]):
                    raise ValueError(f"Unknown ASV5 codec: {parts[3]}")
                groups["codec_quality"] = parts[4]
                pair = parts[5] if parts[5] != "-" else parts[1]
            if identifier in metadata:
                raise ValueError(f"Duplicate official metadata ID: {identifier}")
            metadata[identifier] = label, groups, pair
    result = []
    for path, label in records:
        identifier = utterance_id(path, dataset, data_root)
        if dataset == "cfad":
            groups = {"condition": "clean", "partition": split}
            pair = cfad_pair_id(Path(identifier), "clean")
        else:
            observed, groups, pair = metadata[identifier]
            if observed != label:
                raise ValueError(f"Metadata label mismatch: {identifier}")
        result.append(EvaluationExample(identifier, path, label, groups, pair))
    return validate_examples(result)


def cfad_pair_id(path, version):
    stem = path.stem
    if version == "noise":
        match = re.fullmatch(r"(.+)_([^_]+)_snr(-?\d+)", stem)
        if not match:
            raise ValueError(f"Unrecognized CFAD noise filename: {path}")
        stem = match[1]
    elif version == "codec":
        match = re.fullmatch(r"(.+)_(mp3|flac|ogg|m4a|aac|wma)", stem, re.IGNORECASE)
        if not match:
            raise ValueError(f"Unrecognized CFAD codec filename: {path}")
        stem = match[1]
    # Class/source folders disambiguate repeated filenames; no global stem guessing.
    return "/".join((*path.parts[2:-1], stem)).replace("real_clean", "real").replace(
        "fake_clean", "fake").replace("real_noise", "real").replace(
        "fake_noise", "fake").replace("real_codec", "real").replace("fake_codec", "fake")


def cfad_conditions(data_root, split="test_unseen"):
    if split not in ("test_unseen", "test_seen"):
        raise ValueError("P4 CFAD only accepts official test partitions")
    root = dataset_root(data_root, "cfad")
    examples = native_examples("cfad", split, data_root)
    for version, folder in (("noise", "noisy_version"), ("codec", "codec_version")):
        split_root = root / folder / f"{split}_{version}"
        for prefix, label in (("real", 0), ("fake", 1)):
            class_root = split_root / f"{prefix}_{version}"
            if not class_root.is_dir():
                raise FileNotFoundError(class_root)
            for path in sorted(class_root.rglob("*.wav")):
                groups = {"condition": version, "partition": split}
                if version == "noise":
                    match = re.fullmatch(r"(.+)_([^_]+)_snr(-?\d+)", path.stem)
                    if not match:
                        raise ValueError(f"Unrecognized CFAD noise filename: {path}")
                    groups.update(noise=match[2], snr=match[3], noise_snr=f"{match[2]}_snr{match[3]}")
                else:
                    groups["codec"] = path.stem.rsplit("_", 1)[-1].lower()
                identifier = path.relative_to(root).as_posix()
                examples.append(EvaluationExample(identifier, str(path), label, groups,
                                                  cfad_pair_id(Path(identifier), version)))
    return validate_examples(examples)


def asv2021_df(data_root, *, keys=None, audio_root=None):
    """Official DF eval/notrim subset only, excluding progress and hidden trials."""
    root = Path(data_root) / "asvspoof2021" / "extracted"
    keys = Path(keys) if keys else root / "keys" / "DF" / "CM" / "trial_metadata.txt"
    if not keys.is_file():
        raise FileNotFoundError(f"Official DF keys missing: {keys}. Supply --df-keys from DF-keys-full; the trial-ID list has no labels.")
    audio_root = Path(audio_root) if audio_root else root / "ASVspoof2021_DF_eval" / "flac"
    result = []
    for parts in protocol_rows(keys, 13):
        if parts[7] != "eval" or parts[6] != "notrim":
            continue
        identifier = parts[1] + ".flac"
        relative_path(identifier)
        result.append(EvaluationExample(identifier, str(audio_root / identifier), binary_label(parts[5]),
                                        {"codec": parts[2], "origin": parts[3], "attack": parts[4],
                                         "vocoder": parts[8], "partition": "eval_notrim"}))
    return validate_examples(result)


def partialspoof(data_root):
    """Utterance-level eval stress test, not segment-localization evaluation."""
    root = Path(data_root) / "partialspoof" / "extracted"
    keys = root / "protocols/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.eval.trl.txt"
    result = []
    for parts in protocol_rows(keys, 5):
        identifier = parts[1] + ".wav"
        relative_path(identifier)
        result.append(EvaluationExample(identifier, str(root / "eval/database/eval/con_wav" / identifier),
                                        binary_label(parts[4]), {"attack": parts[3]}))
    return validate_examples(result)


def mlaad_mailabs(data_root, mailabs_root):
    """Pair official MLAAD fake metadata with its referenced genuine M-AILABS files."""
    if mailabs_root is None or not Path(mailabs_root).is_dir():
        raise FileNotFoundError("MLAAD requires --mailabs-root containing the genuine M-AILABS corpus; no substitute negatives are permitted")
    root = Path(data_root) / "mlaad"
    genuine_root = Path(mailabs_root)
    result, genuine = [], {}
    for metadata in sorted((root / "fake").glob("*/*/meta.csv")):
        with metadata.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            if not {"path", "original_file", "language", "model_name", "architecture"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing official MLAAD metadata columns: {metadata}")
            for row in reader:
                path = relative_path(row["path"])
                original = relative_path(row["original_file"])
                # Published meta.csv paths are dataset-relative, not meta.csv-relative.
                if path.parts[0] != "fake":
                    raise ValueError(f"MLAAD path must start with fake/: {path}")
                groups = {"language": row["language"], "model": row["model_name"], "architecture": row["architecture"]}
                result.append(EvaluationExample(path.as_posix(), str(root / path), 1, groups, original.as_posix()))
                identifier = "mailabs/" + original.as_posix()
                item = EvaluationExample(identifier, str(genuine_root / original), 0,
                                         {"source_locale": original.parts[0]}, original.as_posix())
                if identifier in genuine and genuine[identifier] != item:
                    raise ValueError(f"Conflicting M-AILABS reference: {identifier}")
                genuine[identifier] = item
    missing = [item.path for item in genuine.values() if not Path(item.path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced M-AILABS files missing; first: {missing[0]}")
    return validate_examples(result + list(genuine.values()))
