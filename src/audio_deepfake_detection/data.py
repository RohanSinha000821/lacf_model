from __future__ import annotations

import csv
from pathlib import Path

import soundfile as sf
import torch
import torchaudio



Record = tuple[str, int]

# Global label convention:
# 0 = bona fide
# 1 = spoof


# ASVspoof 2019 LA

ASV2019_SPLITS = {
    "train": {
        "protocol": "ASVspoof2019.LA.cm.train.trn.txt",
        "audio_dir": "ASVspoof2019_LA_train/flac",
    },
    "dev": {
        "protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
        "audio_dir": "ASVspoof2019_LA_dev/flac",
    },
    "eval": {
        "protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
        "audio_dir": "ASVspoof2019_LA_eval/flac",
    },
}


def read_asvspoof2019( root: str | Path,split: str,) -> list[Record]:

    if split not in ASV2019_SPLITS:
        raise ValueError(
            f"Unknown ASVspoof2019 split: {split}"
        )

    root = Path(root)
    info = ASV2019_SPLITS[split]

    protocol = (root/ "ASVspoof2019_LA_cm_protocols"/ info["protocol"])

    audio_root = root / info["audio_dir"]

    if not protocol.is_file():
        raise FileNotFoundError(protocol)

    records: list[Record] = []

    with protocol.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"{protocol}:{line_number}: "
                    f"expected 5 fields, got {len(parts)}"
                )

            _, utterance_id, _, _, key = parts

            key = key.lower()

            if key == "bonafide":
                label = 0

            elif key == "spoof":
                label = 1

            else:
                raise ValueError(
                    f"{protocol}:{line_number}: "
                    f"unknown label {key!r}"
                )

            audio_path = (
                audio_root
                / f"{utterance_id}.flac"
            )

            records.append(
                (str(audio_path), label)
            )

    return records


# ASVspoof 5 Track 1

ASV5_SPLITS = {
    "train": {
        "protocol": "ASVspoof5.train.tsv",
        "audio_dir": "flac_T",
    },
    "dev": {
        "protocol": "ASVspoof5.dev.track_1.tsv",
        "audio_dir": "flac_D",
    },
    "eval": {
        "protocol": "ASVspoof5.eval.track_1.tsv",
        "audio_dir": "flac_E_eval",
    },
}


def read_asvspoof5(root: str | Path,split: str,) -> list[Record]:

    if split not in ASV5_SPLITS:
        raise ValueError(
            f"Unknown ASVspoof5 split: {split}"
        )

    root = Path(root)
    info = ASV5_SPLITS[split]

    protocol = (
        root
        / "protocols"
        / info["protocol"]
    )

    audio_root = root / info["audio_dir"]

    if not protocol.is_file():
        raise FileNotFoundError(protocol)

    records: list[Record] = []

    with protocol.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 10:
                raise ValueError(
                    f"{protocol}:{line_number}: "
                    f"expected 10 fields, got {len(parts)}"
                )

            utterance_id = parts[1]
            key = parts[8].lower()

            if key == "bonafide":
                label = 0

            elif key == "spoof":
                label = 1

            else:
                raise ValueError(
                    f"{protocol}:{line_number}: "
                    f"unknown label {key!r}"
                )

            audio_path = (
                audio_root
                / f"{utterance_id}.flac"
            )

            records.append(
                (str(audio_path), label)
            )

    return records



# CFAD

CFAD_SPLITS = {
    "train": "train_clean",
    "dev": "dev_clean",
    "test_seen": "test_seen_clean",
    "test_unseen": "test_unseen_clean",
}


def read_cfad(root: str | Path,split: str,) -> list[Record]:

    if split not in CFAD_SPLITS:
        raise ValueError(
            f"Unknown CFAD split: {split}"
        )

    root = Path(root)

    split_root = (
        root
        / "clean_version"
        / CFAD_SPLITS[split]
    )

    real_root = split_root / "real_clean"
    fake_root = split_root / "fake_clean"

    if not real_root.is_dir():
        raise FileNotFoundError(real_root)

    if not fake_root.is_dir():
        raise FileNotFoundError(fake_root)

    records: list[Record] = []

    # Bona fide = 0
    for path in sorted(
        real_root.rglob("*.wav")
    ):
        records.append(
            (str(path), 0)
        )

    # Spoof = 1
    for path in sorted(
        fake_root.rglob("*.wav")
    ):
        records.append(
            (str(path), 1)
        )

    return records



# SpeechFake

SPEECHFAKE_SPLITS = {
    "train": "train_all.csv",
    "dev": "dev_all.csv",
    "test": "test_all.csv",
}


def _speechfake_metadata_path(root: Path,split: str,) -> Path:

    return (root/ "metadata"/ "experiments"/ "baseline"/ SPEECHFAKE_SPLITS[split])


def _read_speechfake_metadata(root: Path,split: str,) -> list[dict[str, str]]:

    metadata_path = _speechfake_metadata_path(root,split,)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            metadata_path
        )

    with metadata_path.open("r",encoding="utf-8",newline="",) as handle:

        return list(
            csv.DictReader(handle)
        )


def _index_speechfake_rows(rows: list[dict[str, str]],*,split: str,) -> dict[str, tuple[str, ...]]:

    indexed: dict[str,tuple[str, ...],] = {}

    for row in rows:

        relative_path = Path(
            row["file"].strip()
        ).as_posix()

        signature = (
            row.get("label", "").strip().lower(),
            row.get("generator", "").strip(),
            row.get("model", "").strip(),
            row.get("speaker", "").strip(),
            row.get("language", "").strip(),
        )

        if (relative_path in indexed and indexed[relative_path] != signature):
            raise ValueError(
                "Conflicting SpeechFake metadata "
                f"within {split} for {relative_path}"
            )

        indexed[relative_path] = signature

    return indexed


def read_speechfake(root: str | Path,split: str,) -> list[Record]:

    if split not in SPEECHFAKE_SPLITS:
        raise ValueError(
            f"Unknown SpeechFake split: {split}"
        )

    root = Path(root)

    indexed_rows = _index_speechfake_rows(_read_speechfake_metadata( root,split,),split=split,)

    excluded_paths: set[str] = set()

    # Prevent source train/dev leakage.
    if split == "train":

        dev_rows = _index_speechfake_rows(_read_speechfake_metadata(root,"dev",),split="dev",)

        excluded_paths = (indexed_rows.keys() & dev_rows.keys())

        for relative_path in excluded_paths:

            if (indexed_rows[relative_path]!= dev_rows[relative_path]):
                raise ValueError(
                    "Conflicting SpeechFake train/dev "
                    f"metadata for {relative_path}"
                )

    records: list[Record] = []

    for relative_path, signature in indexed_rows.items():

        if relative_path in excluded_paths:
            continue

        label_text = signature[0]

        if label_text == "bonafide":
            label = 0

        elif label_text == "spoof":
            label = 1

        else:
            raise ValueError(
                "Unknown SpeechFake label: "
                f"{label_text!r}"
            )

        audio_path = ( root/ relative_path)

        records.append((str(audio_path), label))

    return records




# Common audio utilities


def load_audio_segment(path: str | Path,*, seconds: float | None = None,  random_crop: bool = False,) -> tuple[torch.Tensor, int]:
    """
    Load mono float32 audio at its native sample rate.

    This function does not define any model policy.

    The caller decides:
        - whether to use the full utterance,
        - the segment duration,
        - whether the segment is random.

    Returns
    -------
    waveform:
        1-D float32 tensor [time]

    sample_rate:
        Native sample rate.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with sf.SoundFile(path,mode="r",) as audio:

        sample_rate = int(audio.samplerate)

        total_frames = int(len(audio))

        if total_frames <= 0:
            raise ValueError(
                f"Empty audio file: {path}"
            )

        if seconds is None:
            start_frame = 0
            frames_to_read = total_frames

        else:

            if seconds <= 0:
                raise ValueError(
                    "seconds must be positive"
                )

            target_frames = max( 1,round(seconds * sample_rate),)

            if total_frames <= target_frames:
                start_frame = 0
                frames_to_read = total_frames

            else:

                frames_to_read = target_frames

                max_start = (total_frames - target_frames)

                if random_crop:

                    start_frame = int(
                        torch.randint(
                            low=0,
                            high=max_start + 1,
                            size=(1,),
                        ).item()
                    )

                else:

                    start_frame = (max_start // 2)

        audio.seek(start_frame)

        data = audio.read(
            frames=frames_to_read,
            dtype="float32",
            always_2d=True,
        )

    waveform = torch.from_numpy(
        data
    )

    # soundfile returns:
    # [time, channels]
    #
    # Convert to mono:
    # [time]
    waveform = waveform.mean( dim=1)

    return ( waveform.contiguous(),sample_rate,)


def resample_audio(waveform: torch.Tensor,source_rate: int,target_rate: int,) -> torch.Tensor:
    """
    Resample a 1-D waveform.
    """

    if source_rate <= 0:
        raise ValueError("source_rate must be positive")

    if target_rate <= 0:
        raise ValueError("target_rate must be positive")

    if source_rate == target_rate:
        return waveform

    return torchaudio.functional.resample(waveform,orig_freq=source_rate,new_freq=target_rate,)