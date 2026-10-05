from __future__ import annotations

import argparse
import os
import shutil

from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path

from audio_deepfake_detection.data import (
    read_asvspoof2019,
    read_asvspoof5,
    read_cfad,
    read_speechfake,
)


# Supported datasets
# Each invocation adds selected train/dev data to the shared cache.


DATASET_NAMES = (
    "asv2019",
    "asv5",
    "cfad",
    "speechfake",
)


DISPLAY_NAMES = {
    "asv2019": "ASVspoof2019",
    "asv5": "ASVspoof5",
    "cfad": "CFAD",
    "speechfake": "SpeechFake",
}


# Arguments

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=("Prepare local audio cache for LODO training"))

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=DATASET_NAMES,
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "/mnt/salt/datasets/audio-deepfake")),
        help="Original dataset root",
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/mnt/drive/audio-deepfake-cache"),
        help="Local cache root",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args



# Dataset locations


def dataset_root(base: Path,dataset_name: str,) -> Path:

    folder_names = {
        "asv2019": "asvspoof2019",
        "asv5": "asvspoof5",
        "cfad": "cfad",
        "speechfake": "speechfake",
    }

    return (base/ folder_names[dataset_name]/ "extracted")


# Dataset reading

def read_records(dataset_name: str,root: Path,split: str,):

    if dataset_name == "asv2019":
        return read_asvspoof2019(root,split,)

    if dataset_name == "asv5":
        return read_asvspoof5(root,split,)

    if dataset_name == "cfad":
        return read_cfad(root,split,)

    if dataset_name == "speechfake":
        return read_speechfake(root,split,)

    raise ValueError(f"Unknown dataset: {dataset_name}")


# Small protocol / metadata files


def copy_support_files(dataset_name: str,source_root: Path,cache_root: Path,) -> None:

    if dataset_name == "asv2019":
        relative_paths = (
            Path("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"),
            Path("ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"),
        )

    elif dataset_name == "asv5":
        relative_paths = (
            Path("protocols/ASVspoof5.train.tsv"),
            Path("protocols/ASVspoof5.dev.track_1.tsv"),
        )

    elif dataset_name == "speechfake":
        relative_paths = (
            Path("metadata/experiments/baseline/train_all.csv"),
            Path("metadata/experiments/baseline/dev_all.csv"),
        )

    else:
        # CFAD does not require a protocol file.
        return

    for relative_path in relative_paths:
        source = source_root / relative_path
        destination = cache_root / relative_path

        destination.parent.mkdir(parents=True,exist_ok=True,)

        if (destination.is_file() and destination.stat().st_size == source.stat().st_size):
            continue

        shutil.copyfile(source,destination,)



# Copy one audio file


def copy_audio(source_path: Path,source_dataset_root: Path,cache_dataset_root: Path,) -> bool:

    relative_path = (source_path.relative_to(source_dataset_root))

    destination = (cache_dataset_root/ relative_path)

    destination.parent.mkdir(parents=True,exist_ok=True,)

    # Resumable:
    # if the complete file is already present,
    # do not copy it again.
    if destination.is_file():

        if (destination.stat().st_size== source_path.stat().st_size):
            return False

    shutil.copyfile(source_path,destination,)

    return True



# Cache one dataset

def cache_dataset(dataset_name: str,source_base: Path,cache_base: Path,workers: int,) -> None:

    source_root = dataset_root(source_base,dataset_name,)
    cache_root = dataset_root(cache_base,dataset_name,)

    print()
    print("=" * 60)
    print(DISPLAY_NAMES[dataset_name])
    print("=" * 60)

    train_records = read_records(dataset_name,source_root,"train",)
    dev_records = read_records(dataset_name,source_root,"dev",)

    print("train:",len(train_records),)

    print("dev:",len(dev_records),)

    # Remove duplicate paths if any.
    paths = sorted(
        {
            Path(path)
            for path, _ in (train_records + dev_records)
        }
    )

    print("unique audio files:",len(paths),)

    cache_root.mkdir(parents=True,exist_ok=True,)

    copy_support_files(dataset_name,source_root,cache_root,)

    def worker(path: Path,) -> bool:

        return copy_audio(path,source_root,cache_root,)

    copied = 0

    # Python 3.12's executor.map submits the entire iterable eagerly.
    # Keep the number of pending copy jobs bounded for large datasets.
    with ThreadPoolExecutor(max_workers=workers) as executor:

        processed = 0

        for batch in batched(paths, 1000):

            for was_copied in executor.map(worker, batch):

                processed += 1

                if was_copied:
                    copied += 1

                if processed % 5000 == 0:

                    print(
                        f"{processed:,}/"
                        f"{len(paths):,} "
                        f"files processed"
                    )

    print("new files copied:",f"{copied:,}",)



    # Verify cached train/dev records

    cached_train = read_records(
        dataset_name,
        cache_root,
        "train",
    )

    cached_dev = read_records(
        dataset_name,
        cache_root,
        "dev",
    )

    if (len(cached_train)!= len(train_records)):
        raise RuntimeError(
            f"{dataset_name}: "
            "cached train count mismatch"
        )

    if (len(cached_dev)!= len(dev_records)):
        raise RuntimeError(
            f"{dataset_name}: "
            "cached dev count mismatch"
        )

    missing = []

    for path, _ in (cached_train + cached_dev):

        if not Path(path).is_file():
            missing.append(path)
            if len(missing) >= 10:
                break

    if missing:

        raise RuntimeError(
            "Missing cached files:\n"
            + "\n".join(missing)
        )

    print("verification: PASS")



# Main

def main() -> None:

    args = parse_args()

    source_root = args.source_root.resolve()
    cache_root = args.cache_root.resolve()
    salt_root = Path("/mnt/salt").resolve()

    if (cache_root == source_root or source_root in cache_root.parents):
        raise ValueError(
            "Cache root must not equal or be inside "
            "the source root"
        )

    if (cache_root == salt_root or salt_root in cache_root.parents):
        raise ValueError(
            "Cache root must not be under /mnt/salt"
        )

    dataset_names = tuple(dict.fromkeys(args.datasets))

    print()
    print("=" * 60)
    print("Preparing local audio cache")
    print("=" * 60)
    print("source:",source_root,)
    print("cache:",cache_root,)
    print( "workers:",args.workers,)
    print()
    print("Datasets:")

    for name in dataset_names:

        print( " ",DISPLAY_NAMES[name],)

    print()

    for dataset_name in dataset_names:

        cache_dataset(
            dataset_name,
            source_root,
            cache_root,
            args.workers,
        )

    print()
    print("=" * 60)
    print("Cache preparation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
