"""Copy native P1 target audio unchanged; validate membership, labels and file bytes."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
from itertools import batched
import json
import os
from pathlib import Path
import shutil
import tempfile

from audio_deepfake_detection.data import ASV2019_SPLITS, ASV5_SPLITS, SPEECHFAKE_SPLITS
from audio_deepfake_detection.protocol import FOLDS, canonical_score_labels, dataset_root, read_dataset

TARGET_SPLITS = {fold['target']: fold['target_split'] for fold in FOLDS.values()}


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def validate_roots(source, cache):
    source, cache = Path(source).resolve(), Path(cache).resolve()
    if source == cache or source in cache.parents or cache in source.parents:
        raise ValueError('Source and cache roots must be separate, non-nested directories')
    if cache == Path('/mnt/salt') or Path('/mnt/salt') in cache.parents:
        raise ValueError('Cache must not be under /mnt/salt')
    return source, cache


def copy_verified(source, source_root, cache_root, reserve_bytes=0):
    """Exclusive publication; existing cache bytes must match, never overwrite them."""
    source = Path(source)
    relative = source.relative_to(source_root)
    if not source.resolve().is_relative_to(source_root.resolve()):
        raise ValueError(f'Source escapes dataset root: {source}')
    destination = cache_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.parent.resolve().is_relative_to(cache_root.resolve()) or destination.is_symlink():
        raise ValueError(f'Unsafe cache path: {destination}')
    size = source.stat().st_size
    if not source.is_file() or size == 0:
        raise ValueError(f'Invalid empty/non-file audio or metadata: {source}')
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != size:
            raise ValueError(f'Existing cache size/type mismatch; preserved: {destination}')
        sha = digest(source)
        if digest(destination) != sha:
            raise ValueError(f'Existing cache content mismatch; preserved: {destination}')
        return {'path': relative.as_posix(), 'bytes': size, 'sha256': sha, 'copied': False}
    if shutil.disk_usage(cache_root).free < size + reserve_bytes:
        raise RuntimeError('Insufficient cache space; no existing file changed')
    # This owned temporary file is the only file cleaned up on interruption/error.
    fd, temporary_name = tempfile.mkstemp(prefix='.' + destination.name + '.copy-', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        sha = hashlib.sha256()
        written = 0
        with os.fdopen(fd, 'wb') as dst, source.open('rb') as src:
            for chunk in iter(lambda: src.read(1024 * 1024), b''):
                dst.write(chunk); sha.update(chunk); written += len(chunk)
            dst.flush()
        if written != size or source.stat().st_size != size or digest(temporary) != sha.hexdigest():
            raise ValueError(f'Copy verification failed: {source}')
        # Same-filesystem hard link publishes a complete verified file without replacement.
        os.link(temporary, destination)
        return {'path': relative.as_posix(), 'bytes': written, 'sha256': sha.hexdigest(), 'copied': True}
    finally:
        temporary.unlink(missing_ok=True)


def support_path(domain, split):
    if domain == 'asv2019':
        return Path('ASVspoof2019_LA_cm_protocols') / ASV2019_SPLITS[split]['protocol']
    if domain == 'asv5':
        return Path('protocols') / ASV5_SPLITS[split]['protocol']
    if domain == 'speechfake':
        return Path('metadata/experiments/baseline') / SPEECHFAKE_SPLITS[split]
    return None


def cache_target(domain, source_base, cache_base, workers, state_dir, reserve_bytes):
    split = TARGET_SPLITS[domain]
    source_root, cache_root = dataset_root(source_base, domain), dataset_root(cache_base, domain)
    print(f'{timestamp()} Reading canonical {domain}/{split} membership', flush=True)
    records = read_dataset(domain, split, source_base)
    expected = canonical_score_labels(domain, split, source_base, records=records)
    class_counts = Counter(label for _, label in records)
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata = support_path(domain, split)
    metadata_result = copy_verified(source_root / metadata, source_root, cache_root) if metadata else None
    manifest_path = state_dir / f'{domain}_{split}_files.jsonl'
    status = {'dataset': domain, 'split': split, 'source_root': str(source_root), 'cache_root': str(cache_root),
              'expected_records': len(records), 'bona_fide': class_counts[0], 'spoof': class_counts[1],
              'processed': 0, 'copied': 0, 'reused': 0, 'bytes_verified': 0,
              'phase': 'copying', 'observed_at': timestamp(), 'metadata': metadata_result}
    write_json(state_dir / 'status.json', status)
    print(json.dumps(status), flush=True)
    # On explicit restart, recheck every selected file. Journal append preserves prior evidence.
    with manifest_path.open('a', buffering=1) as manifest, ThreadPoolExecutor(max_workers=workers) as executor:
        for batch in batched(records, 250):
            if shutil.disk_usage(cache_base).free < reserve_bytes:
                raise RuntimeError('Cache free space below reserve; stopping without deleting anything')
            paths = [Path(path) for path, _ in batch]
            results = executor.map(lambda path: copy_verified(path, source_root, cache_root, reserve_bytes), paths)
            for (_, label), result in zip(batch, results, strict=True):
                manifest.write(json.dumps({**result, 'label': label}) + '\n')
                status['processed'] += 1
                status['copied' if result['copied'] else 'reused'] += 1
                status['bytes_verified'] += result['bytes']
            status['observed_at'] = timestamp()
            write_json(state_dir / 'status.json', status)
            if status['processed'] % 1000 == 0 or status['processed'] == len(records):
                print(json.dumps(status), flush=True)
    # Path-independent IDs and labels must match exactly, not only total count.
    cached = canonical_score_labels(domain, split, cache_base)
    if cached != expected:
        raise ValueError(f'Cached membership/labels differ for {domain}/{split}')
    status.update(phase='complete', observed_at=timestamp(), canonical_membership_and_labels_match=True,
                  files_sha256_verified=True, journal_sha256=digest(manifest_path), journal=str(manifest_path))
    write_json(state_dir / f'{domain}_{split}_complete.json', status)
    write_json(state_dir / 'status.json', status)
    print(f'{timestamp()} COMPLETE {domain}/{split}: {len(records):,} recordings; full audio unchanged', flush=True)
    return status


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=tuple(TARGET_SPLITS), default=list(TARGET_SPLITS))
    parser.add_argument('--source-root', type=Path, default=Path('/mnt/salt/datasets/audio-deepfake'))
    parser.add_argument('--cache-root', type=Path, default=Path('/mnt/drive/audio-deepfake-cache'))
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--reserve-gib', type=int, default=20)
    args = parser.parse_args()
    if args.workers < 1 or args.reserve_gib < 1:
        parser.error('workers and reserve-gib must be positive')
    return args


def main():
    args = parse_args()
    source, cache = validate_roots(args.source_root, args.cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    state = args.state_dir.resolve()
    if state == source or source in state.parents or state == cache or cache in state.parents:
        raise ValueError('State directory must be separate from source/cache audio roots')
    state.mkdir(parents=True, exist_ok=True)
    lock_path = cache / '.test-cache.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        try:
            for domain in dict.fromkeys(args.datasets):
                completed.append(cache_target(domain, source, cache, args.workers, state, args.reserve_gib * 1024**3))
            write_json(state / 'all_complete.json', {'completed_at': timestamp(), 'datasets': completed,
                       'total_records': sum(item['expected_records'] for item in completed),
                       'audio_transformed': False, 'inference_performed': False})
            print('All requested native P1 target caches verified; no model inference performed.', flush=True)
        except BaseException as error:
            write_json(state / 'failure.json', {'observed_at': timestamp(), 'error': repr(error),
                       'completed_datasets': [item['dataset'] for item in completed], 'files_preserved': True})
            raise


if __name__ == '__main__':
    main()
