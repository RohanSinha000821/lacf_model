"""Step 2 only: source train/dev fixed crops, thirteen WavLM layers and raw CLAP."""

import argparse
import hashlib
import itertools
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Subset

from audio_deepfake_detection.lacf.data import _initialize_worker
from audio_deepfake_detection.lacf.diagnostic_cache import (
    FixedCropCollator, FixedCropDataset, all_layer_features, crop_intervals, grouped_views,
    pooled_wavlm_layers, projected_clap_audio,
)
from audio_deepfake_detection.lacf.model import unit_vectors
from audio_deepfake_detection.lacf.pretrained import load_local_model
from audio_deepfake_detection.lacf.training import device_batch, report_progress
from audio_deepfake_detection.protocol import checkpoint_digest, read_dataset, utterance_id


ROOT = Path(__file__).resolve().parents[2]
COUNTS = {"asv2019": (25380, 24844), "asv5": (182357, 140950),
          "cfad": (38400, 14400), "speechfake": (699608, 117463)}


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def verify_scientific_identity(original, identity):
    # JSON stores integer config keys (e.g. id2label) as strings. Compare the
    # serialized form without relaxing revision, processor or crop checks.
    for name in ("processors", "pretrained", "precision", "views", "crop_seed", "crop_rng"):
        if json.loads(json.dumps(identity[name])) != original[name]:
            raise ValueError(f"Preserved cache scientific/input identity changed: {name}")


def source_records(data_root):
    sources = {}
    for domain, counts in COUNTS.items():
        for split, count in zip(("train", "dev"), counts):
            records = read_dataset(domain, split, data_root)
            ids = [utterance_id(path, domain, data_root) for path, _ in records]
            if len(records) != count or len(set(ids)) != count:
                raise ValueError(f"Canonical membership/count changed: {domain}/{split}")
            sources[domain, split] = (records, ids)
    return sources


def loader(records, ids, collator, *, training, workers, start=0):
    options = {"prefetch_factor": 2} if workers else {}
    dataset = FixedCropDataset(records, ids, training=training)
    if start:
        dataset = Subset(dataset, range(start, len(dataset)))
    return DataLoader(dataset, batch_size=None,
                      collate_fn=FixedCropCollator(collator), num_workers=workers,
                      worker_init_fn=_initialize_worker, pin_memory=True, **options)


def encode_chunk(model, records, batch_size, clap_batch_size=None):
    size = sum(len(record["views"]) for record in records)
    arrays = {"wavlm_layers": np.empty((size, 13, 768), np.float32),
              "clap_projected": np.empty((size, 512), np.float32)}
    branches = ("wavlm", "clap") if clap_batch_size is None else ("wavlm",)
    for rows, batch in grouped_views(records, batch_size, branches=branches):
        if clap_batch_size is None:
            encoded = all_layer_features(model, batch, "cuda")
            for key in arrays:
                arrays[key][rows] = encoded[key]
        else:
            arrays["wavlm_layers"][rows] = pooled_wavlm_layers(model.wavlm, batch["wavlm"], "cuda")
    if clap_batch_size is not None:
        if clap_batch_size < 1:
            raise ValueError("Expected positive CLAP batch size")
        views = [view for record in records for view in record["views"]]
        for start in range(0, len(views), clap_batch_size):
            selected = views[start:start + clap_batch_size]
            inputs = {name: torch.cat([view["clap"][name] for view in selected])
                      for name in selected[0]["clap"]}
            arrays["clap_projected"][start:start + len(selected)] = projected_clap_audio(model.clap, inputs, "cuda")
    return arrays


def preserved_chunks(directory, records, names, *, training, completed=None):
    """Continue explicitly requested feature extraction, never model training.

    Completed splits keep their existing manifest chain. The interrupted split
    additionally verifies payload hashes, canonical IDs and exact fixed views.
    """
    chunks, done, unique_rows = [], 0, 0
    for index, chunk in enumerate(sorted(directory.glob("chunk_*"))):
        if chunk.name != f"chunk_{index:05d}" or not (chunk / "complete.json").is_file():
            raise ValueError(f"Incomplete/nonsequential shard preserved; inspect before continuing: {chunk}")
        digest = checkpoint_digest(chunk / "complete.json")
        marker = json.loads((chunk / "complete.json").read_text())
        descriptor = {"path": chunk.name, "records": marker["records"], "unique_rows": marker["unique_rows"],
                      "completion_sha256": digest}
        if completed is not None:
            if index >= len(completed["chunks"]) or descriptor != completed["chunks"][index]:
                raise ValueError("Completed split manifest chain changed")
        else:
            for name, expected in marker["files_sha256"].items():
                if name not in ("wavlm_layers.npy", "clap_projected.npy", "records.json"):
                    raise ValueError("Unexpected cache payload")
                if checkpoint_digest(chunk / name) != expected:
                    raise ValueError(f"Preserved shard checksum mismatch: {chunk / name}")
            metadata = json.loads((chunk / "records.json").read_text())
            if len(metadata) != marker["records"]:
                raise ValueError("Preserved shard record count changed")
            offset = 0
            for row in metadata:
                if done >= len(records) or (row["index"], row["id"], row["label"]) != (done, names[done], records[done][1]):
                    raise ValueError("Preserved canonical record identity changed")
                intervals, aliases = crop_intervals(row["frames"], row["rate"], row["id"], training=training)
                if row["intervals"] != [list(value) for value in intervals] or row["view_rows"] != [offset + v for v in aliases]:
                    raise ValueError("Preserved fixed crop or row aliases changed")
                offset += len(intervals)
                done += 1
            if offset != marker["unique_rows"]:
                raise ValueError("Preserved unique-row count changed")
            for key, shape in (("wavlm_layers", (offset, 13, 768)), ("clap_projected", (offset, 512))):
                array = np.load(chunk / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError("Invalid preserved feature array")
        if completed is not None:
            done += marker["records"]
        unique_rows += marker["unique_rows"]
        chunks.append(descriptor)
    if done > len(records) or completed is not None and (done != len(records) or chunks != completed["chunks"]
            or unique_rows != completed["unique_rows"] or done != completed["records"]):
        raise ValueError("Incomplete preserved split manifest")
    return chunks, done, unique_rows


def probe(model, collator, sources, output):
    """Bounded source-only numerical/memory checks; never a recipe search."""
    logging.info("Step 2 source probe: checking sixteen native short/long recordings")
    evidence = json.loads((ROOT / "docs/local/lacf_part2_20261008/source_checks.json").read_text())
    records, ids = [], []
    for (domain, split), (canonical, canonical_ids) in sources.items():
        lookup = dict(canonical)
        for kind in ("longest", "shortest"):
            _, path, label = evidence["splits"][f"{domain}/{split}"][kind]
            if lookup.get(path) != label:
                raise ValueError("Source-only probe example no longer belongs to its split")
            records.append((path, label))
            ids.append(f"{domain}/{split}/{canonical_ids[next(i for i, row in enumerate(canonical) if row[0] == path)]}")
    prepared = list(loader(records, ids, collator, training=False, workers=4))
    native = encode_chunk(model, prepared, 1)
    direct = []
    for record in prepared:
        batch = device_batch(record["views"][0], "cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            primary = model.encode_audio(batch)
        index = len(direct)
        w = torch.from_numpy(native["wavlm_layers"][index:index + 1, -1]).cuda()
        c = torch.from_numpy(native["clap_projected"][index:index + 1]).cuda()
        # The existing CLAP API normalizes once, then LACF normalizes its result.
        direct.append({"id": record["id"], "last_layer_bit_exact": torch.equal(w, primary["z_w"]),
                       "clap_bit_exact": torch.equal(unit_vectors(unit_vectors(c)), primary["v_c"])})
    if not all(row["last_layer_bit_exact"] and row["clap_bit_exact"] for row in direct):
        save_json(output / "probe_failed.json", {"direct_parity": direct})
        raise ValueError("All-layer extraction differs from the native primary scorer")
    # Real longest-source inputs, repeated only for an operational 32-input
    # memory/throughput probe. These are not scientific training examples.
    longest = [i for i, row in enumerate(prepared)
               if row["views"][0]["wavlm"]["input_values"].shape[-1] == 160000]
    indices = [longest[i % len(longest)] for i in range(32)]
    benchmark = [prepared[i] for i in indices]
    expected = {key: value[indices] for key, value in native.items()}
    results = []
    for size in (1, 16, 32):
        logging.info("Step 2 source probe: 32 longest inputs, encoder batch %d", size)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        try:
            values = encode_chunk(model, benchmark, size)
            result = {"batch_size": size, "seconds": time.monotonic() - started,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "bit_exact": all(np.array_equal(values[key], expected[key]) for key in native),
                      "max_absolute_difference": {key: float(np.max(np.abs(values[key] - expected[key])))
                                                  for key in native}}
            del values
        except torch.cuda.OutOfMemoryError as error:
            result = {"batch_size": size, "failure": str(error), "bit_exact": False}
            torch.cuda.empty_cache()
        results.append(result)
        logging.info("Source batch result: %s", result)
    # Retain native arithmetic if a batched kernel changes the recorded features.
    if not results[0]["bit_exact"]:
        save_json(output / "probe_failed.json", {"direct_parity": direct, "batch_probes": results})
        raise ValueError("Native feature extraction is not reproducible")
    successful = [row for row in results if row["bit_exact"]]
    selected = min(successful, key=lambda row: row["seconds"])["batch_size"] if successful else 1
    worker_results = []
    sample_records, sample_ids = [], []
    for (domain, split), (canonical, canonical_ids) in sources.items():
        if split == "train":
            sample_records.extend(canonical[:48])
            sample_ids.extend(f"{domain}/{split}/{value}" for value in canonical_ids[:48])
    for workers in (8, 16, 32):
        logging.info("Step 2 CPU preprocessing probe: %d workers", workers)
        stream = iter(loader(sample_records, sample_ids, collator, training=True, workers=workers))
        for _ in range(64):
            next(stream)
        started, unique = time.monotonic(), 0
        for record in stream:
            unique += len(record["views"])
        seconds = time.monotonic() - started
        worker_results.append({"workers": workers, "seconds_after_64_record_warmup": seconds,
                               "unique_views": unique, "unique_views_per_second": unique / seconds})
        del stream
        logging.info("Source loading result: %s", worker_results[-1])
    save_json(output / "probe.json", {"direct_parity": direct, "batch_probes": results,
                                      "encoder_batch_size": selected, "source_only": True,
                                      "worker_probes": worker_results,
                                      "scope": "16 source extrema, repeated 32 longest inputs, 192 source train records; no target access or probe fitting"})


def build(model, collator, sources, revisions, args, logger):
    probe_report = json.loads((args.output / "probe.json").read_text())
    batch_size = args.encoder_batch_size or probe_report["encoder_batch_size"]
    execution_sha256 = None
    if args.clap_batch_size is not None or args.encoder_batch_size is not None:
        if args.benchmark_report is None:
            raise ValueError("Accelerated extraction requires its source-only benchmark report")
        benchmark = json.loads(args.benchmark_report.read_text())
        if not benchmark.get("source_only") or benchmark["selected"] != {
                "wavlm_batch_size": batch_size, "clap_batch_size": args.clap_batch_size}:
            raise ValueError("Batch settings differ from measured source-only selection")
        for name, expected in benchmark["code_sha256"].items():
            if checkpoint_digest(ROOT / name) != expected:
                raise ValueError("Extraction code changed after the measured benchmark")
    identity = {"step": 2, "source_splits_only": True, "precision": "BF16 mixed, FP32 pooling/features",
                "wavlm_layers": "projected frontend then Transformer outputs 1–12; masked mean",
                "clap_features": "raw audio_projection output before get_audio_features normalization",
                "views": {"train": ["center", "random_0", "random_1"], "dev": ["center"]},
                "crop_seed": 1234, "crop_rng": "sha256(1234, canonical domain/split/id, view), torch.Generator",
                "encoder_batch_size": batch_size, "batching": "equal unpadded waveform lengths only",
                "workers": args.workers, "prefetch": 2, "chunk_records": args.chunk_records,
                "processors": collator.configuration(), "pretrained": revisions,
                "torch": str(torch.__version__), "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "platform": platform.platform(),
                "code_sha256": {str(path.relative_to(ROOT)): checkpoint_digest(path) for path in
                    [Path(__file__).resolve(), ROOT / "src/audio_deepfake_detection/lacf/diagnostic_cache.py",
                     ROOT / "src/audio_deepfake_detection/lacf/model.py",
                     ROOT / "src/audio_deepfake_detection/lacf/pretrained.py", ROOT / "src/audio_deepfake_detection/data.py"]},
                "cache_copy_checksum_status": "verified_owner_reported_ssh1",
                "completed_primary_checkpoints_loaded": False, "trained_probe_seeds": [1234, 1235, 1236]}
    if args.continue_cache:
        original = json.loads((args.output / "config.json").read_text())
        verify_scientific_identity(original, identity)
        for name, expected in original["code_sha256"].items():
            if name not in ("scripts/lacf/cache_diagnostics.py", "src/audio_deepfake_detection/lacf/diagnostic_cache.py"):
                if checkpoint_digest(ROOT / name) != expected:
                    raise ValueError("Underlying model or audio code changed")
        save_json(args.output / "execution_accelerated.json", {
            **identity, "encoder_batch_size": batch_size, "clap_batch_size": args.clap_batch_size,
            "original_config_sha256": checkpoint_digest(args.output / "config.json"),
            "source_benchmark_sha256": checkpoint_digest(args.benchmark_report),
            "numerical_policy": "equal-length WavLM; CLAP fixed mel shape; BF16 batch rounding recorded, not bit-exact",
            "authorization": "Owner requested faster GPU extraction on 2026-10-16; preserve completed shards"})
        execution_sha256 = checkpoint_digest(args.output / "execution_accelerated.json")
    else:
        save_json(args.output / "config.json", identity)
    totals = {}
    for (domain, split), (records, ids) in sources.items():
        directory = args.output / domain / split
        directory.mkdir(parents=True, exist_ok=args.continue_cache)
        canonical_hash = hashlib.sha256(json.dumps(list(zip(ids, [label for _, label in records]))).encode()).hexdigest()
        names = [f"{domain}/{split}/{identifier}" for identifier in ids]
        completed = json.loads((directory / "complete.json").read_text()) if (directory / "complete.json").exists() else None
        if completed is not None and completed["canonical_ids_labels_sha256"] != canonical_hash:
            raise ValueError("Completed split canonical membership changed")
        chunks, done, unique_rows = preserved_chunks(directory, records, names, training=split == "train", completed=completed)
        if completed is not None:
            totals[f"{domain}/{split}"] = completed
            logger.info("%s/%s: preserving %d complete records; no re-encoding", domain, split, done)
            continue
        reused = done
        stream = iter(loader(records, names, collator, training=split == "train", workers=args.workers, start=done))
        started = time.monotonic()
        while prepared := list(itertools.islice(stream, args.chunk_records)):
            if [row["index"] for row in prepared] != list(range(done, done + len(prepared))):
                raise ValueError("Source record order changed")
            arrays = encode_chunk(model, prepared, batch_size, args.clap_batch_size)
            chunk = directory / f"chunk_{len(chunks):05d}"
            chunk.mkdir(exist_ok=False)
            metadata, offset = [], 0
            for record in prepared:
                metadata.append({**{key: value for key, value in record.items() if key != "views"},
                                 "view_rows": [offset + row for row in record["view_rows"]]})
                offset += len(record["views"])
            files = {}
            for key, array in arrays.items():
                path = chunk / f"{key}.npy"
                with path.open("xb") as handle:
                    np.save(handle, array, allow_pickle=False)
                files[path.name] = checkpoint_digest(path)
            save_json(chunk / "records.json", metadata)
            files["records.json"] = checkpoint_digest(chunk / "records.json")
            save_json(chunk / "complete.json", {"records": len(prepared), "unique_rows": offset,
                                              "files_sha256": files, "execution_sha256": execution_sha256})
            chunks.append({"path": chunk.name, "records": len(prepared), "unique_rows": offset,
                           "completion_sha256": checkpoint_digest(chunk / "complete.json")})
            done += len(prepared)
            unique_rows += offset
            report_progress(logger, f"{domain}/{split} cache (starting at {reused})", done - reused, len(records) - reused, started)
        if done != len(records):
            raise ValueError("Incomplete source split")
        totals[f"{domain}/{split}"] = {"records": done, "unique_rows": unique_rows,
                                      "logical_views": done * (3 if split == "train" else 1),
                                      "canonical_ids_labels_sha256": canonical_hash,
                                      "seconds": time.monotonic() - started, "reused_records": reused,
                                      "seconds_scope": "this execution only", "chunks": chunks}
        save_json(directory / "complete.json", totals[f"{domain}/{split}"])
    save_json(args.output / "complete.json", {"step": 2, "splits": totals,
                                              "config_sha256": checkpoint_digest(args.output / "config.json"),
                                              "accelerated_execution_sha256": execution_sha256})
    logger.info("Step 2 complete. Stop before trained probes; share results first.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--chunk-records", type=int, default=128)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--continue-cache", action="store_true", help="Explicitly preserve/verify existing features and extract only missing records; not training resume")
    parser.add_argument("--encoder-batch-size", type=int)
    parser.add_argument("--clap-batch-size", type=int)
    parser.add_argument("--benchmark-report", type=Path)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 0 or args.chunk_records < 1:
        parser.error("Invalid loading settings")
    if any(value is not None and value < 1 for value in (args.encoder_batch_size, args.clap_batch_size)):
        parser.error("Invalid encoder batch size")
    if args.continue_cache and (args.probe or not (args.output / "config.json").is_file()):
        parser.error("Continuation requires an existing cache config and cannot run the initial probe")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA and BF16 support required; no precision fallback")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "complete.json").exists() or (args.output / "config.json").exists() and not args.continue_cache or (args.output / "probe.json").exists() and args.probe:
        raise FileExistsError("Existing extraction/probe preserved; no overwrite or automatic restart")
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.manual_seed(1234)
    np.random.seed(1234)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)
    sources = source_records(args.data_root)
    model, collator, revisions = load_local_model()
    model = model.cuda().eval()
    if args.probe:
        probe(model, collator, sources, args.output)
    else:
        if args.workers is None:
            report = json.loads((args.output / "probe.json").read_text())
            args.workers = max(report["worker_probes"], key=lambda row: row["unique_views_per_second"])["workers"]
        build(model, collator, sources, revisions, args, logger)


if __name__ == "__main__":
    main()
