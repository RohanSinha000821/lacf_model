"""P2 external zero-shot, P3 generators, P4 robustness: one evaluator for all models.

Default is CPU report-only from registered scores. --export-scores explicitly
enables native WavLM/AASIST inference; it never trains or changes checkpoints.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import uuid

import numpy as np

from audio_deepfake_detection.analysis import evaluate_frozen, group_reports, paired_raw_deltas
from audio_deepfake_detection.evaluation_data import (
    asv2021_df, cfad_conditions, mlaad_mailabs, native_examples, partialspoof,
)
from audio_deepfake_detection.metrics import calibrate_source_scores, read_score_csv, write_score_csv
from audio_deepfake_detection.protocol import (
    FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest, verify_completed_run,
    verify_score_digest, verify_source_macro,
)


def write_json_once(path, value):
    """Exclusive creation: never overwrite an existing frozen record or report."""
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
        os.link(temporary, path)  # Atomic publication, fails if the destination exists.
    finally:
        temporary.unlink(missing_ok=True)


def examples_digest(examples):
    digest = hashlib.sha256()
    for item in sorted(examples, key=lambda item: item.identifier):
        digest.update(json.dumps([item.identifier, item.label, item.groups, item.pair_id],
                                 sort_keys=True).encode() + b"\n")
    return digest.hexdigest()


def selection_identity(run_dir, completion, digest):
    return {"model": run_dir.parent.parent.name, "fold": run_dir.parent.name,
            "seed": int(run_dir.name), "checkpoint_sha256": digest,
            "training_completion_sha256": checkpoint_digest(run_dir / "training_complete.json"),
            "checkpoint_epoch": completion["best_epoch"]}


def verify_selection(record, identity):
    for key, value in identity.items():
        if record.get(key) != value:
            raise ValueError(f"P2 source-only checkpoint selection record mismatch: {key}")
    if (record.get("target_results_used") is not False
            or not isinstance(record.get("source_only_selection_reason"), str)
            or not record["source_only_selection_reason"].strip()):
        raise ValueError("P2 requires a documented source-only selection, with no target feedback")


def source_scores(run_dir, fold_name, seed, digest, root, exporter=None):
    manifest = load_score_manifest(run_dir, fold_name, seed, digest, create=exporter is not None)
    result = {}
    for domain in FOLDS[fold_name]["sources"]:
        path = run_dir / f"source_dev_{domain}.csv"
        if not path.exists() and exporter is not None:
            module, model, device, workers, prefetch = exporter
            result[domain] = module.score_split(model, domain, "dev", root, path, device,
                                                num_workers=workers, prefetch_factor=prefetch, manifest=manifest)
        else:
            verify_score_digest(path, manifest)
            expected = canonical_score_labels(domain, "dev", root)
            result[domain] = read_score_csv(path, domain, expected_labels=expected)
    return calibrate_source_scores(result), manifest


def load_exporter(args, completion):
    """Existing model verifiers and source exporters are reused, not reimplemented."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Native score export requires CUDA; omit --export-scores for CPU reporting")
    script = Path(__file__).parent / args.model / "evaluate.py"
    spec = importlib.util.spec_from_file_location(f"extended_{args.model}_adapter", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    module.verify_checkpoint(checkpoint, completion, args.run_dir.parent.name,
                             int(args.run_dir.name), args.run_dir.parent.parent.name)
    if args.model == "aasist":
        if completion.get("best_checkpoint_sha256") != checkpoint_digest(args.run_dir / "best.pt"):
            raise ValueError("AASIST checkpoint differs from its completion hash")
        model = module.AASIST()
    else:
        model = module.WavLMWA(model_name=checkpoint["config"]["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda")
    return module, model.to(device).eval(), device, args.num_workers, args.prefetch_factor


def export_rows(exporter, model_kind, examples, dataset):
    import torch
    module, model, device, workers, prefetch = exporter
    records = [(item.path, item.label) for item in examples]
    if model_kind == "wavlm":
        loader = module.make_wavlm_eval_loader(module.WavLMDataset(records, training=False), batch_size=1,
                                               num_workers=workers, prefetch_factor=prefetch, pin_memory=True)
    else:
        loader = module.make_eval_loader(module.AASISTDataset(records, training=False),
                                         num_workers=workers, prefetch_factor=prefetch)
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            if model_kind == "wavlm":
                waveforms, masks, labels = batch
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(waveforms.to(device, non_blocking=True), masks.to(device, non_blocking=True))
            else:
                waveforms, labels = batch
                logits = model(waveforms.to(device, non_blocking=True))
            values = model.spoof_score(logits).float().cpu().tolist()
            if len(values) != len(labels):
                raise ValueError("Native export count mismatch")
            for label, value in zip(labels.tolist(), values):
                item = examples[offset]
                if label != item.label:
                    raise ValueError("Evaluation loader reordered labels")
                yield item.identifier, dataset, label, value
                offset += 1
            if offset % 5000 == 0:
                print(f"{dataset}: {offset:,}/{len(examples):,}", flush=True)
    if offset != len(examples):
        raise ValueError("Native export did not score the entire partition")


def ordered_scores(path, dataset, examples):
    expected = {item.identifier: item.label for item in examples}
    read_score_csv(path, dataset, expected_labels=expected)
    with path.open(newline="", encoding="utf-8") as handle:
        indexed = {row["utterance_id"]: float(row["raw_score"]) for row in csv.DictReader(handle)}
    return np.asarray([indexed[item.identifier] for item in examples])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("p2", "p3", "p4"), required=True)
    parser.add_argument("--dataset", choices=("asv2021_df", "partialspoof", "mlaad_mailabs", "speechfake", "asv2019", "asv5", "cfad"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/salt/datasets/audio-deepfake"))
    parser.add_argument("--source-data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--cfad-split", choices=("test_unseen", "test_seen"), default="test_unseen")
    parser.add_argument("--df-keys", type=Path)
    parser.add_argument("--df-audio-root", type=Path)
    parser.add_argument("--mailabs-root", type=Path)
    parser.add_argument("--selection-record", type=Path, help="P2: one frozen source-selected deployment checkpoint per model/seed")
    parser.add_argument("--write-selection-record", type=Path, help="Source-only mode: write record and exit before opening targets")
    parser.add_argument("--selection-reason", help="Source-only reason/rule for choosing this single P2 fold checkpoint")
    parser.add_argument("--confirm-protocol-frozen", action="store_true")
    parser.add_argument("--export-scores", action="store_true", help="Explicit GPU inference permission; default only reads verified score files")
    parser.add_argument("--model", choices=("wavlm", "aasist"), help="Required only for native score export")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    allowed = {"p2": {"asv2021_df", "partialspoof", "mlaad_mailabs"},
               "p3": {"speechfake", "asv2019", "asv5"}, "p4": {"cfad", "asv5"}}
    if args.dataset not in allowed[args.protocol]:
        parser.error("Dataset is not part of this evaluation protocol")
    if not args.confirm_protocol_frozen:
        parser.error("Comparison-wide freeze confirmation is required; do not run while study recipes remain unfrozen")
    if args.export_scores and not args.model:
        parser.error("--export-scores requires --model")
    if args.protocol == "p2" and not (args.selection_record or args.write_selection_record):
        parser.error("P2 requires one documented source-only checkpoint selection record")
    if args.write_selection_record and not (args.protocol == "p2" and args.selection_reason):
        parser.error("Selection record creation requires P2 and a source-only --selection-reason")
    if min(args.num_workers, args.bootstrap_resamples) < 0 or args.prefetch_factor < 1:
        parser.error("Workers/resamples must be nonnegative, prefetch positive")
    args.run_dir = args.run_dir.resolve()
    return args


def main():
    args = parse_args()
    fold_name, seed = args.run_dir.parent.name, int(args.run_dir.name)
    completion, digest = verify_completed_run(args.run_dir, fold_name, seed)
    identity = selection_identity(args.run_dir, completion, digest)
    fold = FOLDS[fold_name]
    if args.protocol != "p2" and fold["target"] != args.dataset:
        raise ValueError("P3/P4 must use the dataset's held-out fold, not a checkpoint trained on that domain")
    selection = json.loads(args.selection_record.read_text()) if args.selection_record else None
    if selection is not None:
        verify_selection(selection, identity)
    suffix = f"cfad_{args.cfad_split}" if args.dataset == "cfad" else args.dataset
    directory = args.run_dir / "analyses" / args.protocol / suffix
    if (directory / "metrics.json").exists() and not args.write_selection_record:
        raise FileExistsError(f"Report already exists: {directory}")
    exporter = load_exporter(args, completion) if args.export_scores else None
    calibration, source_manifest = source_scores(args.run_dir, fold_name, seed, digest,
                                                  args.source_data_root, exporter)
    verify_source_macro(calibration["source_dev"]["macro_eer"], completion["best_macro_source_dev_eer"])
    source_hashes = {k: v for k, v in source_manifest["score_sha256"].items() if k.startswith("source_dev_")}
    if selection is not None and selection.get("source_score_sha256") != source_hashes:
        raise ValueError("P2 source calibration scores differ from the frozen selection record; target remains unopened")
    if args.write_selection_record:
        write_json_once(args.write_selection_record, {**identity, "source_only_selection_reason": args.selection_reason,
                        "target_results_used": False, "source_dev": calibration["source_dev"],
                        "source_score_sha256": source_hashes,
                        "comparison_wide_freeze_confirmed_by_operator": True})
        print(f"Wrote source-only P2 selection; no target opened: {args.write_selection_record}")
        return
    print("Frozen checkpoint and source-only calibration verified; opening evaluation metadata", flush=True)
    if args.dataset == "asv2021_df":
        examples = asv2021_df(args.data_root, keys=args.df_keys, audio_root=args.df_audio_root)
    elif args.dataset == "partialspoof":
        examples = partialspoof(args.data_root)
    elif args.dataset == "mlaad_mailabs":
        examples = mlaad_mailabs(args.data_root, args.mailabs_root)
    elif args.dataset == "cfad":
        examples = cfad_conditions(args.data_root, args.cfad_split)
    else:
        examples = native_examples(args.dataset, fold["target_split"], args.data_root)
    examples = sorted(examples, key=lambda item: item.identifier)
    binding = {**identity, "protocol": args.protocol, "dataset": args.dataset,
               "metadata_sha256": examples_digest(examples), "data_root": str(args.data_root),
               "source_data_root": str(args.source_data_root), "calibration": calibration,
               "source_score_sha256": source_hashes,
               "selection_record_sha256": checkpoint_digest(args.selection_record) if args.selection_record else None}
    # Reuse P1 scores for generator/ASV5 codec decompositions without any GPU work.
    if args.protocol in ("p3", "p4") and args.dataset != "cfad":
        score_path = args.run_dir / "target_scores.csv"
        if score_path.exists():
            verify_score_digest(score_path, source_manifest)
        else:
            score_path = directory / "scores.csv"
    else:
        score_path = directory / "scores.csv"
    manifest_path = directory / "manifest.json"
    if score_path == directory / "scores.csv":
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("binding") != binding:
                raise ValueError("Supplementary scores belong to different checkpoint, metadata, or source calibration")
            if checkpoint_digest(score_path) != manifest.get("score_sha256"):
                raise ValueError("Supplementary score bytes differ from registered export")
        else:
            if score_path.exists():
                raise ValueError("Unregistered supplementary scores; refusing to attribute them to this checkpoint")
            if exporter is None:
                raise FileNotFoundError("No registered scores. Native export requires explicit --export-scores --model after the freeze")
            write_score_csv(score_path, export_rows(exporter, args.model, examples, args.dataset))
            ordered_scores(score_path, args.dataset, examples)
            write_json_once(manifest_path, {"binding": binding, "score_sha256": checkpoint_digest(score_path),
                                           "model_adapter": args.model, "evaluation_batch_size": 1})
    scores = ordered_scores(score_path, args.dataset, examples)
    report = {**binding, "score_sha256": checkpoint_digest(score_path), "comparison_wide_freeze_confirmed_by_operator": True,
              "label_convention": "0=bona_fide,1=spoof; higher_score=spoof", "metric_scale": "fractions_0_to_1",
              "evaluation_code_sha256": {name: checkpoint_digest(Path(__file__).parents[1] / "src/audio_deepfake_detection" / name)
                                          for name in ("analysis.py", "evaluation_data.py", "metrics.py")},
              "script_sha256": checkpoint_digest(Path(__file__)), "bootstrap_resamples": args.bootstrap_resamples,
              "bootstrap_seed": 2026}
    if args.dataset == "cfad":
        # Separate unlabeled normalization populations for the three distributed corpora.
        report["conditions"] = {}
        for condition in ("clean", "noise", "codec"):
            indices = [i for i, item in enumerate(examples) if item.groups["condition"] == condition]
            subset = [examples[i] for i in indices]
            values = scores[indices]
            report["conditions"][condition] = evaluate_frozen(calibration, [x.label for x in subset], values,
                                                              bootstrap_resamples=args.bootstrap_resamples)
            report["conditions"][condition]["groups"] = group_reports(
                subset, values, calibration, ("noise", "snr", "noise_snr") if condition == "noise" else ("codec",),
                bootstrap_resamples=args.bootstrap_resamples)
        report["paired_clean_deltas"] = paired_raw_deltas(examples, scores, calibration, dimension="condition", reference="clean")
        report["normalization_population"] = "each complete official clean/noise/codec test partition independently; subgroups retain parent statistics"
    else:
        report["overall"] = evaluate_frozen(calibration, [x.label for x in examples], scores,
                                             bootstrap_resamples=args.bootstrap_resamples)
        dimensions = ("generator", "model") if args.dataset == "speechfake" else ("attack",)
        if args.protocol == "p3":
            report["groups"] = group_reports(examples, scores, calibration, dimensions, include_bona_reference=True,
                                               bootstrap_resamples=args.bootstrap_resamples)
            if args.dataset in ("asv2019", "asv5"):
                report["attack_novelty"] = {"target_dataset_present_in_training": False,
                    "attack_ids_are_dataset_namespaced": True,
                    "generator_family_novelty": "not inferred; requires verified cross-dataset family mapping"}
        elif args.protocol == "p4":
            report["groups"] = group_reports(examples, scores, calibration, ("codec",), bootstrap_resamples=args.bootstrap_resamples)
            report["paired_clean_deltas"] = paired_raw_deltas(examples, scores, calibration, dimension="codec", reference="C00")
        report["normalization_population"] = "complete target partition; never recomputed per generator/attack/codec subgroup"
    write_json_once(directory / "metrics.json", report)
    print(f"Wrote {args.protocol.upper()} report: {directory / 'metrics.json'}")


if __name__ == "__main__":
    main()
