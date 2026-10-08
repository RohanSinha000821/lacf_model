import copy
import json
import sys

import pytest

from audio_deepfake_detection import protocol
from audio_deepfake_detection.metrics import write_score_csv
from audio_deepfake_detection.protocol import (
    FINAL_SEEDS, FOLDS, canonical_score_labels, checkpoint_digest, load_score_manifest,
    save_score_manifest, verify_score_digest,
)
from scripts.evaluate_scores import evaluate_run
from scripts.summarize_results import summarize
from scripts.wavlm import evaluate as wavlm


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "outputs" / "wavlm" / "f1" / "1234"
    run_dir.mkdir(parents=True)
    source_root, target_root = tmp_path / "sources", tmp_path / "target"
    opened = []

    def reader(domain, split, data_root):
        opened.append((domain, split))
        root = protocol.dataset_root(data_root, domain)
        return [(str(root / f"{split}/{index}.wav"), label) for index, label in enumerate([0, 0, 1, 1])]

    monkeypatch.setattr(protocol, "read_dataset", reader)
    monkeypatch.setattr(wavlm, "read_dataset", reader)
    best = run_dir / "best.pt"
    best.write_bytes(b"synthetic checkpoint; never loaded for inference")
    completion = {"fold": "f1", "seed": 1234, "best_epoch": 2, "final_epoch": 7, "best_macro_source_dev_eer": 0.0}
    (run_dir / "training_complete.json").write_text(json.dumps(completion))
    manifest = load_score_manifest(run_dir, "f1", 1234, checkpoint_digest(best), create=True)
    for name, membership in manifest["splits"].items():
        domain, split = membership["dataset"], membership["split"]
        scores = [-2, -1, 1, 2] if split == "dev" else [100, 101, 102, 103]
        rows = [(f"{split}/{i}.wav", domain, label, score)
                for i, (label, score) in enumerate(zip([0, 0, 1, 1], scores))]
        write_score_csv(run_dir / name, rows)
        manifest["score_sha256"][name] = checkpoint_digest(run_dir / name)
    save_score_manifest(run_dir, manifest)
    return run_dir, source_root, target_root, opened, completion


def evaluate_synthetic(run):
    directory, source, target, _, _ = run
    return evaluate_run(directory, "f1", seed=1234, source_data_root=source, target_data_root=target, bootstrap_resamples=8)


def write_summaries(tmp_path, template):
    root = tmp_path / "summaries"
    for fold_name, fold in FOLDS.items():
        for index, seed in enumerate(FINAL_SEEDS):
            directory = root / "wavlm" / fold_name / str(seed)
            directory.mkdir(parents=True)
            report = copy.deepcopy(template)
            report.update({"fold": fold_name, "target_domain": fold["target"], "target_split": fold["target_split"],
                           "sources": list(fold["sources"]), "seed": seed})
            report["target"]["eer"] = 0.1 * (index + 1)
            (directory / "metrics.json").write_text(json.dumps(report))
    return root


def test_generic_score_to_report_and_four_fold_summary(synthetic_run, tmp_path):
    report = evaluate_synthetic(synthetic_run)
    directory, _, _, opened, _ = synthetic_run
    assert (directory / "metrics.json").is_file()
    assert opened == [("asv2019", "dev"), ("asv5", "dev"), ("cfad", "dev"), ("speechfake", "test")]
    assert report["seed"] == 1234
    assert report["model"] == "wavlm"
    assert report["target"]["eer"] == 0.0
    assert report["normalization"]["target_unlabeled"]["mean"] == 101.5
    assert report["score_sha256"]["target_scores.csv"] == checkpoint_digest(directory / "target_scores.csv")
    root = write_summaries(tmp_path, report)
    summary = summarize(root, "wavlm")
    assert summary["per_fold"]["f1"]["target.eer"] == pytest.approx({"mean": 0.2, "sample_std": 0.1})
    assert summary["equal_weight_four_fold_mean"]["target.auroc"] == 1.0
    assert summary["target_datasets_pooled"] is False
    assert summary["publication_summary"] is True
    with pytest.raises(FileExistsError):
        evaluate_synthetic(synthetic_run)


@pytest.mark.parametrize("case", ["missing_fold", "missing_seed", "no_seed", "wrong_seed", "wrong_target",
                                  "wrong_split", "wrong_sources", "no_ci", "missing_metric", "extra_metric",
                                  "missing_ci_metric", "nonfinite", "wrong_ci_assumptions"])
def test_summary_rejects_incomplete_or_mismatched_runs(synthetic_run, tmp_path, case):
    root = write_summaries(tmp_path, evaluate_synthetic(synthetic_run))
    path = root / "wavlm" / ("f4" if case == "missing_fold" else "f1") / "3456" / "metrics.json"
    if case in ("missing_fold", "missing_seed"):
        path.unlink()
    else:
        report = json.loads(path.read_text())
        if case == "no_seed":
            del report["seed"]
        elif case == "wrong_seed":
            report["seed"] = 1234
        elif case == "wrong_target":
            report["target_domain"] = "asv5"
        elif case == "wrong_split":
            report["target_split"] = "dev"
        elif case == "wrong_sources":
            report["sources"] = ["speechfake", "asv5", "cfad"]
        elif case == "no_ci":
            del report["confidence_intervals"]
        elif case == "missing_metric":
            del report["strict_raw_transfer"]["target_operating_points"]["1%"]
        elif case == "extra_metric":
            report["strict_raw_transfer"]["target_operating_points"]["1%"]["accuracy"] = 1
        elif case == "missing_ci_metric":
            del report["confidence_intervals"]["intervals"]["eer"]
        elif case == "nonfinite":
            report["target"]["eer"] = float("nan")
        elif case == "wrong_ci_assumptions":
            report["confidence_intervals"]["source_thresholds_fixed_within_resamples"] = False
        path.write_text(json.dumps(report))
    with pytest.raises((ValueError, FileNotFoundError)):
        summarize(root, "wavlm")


def test_summary_frozen_seeds_and_exploratory_override(synthetic_run, tmp_path):
    root = write_summaries(tmp_path, evaluate_synthetic(synthetic_run))
    with pytest.raises(ValueError, match="exactly seeds"):
        summarize(root, "wavlm", (1234, 2345))
    assert summarize(root, "wavlm", (1234, 2345), non_final=True)["publication_summary"] is False
    with pytest.raises(ValueError, match="distinct"):
        summarize(root, "wavlm", (1234, 1234), non_final=True)


@pytest.fixture
def single_seed_wavlm_reports(synthetic_run, tmp_path):
    template = evaluate_synthetic(synthetic_run)
    root = tmp_path / "single_seed_summaries"
    for fold_name, fold in FOLDS.items():
        directory = root / "wavlm_bs96" / fold_name / "1234"
        directory.mkdir(parents=True)
        report = copy.deepcopy(template)
        report.update({"model": "wavlm_bs96", "fold": fold_name, "seed": 1234,
                       "sources": list(fold["sources"]), "target_domain": fold["target"],
                       "target_split": fold["target_split"]})
        (directory / "metrics.json").write_text(json.dumps(report))
    return root


def test_wavlm_single_seed_summary_has_no_invented_standard_deviation(single_seed_wavlm_reports):
    summary = summarize(single_seed_wavlm_reports, "wavlm_bs96")
    assert summary["seeds"] == [1234] and summary["training_seed_count"] == 1
    assert summary["across_seed_variability_estimated"] is False
    assert summary["seed_policy"] == "single_seed_compute_budget"
    assert summary["publication_summary"] is True
    assert summary["equal_weight_four_fold_mean"]["target.auroc"] == 1.0
    for fold in summary["per_fold"].values():
        assert fold["target.eer"] == {"mean": 0.0, "sample_std": None}
        assert all(metric["sample_std"] is None for metric in fold.values())
    json.dumps(summary, allow_nan=False)


def test_single_seed_cli_defaults_follow_model_policy(single_seed_wavlm_reports, monkeypatch):
    from scripts import summarize_results

    monkeypatch.setattr(sys, "argv", ["summarize_results.py", "--model", "wavlm_bs96",
                                     "--root", str(single_seed_wavlm_reports)])
    summarize_results.main()
    summary = json.loads((single_seed_wavlm_reports / "wavlm_bs96/summary.json").read_text())
    assert summary["seeds"] == [1234] and summary["per_fold"]["f1"]["target.eer"]["sample_std"] is None
    with pytest.raises(FileExistsError):
        summarize_results.main()


def test_single_seed_still_requires_four_folds_and_bootstrap(single_seed_wavlm_reports):
    root = single_seed_wavlm_reports
    path = root / "wavlm_bs96/f4/1234/metrics.json"
    report = json.loads(path.read_text())
    path.unlink()
    with pytest.raises(FileNotFoundError):
        summarize(root, "wavlm_bs96")
    del report["confidence_intervals"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="confidence intervals"):
        summarize(root, "wavlm_bs96")


def test_seed_exception_does_not_relax_other_model_final_summaries(single_seed_wavlm_reports):
    from audio_deepfake_detection.protocol import final_seeds_for

    assert final_seeds_for("aasist") == (1234,)
    assert final_seeds_for("lacf") == (1234,)
    assert final_seeds_for("ssl_gating") == FINAL_SEEDS
    with pytest.raises(ValueError, match="exactly seeds"):
        summarize(single_seed_wavlm_reports, "ssl_gating", (1234,))
    with pytest.raises(ValueError, match="exactly seeds"):
        summarize(single_seed_wavlm_reports, "lacf", FINAL_SEEDS)
    with pytest.raises(ValueError, match="exactly seeds"):
        summarize(single_seed_wavlm_reports, "aasist", FINAL_SEEDS)
    with pytest.raises(ValueError, match="exactly seeds"):
        summarize(single_seed_wavlm_reports, "wavlm_bs96", FINAL_SEEDS)


@pytest.mark.parametrize("seeds", [(), (1234, 1234), (-1,), (True,)])
def test_summary_rejects_invalid_seed_lists(single_seed_wavlm_reports, seeds):
    with pytest.raises(ValueError, match="distinct"):
        summarize(single_seed_wavlm_reports, "wavlm_bs96", seeds, non_final=True)


@pytest.mark.parametrize("case", ["missing_completion", "missing_checkpoint", "wrong_fold", "wrong_seed",
                                  "nan_macro", "wrong_macro", "no_manifest", "changed_checkpoint",
                                  "changed_scores", "wrong_manifest_split", "wrong_ids", "wrong_label", "wrong_split"])
def test_generic_guards_before_target_access(synthetic_run, case):
    directory, _, _, opened, completion = synthetic_run
    completion_path = directory / "training_complete.json"
    manifest_path = directory / "score_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if case == "missing_completion":
        completion_path.unlink()
    elif case == "missing_checkpoint":
        (directory / "best.pt").unlink()
    elif case in ("wrong_fold", "wrong_seed", "nan_macro", "wrong_macro"):
        completion[{"wrong_fold": "fold", "wrong_seed": "seed", "nan_macro": "best_macro_source_dev_eer",
                    "wrong_macro": "best_macro_source_dev_eer"}[case]] = {
            "wrong_fold": "f2", "wrong_seed": 2345, "nan_macro": float("nan"), "wrong_macro": 0.1,
        }[case]
        completion_path.write_text(json.dumps(completion))
        manifest["training_completion_sha256"] = checkpoint_digest(completion_path)
        manifest_path.write_text(json.dumps(manifest))
    elif case == "no_manifest":
        manifest_path.unlink()
    elif case == "changed_checkpoint":
        (directory / "best.pt").write_bytes(b"changed")
    elif case == "changed_scores":
        with (directory / "source_dev_asv2019.csv").open("a") as handle:
            handle.write("unexpected,asv2019,0,0\n")
    elif case == "wrong_manifest_split":
        manifest["splits"]["source_dev_asv2019.csv"]["split"] = "train"
        manifest_path.write_text(json.dumps(manifest))
    else:
        path = directory / "source_dev_asv2019.csv"
        text = path.read_text()
        if case == "wrong_ids":
            text = text.replace("dev/0.wav", "dev/unexpected.wav")
        elif case == "wrong_label":
            text = text.replace("dev/0.wav,asv2019,0", "dev/0.wav,asv2019,1")
        elif case == "wrong_split":
            text = text.replace("dev/", "train/")
        path.write_text(text)
        manifest["score_sha256"][path.name] = checkpoint_digest(path)
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate_synthetic(synthetic_run)
    assert all(domain != "speechfake" for domain, _ in opened)


def test_wavlm_existing_score_validation(synthetic_run):
    directory, source, _, _, _ = synthetic_run
    manifest = json.loads((directory / "score_manifest.json").read_text())
    path = directory / "source_dev_asv2019.csv"
    labels, scores = wavlm.score_split(None, "asv2019", "dev", source, path, None,
                                     num_workers=0, prefetch_factor=2, manifest=manifest)
    assert labels.tolist() == [0, 0, 1, 1]
    assert scores.tolist() == [-2, -1, 1, 2]
    path.write_text(path.read_text().replace("dev/0.wav", "train/0.wav"))
    manifest["score_sha256"][path.name] = checkpoint_digest(path)
    with pytest.raises(ValueError, match="Unexpected"):
        wavlm.score_split(None, "asv2019", "dev", source, path, None,
                          num_workers=0, prefetch_factor=2, manifest=manifest)


def test_wavlm_recovers_legacy_prefix_then_registers_full_export(synthetic_run, monkeypatch):
    directory, _, target, _, _ = synthetic_run
    manifest = json.loads((directory / "score_manifest.json").read_text())
    path = directory / "target_scores.csv"
    content = path.read_text().splitlines(keepends=True)
    path.unlink()
    path.with_name(path.name + ".tmp").write_text("".join(content[:3]))
    del manifest["score_sha256"][path.name]
    save_score_manifest(directory, manifest)
    visited = []

    def loader(dataset, **kwargs):
        visited.extend(dataset.records)
        assert kwargs["batch_size"] == 1
        return object()

    def rows(model, loader, records, domain, root, device, *, start_offset):
        assert start_offset == 2
        for index, (record, label) in enumerate(records, start_offset):
            yield protocol.utterance_id(record, domain, root), domain, label, 100 + index

    monkeypatch.setattr(wavlm, "make_wavlm_eval_loader", loader)
    monkeypatch.setattr(wavlm, "score_rows", rows)
    labels, scores = wavlm.score_split(None, "speechfake", "test", target, path, None,
                                     num_workers=0, prefetch_factor=2, manifest=manifest,
                                     adopt_legacy_partial=True)
    assert len(visited) == 2
    assert labels.tolist() == [0, 0, 1, 1]
    assert scores.tolist() == [100, 101, 102, 103]
    verify_score_digest(path, manifest)
    assert path.read_text() == "".join(content)


@pytest.mark.parametrize("case", ["fold", "seed", "config_fold", "epoch", "sources", "target", "macro"])
def test_wavlm_checkpoint_guards(synthetic_run, case):
    _, _, _, _, completion = synthetic_run
    checkpoint = {"fold": "f1", "epoch": 2, "macro_dev_eer": 0.0,
                  "config": {"fold": "f1", "seed": 1234, "sources": FOLDS["f1"]["sources"], "target": "speechfake"}}
    wavlm.verify_checkpoint(checkpoint, completion, "f1", 1234)
    if case in ("fold", "epoch"):
        checkpoint[case] = "f2" if case == "fold" else 3
    elif case == "macro":
        checkpoint["macro_dev_eer"] = float("nan")
    else:
        key = "fold" if case == "config_fold" else case
        checkpoint["config"][key] = {"config_fold": "f2", "seed": 2345, "sources": ("speechfake",), "target": "asv5"}[case]
    with pytest.raises(ValueError):
        wavlm.verify_checkpoint(checkpoint, completion, "f1", 1234)


def test_canonical_records_reject_duplicate_ids(tmp_path):
    root = protocol.dataset_root(tmp_path, "asv2019")
    with pytest.raises(ValueError, match="Invalid canonical"):
        canonical_score_labels("asv2019", "dev", tmp_path, records=[(str(root / "a.wav"), 0)] * 2)


def test_wavlm_batch_family_checkpoint_identity(synthetic_run):
    completion = synthetic_run[4]
    checkpoint = {"fold": "f1", "epoch": 2, "macro_dev_eer": 0.0,
                  "config": {"fold": "f1", "seed": 1234, "sources": FOLDS["f1"]["sources"],
                             "target": "speechfake", "run_name": "wavlm_bs96"}}
    wavlm.verify_checkpoint(checkpoint, completion, "f1", 1234, "wavlm_bs96")
    with pytest.raises(ValueError, match="output family"):
        wavlm.verify_checkpoint(checkpoint, completion, "f1", 1234, "wavlm")


def test_manifest_refuses_legacy_or_unregistered_scores(synthetic_run):
    directory, _, _, _, _ = synthetic_run
    manifest = json.loads((directory / "score_manifest.json").read_text())
    manifest["score_sha256"] = {}
    with pytest.raises(ValueError, match="unregistered"):
        verify_score_digest(directory / "target_scores.csv", manifest)
    (directory / "score_manifest.json").write_text(json.dumps({"fold": "f1", "seed": 1234,
                                                              "checkpoint_sha256": checkpoint_digest(directory / "best.pt")}))
    with pytest.raises(ValueError, match="manifest"):
        load_score_manifest(directory, "f1", 1234, checkpoint_digest(directory / "best.pt"))


@pytest.mark.parametrize("case", ["valid", "independent", "no_completion", "no_checkpoint", "checkpoint_epoch", "checkpoint_seed",
                                  "changed_checkpoint", "no_manifest", "source_macro"])
def test_wavlm_main_target_gate_without_gpu_inference(synthetic_run, monkeypatch, case):
    directory, _, _, _, _ = synthetic_run
    visited = []
    checkpoint = {
        "fold": "f1", "epoch": 2, "macro_dev_eer": 0.0, "model_state_dict": {},
        "config": {"fold": "f1", "seed": 1234, "sources": FOLDS["f1"]["sources"],
                   "target": "speechfake", "model_name": "synthetic"},
    }
    if case == "no_completion":
        (directory / "training_complete.json").unlink()
    elif case == "no_checkpoint":
        (directory / "best.pt").unlink()
    elif case == "checkpoint_epoch":
        checkpoint["epoch"] = 3
    elif case == "checkpoint_seed":
        checkpoint["config"]["seed"] = 2345
    elif case == "changed_checkpoint":
        (directory / "best.pt").write_bytes(b"changed")
    elif case == "no_manifest":
        (directory / "score_manifest.json").unlink()

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state):
            pass

    def split(model, domain, split_name, *args, **kwargs):
        visited.append(domain)
        return [0, 0, 1, 1], ([2, 1, -1, -2] if case == "source_macro" else [-2, -1, 1, 2])

    monkeypatch.setenv("PROJECT_ROOT", str(directory.parents[3]))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--fold", "f1", "--seed", "1234", "--bootstrap-resamples", "5", "--confirm-run-frozen" if case == "independent" else "--confirm-protocol-frozen"])
    monkeypatch.setattr(wavlm.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(wavlm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wavlm, "WavLMWA", lambda **kwargs: FakeModel())
    monkeypatch.setattr(wavlm, "score_split", split)
    if case in ("valid", "independent"):
        wavlm.main()
        assert visited == ["asv2019", "asv5", "cfad", "speechfake"]
        report = json.loads((directory / "metrics.json").read_text())
        assert report["checkpoint_epoch"] == 2
        assert report["seed"] == 1234
        assert report["comparison_wide_freeze_confirmed_by_operator"] is (case != "independent")
        assert report["completed_run_freeze_confirmed_by_operator"] is True
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            wavlm.main()
        assert "speechfake" not in visited


@pytest.mark.parametrize("family", ["aasist", "lacf"])
def test_single_seed_final_summary_and_cli(single_seed_wavlm_reports, monkeypatch, family):
    from scripts import summarize_results

    root = single_seed_wavlm_reports
    for fold in FOLDS:
        report = json.loads((root / "wavlm_bs96" / fold / "1234/metrics.json").read_text())
        report["model"] = family
        directory = root / family / fold / "1234"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["summarize_results.py", "--model", family, "--root", str(root)])
    summarize_results.main()
    summary = json.loads((root / family / "summary.json").read_text())
    assert summary["seeds"] == [1234] and summary["training_seed_count"] == 1
    assert summary["publication_summary"] is True and summary["four_fold_complete"] is True
    assert summary["seed_policy"] == "single_seed_compute_budget"
    assert summary["across_seed_variability_estimated"] is False
    assert all(metric["sample_std"] is None for fold in summary["per_fold"].values() for metric in fold.values())
    with pytest.raises(FileExistsError):
        summarize_results.main()
    (root / family / "f4/1234/metrics.json").unlink()  # Disposable synthetic fixture only.
    with pytest.raises(FileNotFoundError):
        summarize(root, family)
