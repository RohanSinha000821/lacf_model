import json
import sys
from pathlib import Path

import pytest

from audio_deepfake_detection.analysis import evaluate_frozen, group_reports, paired_raw_deltas
from audio_deepfake_detection.evaluation_data import (
    EvaluationExample as Example, asv2021_df, cfad_conditions, cfad_pair_id,
    mlaad_mailabs, native_examples, partialspoof, relative_path, validate_examples,
)
from audio_deepfake_detection.metrics import calibrate_source_scores, evaluate_transfer, write_score_csv
from audio_deepfake_detection import protocol
from scripts import evaluate_extended as extended


@pytest.fixture
def calibration():
    return calibrate_source_scores({"a": ([0, 0, 1, 1], [-2, -1, 1, 2]),
                                    "b": ([0, 0, 1, 1], [-4, -2, 2, 4])})


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_frozen_evaluator_matches_main_contract(calibration):
    source = {"a": ([0, 0, 1, 1], [-2, -1, 1, 2]), "b": ([0, 0, 1, 1], [-4, -2, 2, 4])}
    original = evaluate_transfer(source, [0, 0, 1, 1], [10, 11, 12, 13], bootstrap_resamples=7)
    extended_report = evaluate_frozen(calibration, [0, 0, 1, 1], [10, 11, 12, 13], bootstrap_resamples=7)
    for key in ("target", "strict_raw_transfer", "unlabeled_zscore_transfer", "confidence_intervals"):
        assert extended_report[key] == original[key]


def test_subgroups_freeze_stats_and_share_bona_reference(calibration):
    examples = [Example(str(i), str(i), label, {"generator": group}) for i, (label, group) in enumerate(
        [(0, "bonafide"), (0, "bonafide"), (1, "TTS"), (1, "VC"), (1, "NV")])]
    reports = group_reports(examples, [0, 1, 2, 50, 100], calibration, ("generator",),
                           include_bona_reference=True, bootstrap_resamples=3)["generator"]
    assert set(reports) == {"TTS", "VC", "NV"}
    for report in reports.values():
        assert report["target"]["n_bona_fide"] == 2
        assert report["target"]["n_spoof"] == 1
        assert report["normalization"]["target_unlabeled"]["mean"] == 30.6
        assert report["strict_raw_transfer"]["thresholds"] == calibration["raw_thresholds"]


@pytest.mark.parametrize("labels", [[1, 1], [0, 0]])
def test_single_class_no_invented_metrics(calibration, labels):
    report = evaluate_frozen(calibration, labels, [0, 2], bootstrap_resamples=5)
    assert report["target"]["eer"] is None and report["target"]["auroc"] is None
    point = report["strict_raw_transfer"]["target_operating_points"]["1%"]
    assert point["acer"] is None
    assert point["bpcer" if labels[0] else "apcer"] is None
    assert len(report["confidence_intervals"]["intervals"]) == 6


def test_degenerate_subgroup_allowed_with_nondegenerate_parent(calibration):
    report = evaluate_frozen(calibration, [1, 1], [3, 3], target_stats={"mean": 1, "std": 2}, bootstrap_resamples=0)
    assert report["target"]["n_spoof"] == 2
    with pytest.raises(ValueError):
        evaluate_frozen(calibration, [1, 1], [3, 3])


@pytest.mark.parametrize("value", ["../x.wav", "/etc/passwd", "", "a/../../x"])
def test_metadata_paths_cannot_escape(value):
    with pytest.raises(ValueError):
        relative_path(value)


def test_duplicate_metadata_rejected():
    with pytest.raises(ValueError):
        validate_examples([Example("same", "a", 0), Example("same", "b", 1)])


def test_df_official_eval_subset_and_missing_keys(tmp_path):
    with pytest.raises(FileNotFoundError, match="Official DF keys missing"):
        asv2021_df(tmp_path)
    keys = tmp_path / "keys.txt"
    write(keys, "s DF_E_1 nocodec asvspoof A14 spoof notrim eval traditional_vocoder - - - -\n"
                "s DF_E_2 nocodec asvspoof - bonafide notrim progress bonafide - - - -\n"
                "s DF_E_3 nocodec asvspoof - bonafide trim hidden bonafide - - - -\n"
                "s DF_E_4 nocodec asvspoof - bonafide notrim eval bonafide - - - -\n")
    examples = asv2021_df(tmp_path, keys=keys, audio_root=tmp_path / "audio")
    assert [x.identifier for x in examples] == ["DF_E_1.flac", "DF_E_4.flac"]
    assert [x.label for x in examples] == [1, 0]
    write(keys, "DF_E_1\n")
    with pytest.raises(ValueError, match="expected 13"):
        asv2021_df(tmp_path, keys=keys)


def test_partialspoof_native_protocol(tmp_path):
    keys = tmp_path / "partialspoof/extracted/protocols/database/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.eval.trl.txt"
    write(keys, "s CON_E_1 - CON spoof\ns CON_E_2 - - bonafide\n")
    examples = partialspoof(tmp_path)
    assert examples[0].path.endswith("eval/database/eval/con_wav/CON_E_1.wav")
    assert [x.label for x in examples] == [1, 0]


def test_speechfake_metadata_uses_verified_generators(tmp_path):
    path = tmp_path / "speechfake/extracted/metadata/experiments/baseline/test_all.csv"
    write(path, "file,label,generator,model,speaker,language\nReal/a.wav,bonafide,-,-,a,en\nFake/b.wav,spoof,NV,HifiGAN,b,en\n")
    examples = native_examples("speechfake", "test", tmp_path)
    assert examples[1].groups["generator"] == "NV"
    write(path, path.read_text().replace(",NV,", ",guessed,"))
    with pytest.raises(ValueError, match="Unknown SpeechFake generator"):
        native_examples("speechfake", "test", tmp_path)


def test_asv5_codec_and_original_id(tmp_path):
    path = tmp_path / "asvspoof5/extracted/protocols/ASVspoof5.eval.track_1.tsv"
    write(path, "s E_1 M C05 2 E_2 AC1 A26 spoof -\ns E_2 M - 0 - - bonafide bonafide -\n")
    examples = native_examples("asv5", "eval", tmp_path)
    assert [x.groups["codec"] for x in examples] == ["C05", "C00"]
    assert [x.pair_id for x in examples] == ["E_2", "E_2"]


def test_asv2019_attack_metadata(tmp_path):
    path = tmp_path / "asvspoof2019/extracted/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    write(path, "s LA_E_1 - A07 spoof\ns LA_E_2 - - bonafide\n")
    examples = native_examples("asv2019", "eval", tmp_path)
    assert examples[0].groups == {"attack": "A07"}


def test_cfad_conditions_native_naming_and_pair_alignment(tmp_path):
    root = tmp_path / "cfad/extracted"
    for folder, suffix, name in (("clean_version", "clean", "clip.wav"),
                                 ("noisy_version", "noise", "clip_airport_snr10.wav"),
                                 ("codec_version", "codec", "clip_aac.wav")):
        for prefix in ("real", "fake"):
            write(root / folder / f"test_unseen_{suffix}" / f"{prefix}_{suffix}" / "source" / name, "")
    examples = cfad_conditions(tmp_path)
    assert len(examples) == 6
    assert len({x.pair_id for x in examples}) == 2
    assert {x.groups.get("snr") for x in examples} == {None, "10"}
    with pytest.raises(ValueError):
        cfad_conditions(tmp_path, "train")
    with pytest.raises(ValueError):
        cfad_pair_id(Path("noisy_version/test_unseen_noise/real_noise/unknown.wav"), "noise")


def test_mlaad_requires_correct_genuine_and_deduplicates(tmp_path):
    with pytest.raises(FileNotFoundError, match="genuine M-AILABS"):
        mlaad_mailabs(tmp_path, None)
    real = tmp_path / "mailabs"
    write(real / "en/a.wav", "")
    metadata = tmp_path / "mlaad/fake/en/model/meta.csv"
    write(metadata, "path|original_file|language|model_name|architecture\n"
                    "./fake/en/model/a.wav|en/a.wav|en|m|arch\n"
                    "./fake/en/model/b.wav|en/a.wav|en|m|arch\n")
    examples = mlaad_mailabs(tmp_path, real)
    assert len(examples) == 3
    assert sum(x.label == 0 for x in examples) == 1
    write(metadata, metadata.read_text().replace("en/a.wav|en", "en/missing.wav|en"))
    with pytest.raises(FileNotFoundError, match="referenced M-AILABS"):
        mlaad_mailabs(tmp_path, real)


def test_mlaad_cross_language_references_do_not_relabel_genuine(tmp_path):
    real = tmp_path / "mailabs"
    write(real / "en_US/a.wav", "")
    for language in ("en", "de"):
        write(tmp_path / f"mlaad/fake/{language}/model/meta.csv",
              "path|original_file|language|model_name|architecture\n"
              f"fake/{language}/model/a.wav|en_US/a.wav|{language}|m|arch\n")
    examples = mlaad_mailabs(tmp_path, real)
    genuine = [item for item in examples if not item.label]
    assert len(genuine) == 1 and genuine[0].groups == {"source_locale": "en_US"}


def test_paired_robustness_equal_original_weight(calibration):
    examples = [Example(str(i), str(i), 1, {"codec": c}, p) for i, (c, p) in enumerate(
        [("C00", "a"), ("C05", "a"), ("C05", "a"), ("C00", "b"), ("C05", "b"), ("C05", "unmatched")])]
    report = paired_raw_deltas(examples, [2, -1, 2, 2, -1, -1], calibration, dimension="codec", reference="C00")["C05"]
    assert report["n_matched_originals"] == 2
    assert report["strict_raw_operating_point_deltas"]["1%"]["delta_apcer"] == 0.75


def test_paired_label_conflict_rejected(calibration):
    examples = [Example("a", "a", 0, {"codec": "C00"}, "p"), Example("b", "b", 1, {"codec": "C05"}, "p")]
    with pytest.raises(ValueError, match="label conflict"):
        paired_raw_deltas(examples, [1, 2], calibration, dimension="codec", reference="C00")


@pytest.mark.parametrize("extra", [[], ["--confirm-protocol-frozen"]])
def test_p2_requires_freeze_and_single_selection(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["evaluate_extended.py", "--protocol", "p2", "--dataset", "asv2021_df", "--run-dir", "outputs/wavlm/f1/1234", *extra])
    with pytest.raises(SystemExit):
        extended.parse_args()


def test_selection_record_binding():
    identity = {"model": "wavlm_bs96", "fold": "f1", "seed": 1234, "checkpoint_sha256": "abc"}
    record = {**identity, "target_results_used": False, "source_only_selection_reason": "Prespecified source-only choice"}
    extended.verify_selection(record, identity)
    for key, value in (("checkpoint_sha256", "changed"), ("target_results_used", True), ("source_only_selection_reason", "")):
        with pytest.raises(ValueError):
            extended.verify_selection({**record, key: value}, identity)


@pytest.fixture
def registered_run(tmp_path, monkeypatch):
    directory = tmp_path / "outputs/wavlm_bs96/f1/1234"
    write(directory / "best.pt", "synthetic: do not deserialize")
    write(directory / "training_complete.json", json.dumps({"fold": "f1", "seed": 1234, "best_epoch": 1,
          "final_epoch": 2, "best_macro_source_dev_eer": 0.0}))
    def reader(dataset, split, root):
        return [(str(protocol.dataset_root(root, dataset) / f"{i}.wav"), label) for i, label in enumerate([0, 0, 1, 1])]
    monkeypatch.setattr(protocol, "read_dataset", reader)
    manifest = protocol.load_score_manifest(directory, "f1", 1234, protocol.checkpoint_digest(directory / "best.pt"), create=True)
    for name, membership in manifest["splits"].items():
        write_score_csv(directory / name, [(f"{i}.wav", membership["dataset"], label, value)
                         for i, (label, value) in enumerate(zip([0, 0, 1, 1], [-2, -1, 1, 2]))])
        manifest["score_sha256"][name] = protocol.checkpoint_digest(directory / name)
    protocol.save_score_manifest(directory, manifest)
    return directory


def test_p3_cpu_report_reuses_p1_and_preserves_outputs(registered_run, monkeypatch):
    examples = [Example(f"{i}.wav", str(i), label, {"generator": "TTS" if label else "bonafide", "model": "m"})
                for i, label in enumerate([0, 0, 1, 1])]
    monkeypatch.setattr(extended, "native_examples", lambda *a: examples)
    monkeypatch.setattr(extended, "load_exporter", lambda *a: pytest.fail("CPU reporting must not load a GPU model"))
    monkeypatch.setattr(sys, "argv", ["evaluate_extended.py", "--protocol", "p3", "--dataset", "speechfake", "--run-dir", str(registered_run), "--confirm-protocol-frozen", "--bootstrap-resamples", "3"])
    digest = protocol.checkpoint_digest(registered_run / "target_scores.csv")
    extended.main()
    report = json.loads((registered_run / "analyses/p3/speechfake/metrics.json").read_text())
    assert report["groups"]["generator"]["TTS"]["target"]["eer"] == 0.0
    assert protocol.checkpoint_digest(registered_run / "target_scores.csv") == digest
    with pytest.raises(FileExistsError):
        extended.main()


def test_source_mismatch_blocks_target_open(registered_run, monkeypatch):
    path = registered_run / "training_complete.json"
    completion = json.loads(path.read_text())
    completion["best_macro_source_dev_eer"] = .5
    path.write_text(json.dumps(completion))
    monkeypatch.setattr(extended, "native_examples", lambda *a: pytest.fail("Target must remain unopened"))
    monkeypatch.setattr(sys, "argv", ["eval", "--protocol", "p3", "--dataset", "speechfake", "--run-dir", str(registered_run), "--confirm-protocol-frozen"])
    with pytest.raises(ValueError):
        extended.main()


def test_source_only_selection_mode_never_opens_target(registered_run, monkeypatch, tmp_path):
    monkeypatch.setattr(extended, "asv2021_df", lambda *a, **k: pytest.fail("Selection must not open target"))
    path = tmp_path / "p2_selection.json"
    monkeypatch.setattr(sys, "argv", ["eval", "--protocol", "p2", "--dataset", "asv2021_df", "--run-dir", str(registered_run), "--confirm-protocol-frozen", "--write-selection-record", str(path), "--selection-reason", "Prespecified F1 deployment checkpoint"])
    extended.main()
    assert json.loads(path.read_text())["target_results_used"] is False
    with pytest.raises(FileExistsError):
        extended.main()


def test_p2_export_registers_and_rejects_tampered_scores(registered_run, monkeypatch, tmp_path):
    completion, digest = protocol.verify_completed_run(registered_run, "f1", 1234)
    selection = tmp_path / "selection.json"
    write(selection, json.dumps({**extended.selection_identity(registered_run, completion, digest),
          "target_results_used": False, "source_only_selection_reason": "Fixed source-only choice",
          "source_score_sha256": {k: v for k, v in json.loads((registered_run / "score_manifest.json").read_text())["score_sha256"].items() if k.startswith("source_dev_")}}))
    examples = [Example(str(i), str(i), label) for i, label in enumerate([0, 0, 1, 1])]
    monkeypatch.setattr(extended, "asv2021_df", lambda *a, **k: examples)
    monkeypatch.setattr(extended, "load_exporter", lambda *a: (None, None, None, 0, 2))
    monkeypatch.setattr(extended, "export_rows", lambda *a: [(x.identifier, "asv2021_df", x.label, float(i)) for i, x in enumerate(examples)])
    monkeypatch.setattr(sys, "argv", ["eval", "--protocol", "p2", "--dataset", "asv2021_df", "--run-dir", str(registered_run),
          "--selection-record", str(selection), "--confirm-protocol-frozen", "--export-scores", "--model", "wavlm", "--bootstrap-resamples", "0"])
    extended.main()
    directory = registered_run / "analyses/p2/asv2021_df"
    report = json.loads((directory / "metrics.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["score_sha256"] == report["score_sha256"]
    assert report["overall"]["target"]["auroc"] == 1
    # Only disposable synthetic fixture files are removed/altered here.
    (directory / "metrics.json").unlink()
    (directory / "scores.csv").write_text((directory / "scores.csv").read_text().replace("0.0", "99.0"))
    with pytest.raises(ValueError, match="score bytes differ"):
        extended.main()


def test_cfad_p4_separate_normalization_populations(registered_run, monkeypatch):
    # Transform the synthetic run to F2, with valid F2 source CSV identities.
    parent = registered_run.parent.parent / "f2/1234"
    parent.mkdir(parents=True)
    write(parent / "best.pt", "synthetic")
    write(parent / "training_complete.json", json.dumps({"fold": "f2", "seed": 1234, "best_epoch": 1,
           "final_epoch": 2, "best_macro_source_dev_eer": 0.0}))
    manifest = protocol.load_score_manifest(parent, "f2", 1234, protocol.checkpoint_digest(parent / "best.pt"), create=True)
    for name, membership in manifest["splits"].items():
        if not name.startswith("source_dev_"):
            continue
        write_score_csv(parent / name, [(f"{i}.wav", membership["dataset"], label, score)
                         for i, (label, score) in enumerate(zip([0, 0, 1, 1], [-2, -1, 1, 2]))])
        manifest["score_sha256"][name] = protocol.checkpoint_digest(parent / name)
    protocol.save_score_manifest(parent, manifest)
    examples, values = [], {}
    for condition, shift in (("clean", 0), ("noise", 10), ("codec", 100)):
        for label in (0, 1):
            item = Example(f"{condition}_{label}", "unused", label,
                           {"condition": condition, "codec": "aac", "noise": "airport", "snr": "10", "noise_snr": "airport_snr10"}, f"pair_{label}")
            examples.append(item)
            values[item.identifier] = shift + label
    monkeypatch.setattr(extended, "cfad_conditions", lambda *a: examples)
    monkeypatch.setattr(extended, "load_exporter", lambda *a: (None, None, None, 0, 2))
    monkeypatch.setattr(extended, "export_rows", lambda exporter, kind, items, dataset: [(x.identifier, dataset, x.label, values[x.identifier]) for x in items])
    monkeypatch.setattr(sys, "argv", ["eval", "--protocol", "p4", "--dataset", "cfad", "--run-dir", str(parent), "--confirm-protocol-frozen", "--export-scores", "--model", "aasist", "--bootstrap-resamples", "2"])
    extended.main()
    report = json.loads((parent / "analyses/p4/cfad_test_unseen/metrics.json").read_text())
    for condition, mean in (("clean", .5), ("noise", 10.5), ("codec", 100.5)):
        assert report["conditions"][condition]["normalization"]["target_unlabeled"]["mean"] == mean
    assert report["paired_clean_deltas"]["noise"]["n_matched_originals"] == 2


@pytest.mark.parametrize("kind", ["wavlm", "aasist"])
def test_native_export_preserves_order_and_orientation(monkeypatch, kind):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    class Model:
        def __call__(self, *args):
            return torch.tensor([[-1., 2.], [3., -2.]])
        def spoof_score(self, logits):
            return logits[:, 1] - logits[:, 0]
    batch = (torch.zeros(2, 3), torch.tensor([1, 0]))
    if kind == "wavlm":
        batch = (batch[0], torch.ones(2, 3), batch[1])
    module = SimpleNamespace(WavLMDataset=lambda *a, **k: None, AASISTDataset=lambda *a, **k: None,
                             make_wavlm_eval_loader=lambda *a, **k: [batch], make_eval_loader=lambda *a, **k: [batch])
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    examples = [Example("a", "unused", 1), Example("b", "unused", 0)]
    exporter = (module, Model(), torch.device("cpu"), 0, 2)
    rows = list(extended.export_rows(exporter, kind, examples, "external"))
    assert rows == [("a", "external", 1, 3.0), ("b", "external", 0, -5.0)]


def test_wavlm_p1_requires_freeze(monkeypatch):
    from scripts.wavlm import evaluate
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--fold", "f1"])
    monkeypatch.setattr(evaluate, "verify_completed_run", lambda *a: pytest.fail("Freeze must be checked before checkpoints/targets"))
    with pytest.raises(SystemExit):
        evaluate.main()


def test_p2_selection_freezes_source_score_bytes_before_target(registered_run, monkeypatch, tmp_path):
    completion, digest = protocol.verify_completed_run(registered_run, "f1", 1234)
    path = tmp_path / "selection.json"
    write(path, json.dumps({**extended.selection_identity(registered_run, completion, digest),
          "target_results_used": False, "source_only_selection_reason": "Fixed source-only choice",
          "source_score_sha256": {"source_dev_asv2019.csv": "wrong"}}))
    monkeypatch.setattr(extended, "asv2021_df", lambda *a, **k: pytest.fail("Target must remain unopened"))
    monkeypatch.setattr(sys, "argv", ["eval", "--protocol", "p2", "--dataset", "asv2021_df", "--run-dir", str(registered_run),
          "--selection-record", str(path), "--confirm-protocol-frozen"])
    with pytest.raises(ValueError, match="source calibration scores differ"):
        extended.main()
