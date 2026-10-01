import json

from audio_deepfake_detection.metrics import write_score_csv
from audio_deepfake_detection.protocol import FOLDS
from scripts.evaluate_scores import evaluate_run
from scripts.summarize_results import summarize


def test_generic_score_to_report_and_four_fold_summary(tmp_path):
    run_dir = tmp_path / "wavlm" / "f1" / "1234"
    run_dir.mkdir(parents=True)
    for domain in FOLDS["f1"]["sources"]:
        write_score_csv(
            run_dir / f"source_dev_{domain}.csv",
            [(f"{index}.wav", domain, label, score) for index, (label, score) in enumerate(
                ((0, -2.0), (0, -1.0), (1, 1.0), (1, 2.0))
            )],
        )
    write_score_csv(
        run_dir / "target_scores.csv",
        [(f"{index}.wav", "speechfake", label, score) for index, (label, score) in enumerate(
            ((0, 100.0), (0, 101.0), (1, 102.0), (1, 103.0))
        )],
    )
    report = evaluate_run(run_dir, "f1", bootstrap_resamples=5)
    assert (run_dir / "metrics.json").is_file()
    assert report["target"]["eer"] == 0.0
    assert report["normalization"]["target_unlabeled"]["mean"] == 101.5

    for fold_name, fold in FOLDS.items():
        for seed in (1234, 2345):
            directory = tmp_path / "wavlm" / fold_name / str(seed)
            directory.mkdir(parents=True, exist_ok=True)
            run_report = json.loads(json.dumps(report))
            run_report.update({"fold": fold_name, "target_domain": fold["target"], "seed": seed})
            (directory / "metrics.json").write_text(json.dumps(run_report), encoding="utf-8")
    summary = summarize(tmp_path, "wavlm", (1234, 2345))
    assert summary["per_fold"]["f1"]["target.eer"] == {"mean": 0.0, "sample_std": 0.0}
    assert summary["equal_weight_four_fold_mean"]["target.auroc"] == 1.0
    assert summary["target_datasets_pooled"] is False
