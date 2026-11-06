import csv
import json

import pytest

from audio_deepfake_detection.score_recovery import prepare_score_recovery, write_recoverable_scores


EXPECTED = [("a", 0), ("b", 1), ("c", 0)]
IDENTITY = {"checkpoint_sha256": "checkpoint-one", "batch_size": 1}
ROWS = [(identifier, "synthetic", label, float(index)) for index, (identifier, label) in enumerate(EXPECTED)]


def prepare(path, **kwargs):
    return prepare_score_recovery(path, EXPECTED, "synthetic", IDENTITY, **kwargs)


def test_failure_preserves_prefix_and_resume_scores_suffix_only(tmp_path):
    output = tmp_path / "scores.csv"

    def failing_rows():
        yield ROWS[0]
        raise RuntimeError("simulated GPU OOM")

    with pytest.raises(RuntimeError, match="simulated"):
        write_recoverable_scores(output, failing_rows(), EXPECTED, "synthetic", prepare(output))
    partial = output.with_name("scores.csv.tmp")
    prefix = partial.read_bytes()
    state = prepare(output)
    assert state["rows"] == 1
    write_recoverable_scores(output, iter(ROWS[1:]), EXPECTED, "synthetic", state, checkpoint_every=1)
    assert output.read_bytes().startswith(prefix)
    with output.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert prepare(output)["rows"] == 3  # recover publish-before-manifest-registration crash
    assert not partial.exists()


def test_legacy_requires_explicit_adoption_and_records_original_hash(tmp_path):
    output = tmp_path / "scores.csv"
    partial = output.with_name("scores.csv.tmp")
    partial.write_text("utterance_id,dataset,label,raw_score\na,synthetic,0,1.25\n")
    with pytest.raises(ValueError, match="explicit"):
        prepare(output)
    state = prepare(output, adopt_legacy=True)
    assert state["legacy_adopted_by_operator"]
    assert state["legacy_adoption_rows"] == 1
    assert state["legacy_adoption_sha256"] == state["score_sha256"]
    assert prepare(output) == state


@pytest.mark.parametrize("row", ["b,synthetic,1,2", "a,wrong,0,2", "a,synthetic,1,2",
                               "a,synthetic,0,nan", "a,synthetic,0,inf", "a,synthetic,0"])
def test_bad_legacy_prefix_is_preserved_and_rejected(tmp_path, row):
    output = tmp_path / "scores.csv"
    partial = output.with_name("scores.csv.tmp")
    partial.write_text("utterance_id,dataset,label,raw_score\n" + row + "\n")
    before = partial.read_bytes()
    with pytest.raises(ValueError):
        prepare(output, adopt_legacy=True)
    assert partial.read_bytes() == before
    assert not output.exists()


def test_changed_checkpoint_or_membership_cannot_resume(tmp_path):
    output = tmp_path / "scores.csv"
    def interrupted():
        yield ROWS[0]
        raise RuntimeError()
    with pytest.raises(RuntimeError):
        write_recoverable_scores(output, interrupted(), EXPECTED, "synthetic", prepare(output))
    for expected, identity in [(EXPECTED, {"checkpoint_sha256": "other"}),
                               (list(reversed(EXPECTED)), IDENTITY)]:
        with pytest.raises(ValueError, match="identity"):
            prepare_score_recovery(output, expected, "synthetic", identity)


def test_changed_bytes_or_uncommitted_tail_fail_closed(tmp_path):
    output = tmp_path / "scores.csv"
    write_recoverable_scores(output, iter(ROWS), EXPECTED, "synthetic", prepare(output))
    original = output.read_bytes()
    output.write_bytes(original + b"extra bytes\n")
    with pytest.raises(ValueError, match="checksum"):
        prepare(output, adopt_legacy=True)
    assert output.read_bytes() == original + b"extra bytes\n"


def test_complete_tmp_publishes_without_rerunning_model(tmp_path):
    output = tmp_path / "scores.csv"
    state = prepare(output)
    def completed_then_failed():
        yield from ROWS
        raise RuntimeError()
    with pytest.raises(RuntimeError):
        write_recoverable_scores(output, completed_then_failed(), EXPECTED, "synthetic", state)
    recovered = prepare(output)
    assert recovered["rows"] == len(EXPECTED)
    write_recoverable_scores(output, iter(()), EXPECTED, "synthetic", recovered)
    assert output.exists()


def test_truncated_final_line_rejected_even_with_valid_row(tmp_path):
    output = tmp_path / "scores.csv"
    output.with_name("scores.csv.tmp").write_text("utterance_id,dataset,label,raw_score\na,synthetic,0,2")
    with pytest.raises(ValueError, match="final line"):
        prepare(output, adopt_legacy=True)


def test_missing_score_file_with_existing_journal_is_rejected(tmp_path):
    output = tmp_path / "scores.csv"
    output.with_name("scores.csv.progress.json").write_text(json.dumps({"version": 1}))
    with pytest.raises(ValueError):
        prepare(output)
