import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def fake_evaluation_queue(tmp_path):
    script = tmp_path / "scripts/wavlm/evaluate_folds.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / "scripts/wavlm/evaluate_folds.sh", script)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (tmp_path / "activate.sh").write_text(f'export PATH="{binaries}:$PATH"\n')
    stub = binaries / "python"
    stub.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
root = Path(os.environ['FAKE_PROJECT'])
args = sys.argv[1:]
with (root / 'calls.jsonl').open('a') as handle:
    handle.write(json.dumps(args) + '\\n')
if '--fold' in args:
    fold = args[args.index('--fold') + 1]
    if fold == os.environ.get('FAKE_FAIL_FOLD'):
        sys.exit(7)
    (root / 'outputs/wavlm_bs96' / fold / '1234/metrics.json').write_text('{}')
''')
    stub.chmod(0o755)
    for fold in ("f1", "f2", "f3", "f4"):
        directory = tmp_path / "outputs/wavlm_bs96" / fold / "1234"
        directory.mkdir(parents=True)
        (directory / "training_complete.json").write_text("{}")
    return tmp_path, script, {**os.environ, "FAKE_PROJECT": str(tmp_path)}


def calls(root):
    path = root / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_queue_sequential_three_folds_and_partial_summary(fake_evaluation_queue):
    root, script, env = fake_evaluation_queue
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    invoked = calls(root)
    assert [a[a.index("--fold") + 1] for a in invoked[:-1]] == ["f1", "f2", "f3"]
    for args in invoked[:-1]:
        assert "--confirm-run-frozen" in args and "--confirm-protocol-frozen" not in args
        assert args[args.index("--num-workers") + 1] == "8"
        assert args[args.index("--prefetch-factor") + 1] == "2"
        assert args[args.index("--bootstrap-resamples") + 1] == "1000"
        assert "--gpu-memory-limit-gib" not in args
    assert "--non-final" in invoked[-1]
    assert invoked[-1][-3:] == ["f1", "f2", "f3"]


def test_f4_addition_does_not_rerun_previous_folds(fake_evaluation_queue):
    root, script, env = fake_evaluation_queue
    for fold in ("f1", "f2", "f3"):
        (root / "outputs/wavlm_bs96" / fold / "1234/metrics.json").write_text("preserve")
    result = subprocess.run(["bash", str(script), "f4"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    invoked = calls(root)
    assert len(invoked) == 2 and invoked[0][invoked[0].index("--fold") + 1] == "f4"
    assert "--non-final" not in invoked[-1]
    assert all((root / "outputs/wavlm_bs96" / f / "1234/metrics.json").read_text() == "preserve" for f in ("f1", "f2", "f3"))


def test_failure_stops_subsequent_folds(fake_evaluation_queue):
    root, script, env = fake_evaluation_queue
    result = subprocess.run(["bash", str(script)], env={**env, "FAKE_FAIL_FOLD": "f2"}, capture_output=True, text=True)
    assert result.returncode == 7
    assert len(calls(root)) == 2
    assert not (root / "outputs/wavlm_bs96/f3/1234/metrics.json").exists()


def test_explicit_legacy_adoption_is_limited_to_named_fold(fake_evaluation_queue):
    root, script, env = fake_evaluation_queue
    result = subprocess.run(["bash", str(script)], env={**env, "WAVLM_ADOPT_LEGACY_PARTIAL_FOLD": "f1"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    invoked = calls(root)
    assert "--adopt-legacy-partial" in invoked[0]
    assert all("--adopt-legacy-partial" not in args for args in invoked[1:])


def test_lock_prevents_second_evaluation_queue(fake_evaluation_queue):
    root, script, env = fake_evaluation_queue
    with (root / "outputs/wavlm_bs96/.evaluation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
        assert result.returncode != 0
        assert calls(root) == []
