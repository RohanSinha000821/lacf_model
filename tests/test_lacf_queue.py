"""Disposable queue fixtures only; no model loading, real data or GPU access."""

import fcntl
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def queue_project(tmp_path):
    script = tmp_path / "scripts/lacf/train_folds.sh"
    script.parent.mkdir(parents=True)
    script.write_text((Path(__file__).resolve().parents[1] / "scripts/lacf/train_folds.sh").read_text())
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (tmp_path / "activate.sh").write_text(
        f"export PROJECT_ROOT={shlex.quote(str(tmp_path))}\n"
        f"export PATH={shlex.quote(str(binaries))}:$PATH\n")
    stub = binaries / "python"
    stub.write_text('''#!/usr/bin/env python3
import hashlib, json, os, sys
from pathlib import Path
tokens = sys.argv[1:]
if tokens[0] == '-':
    os.execv(sys.executable, [sys.executable, *tokens])  # Actual completion verifier, CPU only.
fold = tokens[tokens.index('--fold') + 1]
with open(os.environ['LACF_TEST_CALLS'], 'a') as handle:
    handle.write(json.dumps(tokens) + '\\n')
if fold == os.environ.get('LACF_TEST_FAIL'):
    sys.exit(9)
sources = {'f1':['asv2019','asv5','cfad'], 'f2':['asv2019','asv5','speechfake'],
           'f3':['asv2019','cfad','speechfake'], 'f4':['asv5','cfad','speechfake']}[fold]
run = Path(os.environ['PROJECT_ROOT']) / 'outputs/lacf' / fold / '1234'
run.mkdir(parents=True)
checkpoint = run / 'best.pt'
checkpoint.write_bytes(b'disposable synthetic checkpoint')
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
config = {'run_name':'lacf', 'fold':fold, 'seed':1234, 'final_seed_plan':[1234],
          'sources':sources, 'precision':'bfloat16_mixed', 'sensitive_dtype':'float32',
          'trainable_parameters':661479, 'selected_epoch':1, 'selected_macro_source_dev_eer':.2,
          'selected_checkpoint_sha256':digest}
(run / 'config.json').write_text(json.dumps(config))
completion = {'fold':fold, 'seed':1234, 'best_epoch':1, 'final_epoch':2,
              'best_macro_source_dev_eer':.2, 'reloaded_dev_eers':dict.fromkeys(sources,.2),
              'best_checkpoint_sha256':digest, 'selected_checkpoint_reloaded_and_source_verified':True}
problem = os.environ.get('LACF_TEST_BAD_COMPLETION')
if problem == 'hash':
    completion['best_checkpoint_sha256'] = 'wrong'
if problem == 'reload':
    completion['selected_checkpoint_reloaded_and_source_verified'] = False
if problem == 'macro':
    completion['reloaded_dev_eers'][sources[0]] = .7
if problem != 'missing':
    (run / 'training_complete.json').write_text(json.dumps(completion))
''')
    stub.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    return tmp_path, script, calls, {**os.environ, "LACF_TEST_CALLS": str(calls)}


def test_queue_order_verified_completion_and_configurable_settings(queue_project):
    _, script, calls, env = queue_project
    settings = ["--batch-size", "12", "--gradient-accumulation-steps", "8", "--num-workers", "4",
                "--eval-workers", "2", "--prefetch-factor", "3", "--confirm-cache-verified"]
    result = subprocess.run(["bash", str(script), *settings], env=env, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    observed = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [row[row.index("--fold") + 1] for row in observed] == ["f1", "f2", "f3", "f4"]
    assert all(row[-len(settings):] == settings for row in observed)
    assert result.stdout.count("Verified completion:") == 4


@pytest.mark.parametrize("problem,count", [("training", 2), ("missing", 1), ("hash", 1), ("reload", 1), ("macro", 1)])
def test_queue_stops_on_training_or_completion_failure(queue_project, problem, count):
    _, script, calls, env = queue_project
    env = {**env, **({"LACF_TEST_FAIL": "f2"} if problem == "training" else {"LACF_TEST_BAD_COMPLETION": problem})}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=40)
    assert result.returncode != 0
    assert len(calls.read_text().splitlines()) == count


def test_queue_refuses_existing_work_and_concurrent_queue(queue_project):
    root, script, calls, env = queue_project
    queue = root / "outputs/lacf"
    existing = queue / "f3/1234"
    existing.mkdir(parents=True)
    (existing / "unrelated.txt").write_text("preserve")
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and not calls.exists()
    assert (existing / "unrelated.txt").read_text() == "preserve"
    (existing / "unrelated.txt").unlink()  # Only the disposable test fixture.
    existing.rmdir()
    with (queue / ".queue.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and not calls.exists()


def test_queue_rejects_fold_override(queue_project):
    _, script, calls, env = queue_project
    result = subprocess.run(["bash", str(script), "--fold", "f3"], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and not calls.exists()
