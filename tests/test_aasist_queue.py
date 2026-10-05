import fcntl
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def queue_project(tmp_path):
    script = tmp_path / "scripts/aasist/train_folds.sh"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts/aasist/train_folds.sh"
    script.write_text(source.read_text())
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (tmp_path / "activate.sh").write_text(
        f"export PROJECT_ROOT={shlex.quote(str(tmp_path))}\n"
        f"export PATH={shlex.quote(str(binaries))}:$PATH\n")
    stub = binaries / "python"
    stub.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
tokens = sys.argv[1:]
if tokens[0] == '-u':
    tokens = tokens[1:]
args = dict(zip(tokens[1::2], tokens[2::2]))
with open(os.environ['TEST_AASIST_CALLS'], 'a') as handle:
    handle.write(json.dumps(args) + '\\n')
if args['--fold'] == os.environ.get('TEST_AASIST_FAIL'):
    sys.exit(9)
if not os.environ.get('TEST_AASIST_NO_COMPLETION'):
    output = Path(os.environ['PROJECT_ROOT']) / 'outputs/aasist' / args['--fold'] / args['--seed']
    (output / 'training_complete.json').write_text('{}')
""")
    stub.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    return tmp_path, script, calls, {**os.environ, "TEST_AASIST_CALLS": str(calls)}


def test_aasist_queue_order_and_seed(queue_project):
    _, script, calls, env = queue_project
    completed = subprocess.run(["bash", str(script), "2345"], env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    observed = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [row["--fold"] for row in observed] == ["f1", "f2", "f3", "f4"]
    assert all(row["--seed"] == "2345" and row["--run-name"] == "aasist" for row in observed)
    assert "no target evaluation" in completed.stdout


@pytest.mark.parametrize("variable,value,count", [("TEST_AASIST_FAIL", "f2", 2),
                                                  ("TEST_AASIST_NO_COMPLETION", "1", 1)])
def test_queue_stops_on_failure_or_missing_completion(queue_project, variable, value, count):
    _, script, calls, env = queue_project
    completed = subprocess.run(["bash", str(script)], env={**env, variable: value},
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode != 0 and "queue stopped" in completed.stdout
    assert len(calls.read_text().splitlines()) == count


@pytest.mark.parametrize("family", ["aasist", "wavlm_bs96"])
def test_queue_does_not_compete_with_active_training(queue_project, family):
    root, script, calls, env = queue_project
    lock = root / "outputs" / family / ".queue.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode != 0 and not calls.exists()


def test_queue_rejects_invalid_seed(queue_project):
    _, script, calls, env = queue_project
    completed = subprocess.run(["bash", str(script), "-1"], env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 2 and not calls.exists()
