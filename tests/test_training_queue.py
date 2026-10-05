import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from audio_deepfake_detection.protocol import validate_run_name
from scripts.wavlm import train


def test_batch_96_default_and_separate_output_family(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f2"])
    args = train.parse_args()
    assert args.batch_size == 96
    assert args.run_name == "wavlm_bs96"
    assert (args.num_workers, args.eval_workers, args.prefetch_factor) == (16, 4, 2)


def test_explicit_batch_and_family(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f1", "--batch-size", "64", "--run-name", "wavlm_bs64"])
    args = train.parse_args()
    assert (args.batch_size, args.run_name) == (64, "wavlm_bs64")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_batch(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f2", "--batch-size", value])
    with pytest.raises(SystemExit):
        train.parse_args()


@pytest.mark.parametrize("value", ["../wavlm", "/tmp/wavlm", "wavlm/f1", "", "."])
def test_invalid_output_family(value):
    with pytest.raises(ValueError):
        validate_run_name(value)


def test_training_uses_requested_batch_and_sources_only(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(sys, "argv", ["train.py", "--fold", "f2", "--batch-size", "96"])
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train, "set_seed", lambda seed: None)

    def reader(domain, split, root):
        opened.append((domain, split))
        return [("synthetic.wav", 0)]

    def loader(datasets, **kwargs):
        assert kwargs["batch_size"] == 96
        raise RuntimeError("synthetic loader boundary; no GPU or training step")

    monkeypatch.setattr(train, "read_dataset", reader)
    monkeypatch.setattr(train, "make_wavlm_train_loader", loader)
    with pytest.raises(RuntimeError, match="synthetic loader boundary"):
        train.main()
    assert (tmp_path / "outputs/wavlm_bs96/f2/1234").is_dir()
    assert opened == [(domain, split) for domain in ("asv2019", "asv5", "speechfake") for split in ("train", "dev")]


@pytest.fixture
def fake_queue(tmp_path):
    # A temporary project and Python stub verify shell scheduling without training.
    queue = tmp_path / "scripts/wavlm/train_folds.sh"
    queue.parent.mkdir(parents=True)
    actual = Path(__file__).resolve().parents[1] / "scripts/wavlm/train_folds.sh"
    queue.write_text(actual.read_text())
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    (tmp_path / "activate.sh").write_text(
        f"export PROJECT_ROOT={shlex.quote(str(tmp_path))}\n"
        f"export PATH={shlex.quote(str(binary_dir))}:$PATH\n"
        f"cd {shlex.quote(str(tmp_path))}\n"
    )
    python_stub = binary_dir / "python"
    python_stub.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
tokens = sys.argv[1:]
if tokens[0] == '-u':
    tokens = tokens[1:]
args = dict(zip(tokens[1::2], tokens[2::2]))
with open(os.environ['QUEUE_TEST_CALLS'], 'a') as handle:
    handle.write(json.dumps(args) + '\\n')
if args['--fold'] == os.environ.get('QUEUE_TEST_FAIL_FOLD'):
    sys.exit(5)
if not os.environ.get('QUEUE_TEST_NO_COMPLETION'):
    output = Path(os.environ['PROJECT_ROOT']) / 'outputs' / args['--run-name'] / args['--fold'] / args['--seed']
    (output / 'training_complete.json').write_text('{}')
print('Synthetic successful trainer')
""")
    python_stub.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = {**os.environ, "QUEUE_TEST_CALLS": str(calls)}
    return queue, calls, env


def queued_calls(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_queue_order_and_recipe(fake_queue):
    queue, calls, env = fake_queue
    result = subprocess.run(["bash", str(queue), "2345"], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    observed = queued_calls(calls)
    assert [args["--fold"] for args in observed] == ["f2", "f3", "f4"]
    assert all(args["--batch-size"] == "96" and args["--run-name"] == "wavlm_bs96"
               and args["--seed"] == "2345" for args in observed)


@pytest.mark.parametrize("failed_fold,expected", [("f2", ["f2"]), ("f3", ["f2", "f3"])])
def test_queue_stops_after_training_failure(fake_queue, failed_fold, expected):
    queue, calls, env = fake_queue
    env["QUEUE_TEST_FAIL_FOLD"] = failed_fold
    result = subprocess.run(["bash", str(queue)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert [args["--fold"] for args in queued_calls(calls)] == expected
    assert "queue stopped" in result.stdout


def test_queue_requires_completion_marker(fake_queue):
    queue, calls, env = fake_queue
    env["QUEUE_TEST_NO_COMPLETION"] = "1"
    result = subprocess.run(["bash", str(queue)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert [args["--fold"] for args in queued_calls(calls)] == ["f2"]


def test_queue_rejects_concurrent_queue(fake_queue):
    import fcntl

    queue, calls, env = fake_queue
    lock = queue.parents[2] / "outputs/wavlm_bs96/.queue.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(queue)], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert not queued_calls(calls)
