"""Checkpoint-bound, append-only recovery of interrupted ordered score exports."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path

from audio_deepfake_detection.protocol import checkpoint_digest


HEADER = ["utterance_id", "dataset", "label", "raw_score"]


def _save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_row(row, expected, dataset):
    identifier, observed_dataset, label, score = row
    if (identifier, observed_dataset, int(label)) != (expected[0], dataset, expected[1]):
        raise ValueError("Partial score rows do not match the canonical ordered split")
    if int(label) not in (0, 1) or not math.isfinite(float(score)):
        raise ValueError("Invalid label or non-finite partial score")


def prepare_score_recovery(output, expected, dataset, identity, *, adopt_legacy=False):
    """Verify every saved row and its hash before returning a resume offset.

    Legacy .tmp files lack historical checksums, so accepting one requires an
    explicit operator decision, recorded permanently in the recovery journal.
    A stale/tampered journal fails closed; no old score bytes are overwritten.
    """
    output = Path(output)
    temporary = output.with_name(output.name + ".tmp")
    journal = output.with_name(output.name + ".progress.json")
    membership = hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest()
    binding = {"version": 1, "dataset": dataset, "output": output.name,
               "membership_sha256": membership, "identity": identity}
    existing = json.loads(journal.read_text()) if journal.exists() else None
    if existing is not None and any(existing.get(k) != v for k, v in binding.items()):
        raise ValueError("Partial score identity changed; refusing to resume")
    artifact = output if output.exists() else temporary
    if not artifact.exists():
        if existing is not None:
            raise ValueError("Recovery journal exists but its score file is missing")
        return {**binding, "rows": 0, "legacy_adopted_by_operator": False}
    if existing is None and not adopt_legacy:
        raise ValueError("Unregistered partial scores: explicit --adopt-legacy-partial is required")
    digest = checkpoint_digest(artifact)
    if existing is not None and existing.get("score_sha256") != digest:
        raise ValueError("Partial score checksum mismatch; preserve the file for inspection")
    count = 0
    with artifact.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != HEADER:
            raise ValueError("Partial score CSV header is invalid")
        for row in reader:
            if count >= len(expected) or len(row) != 4:
                raise ValueError("Partial score CSV has extra or incomplete rows")
            _validate_row(row, expected[count], dataset)
            count += 1
    # Appending after a truncated last line could merge records. Fail closed.
    with artifact.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) != b"\n":
            raise ValueError("Partial score CSV has an incomplete final line")
    if existing is not None and existing.get("rows") != count:
        raise ValueError("Partial row count differs from the recovery journal")
    if output.exists() and count != len(expected):
        raise ValueError("Published score export is incomplete")
    state = {**binding, "rows": count, "score_sha256": digest,
             "legacy_adopted_by_operator": existing.get("legacy_adopted_by_operator", False) if existing else True}
    if existing is None:
        state["legacy_adoption_sha256"] = digest
        state["legacy_adoption_rows"] = count
    else:
        for key in ("legacy_adoption_sha256", "legacy_adoption_rows"):
            if key in existing:
                state[key] = existing[key]
    _save(journal, state)
    return state


def write_recoverable_scores(output, rows, expected, dataset, state, *, checkpoint_every=5000):
    """Append only the unscored suffix, preserving progress even on exceptions.

    A forced kill between journal commits fails checksum verification on the
    next launch instead of silently adopting unregistered new bytes.
    """
    output = Path(output)
    temporary = output.with_name(output.name + ".tmp")
    journal = output.with_name(output.name + ".progress.json")
    if output.exists():
        raise FileExistsError(output)
    count = state["rows"]
    mode = "a" if temporary.exists() else "x"
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if mode == "x":
            writer.writerow(HEADER)

        def commit():
            handle.flush()
            os.fsync(handle.fileno())
            state.update(rows=count, score_sha256=checkpoint_digest(temporary))
            _save(journal, state)

        commit()
        try:
            for row in rows:
                if count >= len(expected):
                    raise ValueError("Scoring produced extra rows")
                _validate_row(row, expected[count], dataset)
                writer.writerow((row[0], row[1], int(row[2]), float(row[3])))
                count += 1
                if count % checkpoint_every == 0:
                    commit()
            if count != len(expected):
                raise ValueError("Scoring ended before the canonical split was complete")
        finally:
            commit()
    temporary.replace(output)
