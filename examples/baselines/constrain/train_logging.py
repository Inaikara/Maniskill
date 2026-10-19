"""Durable run logging for constrain training (Ticket 04).

``RunLogger`` keeps an auditable, crash-safe record of a training run in the
run directory:

- ``run_config.json`` — the exact configuration snapshot written once before
  training starts (hyperparameters, demo path, data counts, base weight
  source and checksum),
- ``logs/loss.jsonl`` — one JSON line per training update, appended and
  flushed per step (a resumed run appends; prior lines are never clobbered),
- ``progress.json`` — the current step/total and recent loss statistics,
  replaced atomically so a reader never sees a partial file.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_write_json(path: Path, payload: dict):
    """Replace ``path`` with ``payload`` atomically (tmp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLogger:
    """Per-step JSONL loss log, atomic progress file and run-config snapshot."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.loss_path = self.run_dir / "logs" / "loss.jsonl"

    def write_config(self, config: dict):
        atomic_write_json(self.run_dir / "run_config.json", config)

    def log_step(self, step: int, loss: float, lr: float = None, extra: dict = None):
        """Append one JSON line for update ``step`` and flush it to disk."""
        record = dict(step=int(step), loss=float(loss), updated_at=_now_iso())
        if lr is not None:
            record["lr"] = float(lr)
        if extra:
            record.update(extra)
        with open(self.loss_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log_progress(self, step: int, total: int, extra: dict = None) -> dict:
        progress = dict(
            step=int(step),
            total=int(total),
            updated_at=_now_iso(),
            unix_time=time.time(),
        )
        if extra:
            progress.update(extra)
        atomic_write_json(self.run_dir / "progress.json", progress)
        return progress

    def truncate_after(self, step: int) -> int:
        """Drop logged lines with ``step`` beyond the given checkpoint step.

        Crash-recovery seam for resuming from a periodic checkpoint inside the
        same run directory: updates computed after the checkpoint but not
        covered by it are re-trained on resume, so their stale log lines (and
        any torn final line from a crashed write) must be removed to keep the
        JSONL an exact record of the actual training trajectory. Returns the
        number of removed lines.
        """
        if not self.loss_path.exists():
            return 0
        kept, removed = [], 0
        for line in self.loss_path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                removed += 1
                continue
            if record.get("step", 0) <= step:
                kept.append(line)
            else:
                removed += 1
        tmp = self.loss_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(kept) + ("\n" if kept else ""))
        os.replace(tmp, self.loss_path)
        return removed
