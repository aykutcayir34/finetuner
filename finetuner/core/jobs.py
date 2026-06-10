"""Background training job manager.

Training runs on a worker thread so the GUI stays responsive. Trainer stdout
is captured into a ring buffer; loss values are parsed out of the log stream
for live charting.
"""

from __future__ import annotations

import contextlib
import io
import re
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

# Matches lines like "step 10: loss 1.2345", "{'loss': 1.23, 'step': 10}", "10/100 | loss: 1.23"
_LOSS_RE = re.compile(r"loss[\"']?[:=\s]+([0-9]*\.?[0-9]+(?:e-?\d+)?)", re.IGNORECASE)
_STEP_RE = re.compile(r"(?:step|it(?:er)?)[\"']?[:=\s/]+(\d+)", re.IGNORECASE)


class _Tee(io.TextIOBase):
    """Write-through stream that feeds the job log and parses metrics."""

    def __init__(self, job: "Job", original):
        self.job = job
        self.original = original
        self._buf = ""

    def write(self, s: str) -> int:
        self.original.write(s)
        self._buf += s
        parts = re.split(r"[\n\r]", self._buf)
        self._buf = parts.pop()  # keep the unterminated tail
        for line in parts:
            if line.strip():
                self.job.add_log(line)
        return len(s)

    def flush(self):
        self.original.flush()


@dataclass
class Job:
    id: int
    name: str
    status: str = "pending"  # pending | running | finished | failed | stopped
    logs: deque = field(default_factory=lambda: deque(maxlen=2000))
    metrics: list = field(default_factory=list)  # [(step, loss)]
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    _step_guess: int = 0

    def add_log(self, line: str):
        self.logs.append(line)
        loss = _LOSS_RE.search(line)
        if loss:
            step_m = _STEP_RE.search(line)
            if step_m:
                self._step_guess = int(step_m.group(1))
            else:
                self._step_guess += 1
            try:
                self.metrics.append((self._step_guess, float(loss.group(1))))
            except ValueError:
                pass

    def log_text(self, last_n: int = 200) -> str:
        return "\n".join(list(self.logs)[-last_n:])

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at


class JobManager:
    def __init__(self):
        self._jobs: dict[int, Job] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def submit(self, name: str, target, *args, **kwargs) -> Job:
        """Run `target(job, *args, **kwargs)` on a worker thread with log capture."""
        with self._lock:
            job = Job(id=self._next_id, name=name)
            self._jobs[job.id] = job
            self._next_id += 1

        def runner():
            job.status = "running"
            job.started_at = time.time()
            tee_out = _Tee(job, __import__("sys").stdout)
            tee_err = _Tee(job, __import__("sys").stderr)
            try:
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    target(job, *args, **kwargs)
                job.status = "stopped" if job.stop_event.is_set() else "finished"
            except Exception:
                job.error = traceback.format_exc()
                job.add_log(job.error)
                job.status = "failed"
            finally:
                job.finished_at = time.time()

        threading.Thread(target=runner, name=f"finetuner-job-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def latest(self) -> Job | None:
        return self._jobs[max(self._jobs)] if self._jobs else None

    def stop(self, job_id: int):
        job = self._jobs.get(job_id)
        if job:
            job.stop_event.set()
            job.add_log("⏹ Stop requested — training will halt at the next step boundary.")


MANAGER = JobManager()
