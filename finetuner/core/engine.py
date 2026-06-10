"""Single persistent MLX engine thread.

MLX streams are thread-local: a model loaded on one thread cannot reliably be
trained or sampled from another ("There is no Stream(gpu, N) in current
thread"). Gradio runs every event handler on a different worker thread, so all
MLX work — model loading, training, generation — is funneled through one
long-lived engine thread. This also serializes GPU work, which is what a
single-device machine wants anyway.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

# mlx-tune's subprocess fallback shells out to `mlx_lm.lora`; make sure the
# interpreter's bin directory is on PATH even when the venv isn't activated.
_bin = str(Path(sys.executable).parent)
if _bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")


class _Engine:
    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="finetuner-mlx-engine",
                                        daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            fn, args, kwargs, done, box = self._q.get()
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 — re-raised on the caller thread
                box["error"] = exc
            finally:
                done.set()

    def call(self, fn, *args, **kwargs):
        """Run `fn` on the engine thread and block until it returns."""
        if threading.current_thread() is self._thread:
            return fn(*args, **kwargs)
        done = threading.Event()
        box: dict = {}
        self._q.put((fn, args, kwargs, done, box))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["result"]


ENGINE = _Engine()
