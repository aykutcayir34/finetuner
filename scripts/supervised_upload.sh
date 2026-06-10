#!/bin/zsh
# Self-healing HF upload: this network drops long-lived upload streams without
# erroring, so the worker hangs forever. Watch the log; if it goes quiet for
# >120s, kill and restart — upload-large-folder resumes from its local cache.
# Success is verified against the Hub API, not the worker's exit code.

REPO="acayir64/Llama-3.2-1B-Instruct-Turkish-Alpaca-mlx"
SRC="outputs/turkish-full/hub-stage"
LOG="/tmp/hf_upload_supervised.log"
# Xet: chunk-level dedupe means restarted workers skip already-uploaded
# chunks of partially-transferred files — exactly what a flaky line needs.
export HF_XET_HIGH_PERFORMANCE=1

check_done() {
  .venv/bin/python - <<'PY'
import sys
from huggingface_hub import HfApi
files = set(HfApi().list_repo_files("acayir64/Llama-3.2-1B-Instruct-Turkish-Alpaca-mlx"))
need_root = {"model.safetensors", "config.json", "tokenizer.json"}
n_ckpt = sum(1 for f in files if f.startswith("adapters/") and f.endswith(".safetensors"))
sys.exit(0 if need_root <= files and n_ckpt >= 21 else 1)
PY
}

attempt=0
while true; do
  attempt=$((attempt+1))
  if check_done; then
    echo "[supervisor] UPLOAD COMPLETE ✅ (verified via Hub API)"
    exit 0
  fi
  if [ "$attempt" -gt 40 ]; then
    echo "[supervisor] giving up after 40 attempts ❌"
    exit 1
  fi
  echo "[supervisor] attempt $attempt $(date '+%H:%M:%S')"
  .venv/bin/hf upload-large-folder "$REPO" "$SRC" --repo-type model --num-workers 4 \
    > "$LOG" 2>&1 &
  up_pid=$!
  while kill -0 $up_pid 2>/dev/null; do
    sleep 30
    age=$(( $(date +%s) - $(stat -f %m "$LOG") ))
    if [ "$age" -gt 120 ]; then
      echo "[supervisor] log quiet ${age}s — restarting worker"
      kill -9 $up_pid 2>/dev/null
      sleep 3
      break
    fi
  done
  sleep 5
done
