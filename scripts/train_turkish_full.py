"""Long Turkish instruction-tuning run through the Finetuner Studio stack.

Full TFLai/Turkish-Alpaca (≈52k rows), Llama-3.2-1B-Instruct-4bit,
LoRA on all attention+MLP projections, 2000 steps. Saves LoRA adapters and a
merged model ready for Hub upload.

Run on Apple Silicon:  python -u scripts/train_turkish_full.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetuner.core import data, models
from finetuner.core.detector import detect, normalize
from finetuner.core.engine import ENGINE
from finetuner.core.jobs import MANAGER
from finetuner.core.registry import mlx_available
from finetuner.core.training import RunConfig, run_training

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
DATASET = "TFLai/Turkish-Alpaca"
STEPS = 2000
OUT = "outputs/turkish-full"


def main() -> int:
    ok, reason = mlx_available()
    if not ok:
        print(f"SKIP: {reason}")
        return 1

    print(f"[1/5] Loading FULL dataset {DATASET} …", flush=True)
    rows = data.load_hub_dataset(DATASET, split="train")
    det = detect(rows)
    print(f"      {len(rows)} rows · detected={det.format} ({det.confidence:.0%})", flush=True)
    assert det.format == "alpaca"

    print(f"[2/5] Loading {MODEL} + LoRA r=16 on attention+MLP …", flush=True)
    model, tokenizer = models.load_model("sft", MODEL, max_seq_length=2048, load_in_4bit=True)
    model = models.apply_lora(
        "sft", model, r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    print("[3/5] Normalizing with the model's chat template …", flush=True)
    dataset = normalize(rows, det, "sft", tokenizer)

    print(f"[4/5] Training {STEPS} steps (batch 2, lr 1e-4, cosine) …", flush=True)
    cfg = RunConfig(task="sft", model_name=MODEL, batch_size=2, learning_rate=1e-4,
                    max_steps=STEPS, warmup_steps=50, output_dir=OUT)
    t0 = time.time()
    job = MANAGER.submit("turkish-full-sft", run_training, cfg, model, tokenizer, dataset)
    while job.status in ("pending", "running"):
        time.sleep(10)
    print(f"      status={job.status} elapsed={(time.time()-t0)/60:.1f}min "
          f"loss_points={len(job.metrics)}", flush=True)
    if job.status != "finished":
        print(job.log_text(60))
        return 1
    if job.metrics:
        print("      loss trajectory:",
              " → ".join(f"{loss:.3f}" for _, loss in job.metrics[::4]), flush=True)

    print("[5/5] Saving merged model …", flush=True)
    merged_dir = f"{OUT}/merged"
    ENGINE.call(model.save_pretrained_merged, merged_dir, tokenizer)
    print(f"      merged model → {merged_dir}", flush=True)
    print(f"      adapters     → {OUT}/adapters", flush=True)

    print("\nFULL TRAINING DONE ✅", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
