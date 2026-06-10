"""End-to-end SFT smoke test through the Finetuner Studio stack (no GUI).

Exercises the exact code paths the Studio uses: model loading + LoRA,
dataset loading + automatic format detection + normalization, and a real
10-step training run through the background job manager with loss capture.

Run on Apple Silicon:  python scripts/smoke_sft.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetuner.core import data, models
from finetuner.core.detector import detect, normalize
from finetuner.core.jobs import MANAGER
from finetuner.core.registry import mlx_available
from finetuner.core.training import RunConfig, run_training

MODEL = "mlx-community/SmolLM2-135M-Instruct"
DATASET = Path(__file__).resolve().parents[1] / "examples" / "alpaca_sample.jsonl"
STEPS = 10


def main() -> int:
    ok, reason = mlx_available()
    if not ok:
        print(f"SKIP: {reason}")
        return 1

    print(f"[1/5] Loading dataset {DATASET.name} …")
    rows = data.load_local_dataset(str(DATASET))
    det = detect(rows)
    print(f"      detected={det.format} confidence={det.confidence:.0%} "
          f"mapping={det.mapping} tasks={det.suggested_tasks}")
    assert det.format == "alpaca" and "sft" in det.suggested_tasks

    print(f"[2/5] Loading model {MODEL} (+LoRA r=8) …")
    model, tokenizer = models.load_model("sft", MODEL, max_seq_length=512, load_in_4bit=True)
    model = models.apply_lora("sft", model, r=8, lora_alpha=16)

    print("[3/5] Normalizing dataset with the model's chat template …")
    dataset = normalize(rows, det, "sft", tokenizer)
    print(f"      {len(dataset)} samples, first 80 chars: {dataset[0]['text'][:80]!r}")

    print(f"[4/5] Training {STEPS} steps via JobManager …")
    cfg = RunConfig(task="sft", model_name=MODEL, batch_size=1,
                    learning_rate=1e-4, max_steps=STEPS, warmup_steps=2,
                    output_dir="outputs/smoke")
    job = MANAGER.submit("smoke-sft", run_training, cfg, model, tokenizer, dataset)
    while job.status in ("pending", "running"):
        time.sleep(2)
    print(f"      status={job.status} elapsed={job.elapsed:.0f}s "
          f"loss_points={len(job.metrics)}")
    if job.status != "finished":
        print("---- logs ----")
        print(job.log_text(40))
        return 1
    if job.metrics:
        first, last = job.metrics[0], job.metrics[-1]
        print(f"      loss: step {first[0]} → {first[1]:.4f}  …  "
              f"step {last[0]} → {last[1]:.4f}")

    print("[5/5] Generation sanity check …")
    try:
        from mlx_lm import generate
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Merhaba! Tek cümleyle kendini tanıt."}],
            tokenize=False, add_generation_prompt=True)
        out = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)
        print(f"      model says: {out.strip()[:200]}")
    except Exception as exc:
        print(f"      generation skipped: {exc}")

    print("\nSMOKE TEST PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
