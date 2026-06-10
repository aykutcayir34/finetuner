"""Upload the Turkish-Alpaca fine-tune (merged model + LoRA adapters) to the Hub.

Usage:  python -u scripts/upload_to_hub.py [repo_id]
Default repo_id: <hf-username>/Llama-3.2-1B-Instruct-Turkish-Alpaca-mlx
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

OUT = Path("outputs/turkish-full")
MERGED = OUT / "merged"
ADAPTERS = OUT / "adapters"

CARD = """---
language:
  - tr
license: llama3.2
base_model: mlx-community/Llama-3.2-1B-Instruct-4bit
datasets:
  - TFLai/Turkish-Alpaca
tags:
  - mlx
  - llama
  - turkish
  - lora
  - instruction-tuning
  - finetuner-studio
pipeline_tag: text-generation
---

# Llama-3.2-1B-Instruct · Turkish-Alpaca (MLX)

**🇹🇷 Türkçe** — Llama-3.2-1B-Instruct (4-bit MLX),
[TFLai/Turkish-Alpaca](https://huggingface.co/datasets/TFLai/Turkish-Alpaca)
veri kümesinin tamamı (51.914 örnek) üzerinde LoRA ile ince ayarlandı. Eğitim,
Apple Silicon üzerinde [Finetuner Studio](https://github.com/aykutcayir34/finetuner)
(arka uç: [mlx-tune](https://github.com/ARahim3/mlx-tune)) ile tamamen yerel
olarak yapıldı; veri kümesi formatı otomatik algılandı, eğitim native MLX ile koştu.

**🇬🇧 English** — Llama-3.2-1B-Instruct (4-bit MLX) fine-tuned with LoRA on
the full Turkish-Alpaca instruction dataset (51,914 rows), trained entirely
locally on an Apple Silicon Mac with **Finetuner Studio**, a low-code GUI
over mlx-tune; the dataset format was auto-detected and training ran on
native MLX.

## Training details · Eğitim ayrıntıları

| | |
|---|---|
| Base model | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Dataset | `TFLai/Turkish-Alpaca` (51,914 rows, auto-detected Alpaca format) |
| Method | LoRA r=16, α=16 on q/k/v/o + gate/up/down projections |
| Steps | 2,000 (batch 2, grad accum 1, seq len 2,048) |
| LR | 1e-4, cosine schedule, 50 warmup steps |
| Val loss | {val_first} → {val_last} |
| Hardware | Apple Silicon (MLX native training, ~{tps} tok/s, ≤{mem} GB) |

🇹🇷 Bu repo, kökte **birleştirilmiş bf16 MLX modelini** (4-bit taban LoRA ile
birleştirilirken dequantize edilir), [`adapters/`](./tree/main/adapters)
altında ise her 100 adımda kaydedilen **tüm ara checkpoint'lerle birlikte ham
LoRA adaptörlerini** içerir.
🇬🇧 This repo contains the **merged bf16 MLX model** at the root (the 4-bit
base is dequantized during LoRA fusion) and the raw **LoRA adapters** under
[`adapters/`](./tree/main/adapters), including all intermediate checkpoints
saved every 100 steps.

## Checkpoints · Ara kayıtlar

| Checkpoint | Val loss | Train loss |
|---|---|---|
{checkpoint_table}

## Usage · Kullanım (MLX)

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("{repo_id}")
prompt = tokenizer.apply_chat_template(
    [{{"role": "user", "content": "Yapay zeka nedir? Kısaca açıkla."}}],
    tokenize=False, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=200))
```

## Built with Finetuner Studio 🎛️

🇹🇷 Bu model, [Finetuner Studio](https://github.com/aykutcayir34/finetuner)
arayüzü üzerinden yapılandırıldı, eğitildi ve izlendi.
🇬🇧 This model was configured, trained and monitored entirely through the
Finetuner Studio GUI:

![Automatic dataset format detection](https://raw.githubusercontent.com/aykutcayir34/finetuner/main/docs/screenshots/03_dataset_detected.png)

![Live training monitor](https://raw.githubusercontent.com/aykutcayir34/finetuner/main/docs/screenshots/05_monitor_loss.png)

## Intended use & limitations · Kullanım amacı ve sınırlamalar

🇹🇷 Türkçe talimat takibi için eğitilmiş küçük (1B) bir modeldir; bilgi yoğun
sorularda hata yapabilir, üretim kullanımı öncesi değerlendirme gerekir.
🇬🇧 A small 1B model for Turkish instruction following — expect factual
mistakes; evaluate before production use.

---
*Trained & published with [Finetuner Studio](https://github.com/aykutcayir34/finetuner) 🎛️*
"""


def parse_metrics(log_path: Path) -> dict:
    """Pull headline numbers and a per-checkpoint loss table out of the log."""
    import re
    text = log_path.read_text() if log_path.exists() else ""
    vals = re.findall(r"Val loss ([0-9.]+)", text)
    tps = re.findall(r"Tokens/sec ([0-9.]+)", text)
    mem = re.findall(r"Peak mem ([0-9.]+) GB", text)

    val_by_iter = {int(i): v for i, v in re.findall(r"Iter (\d+): Val loss ([0-9.]+)", text)}
    train_by_iter = {int(i): v for i, v in re.findall(r"Iter (\d+): Train loss ([0-9.]+)", text)}
    ckpt_iters = sorted(int(i) for i in re.findall(r"Iter (\d+): Saved adapter weights", text))
    rows = [f"| [`{i:07d}_adapters.safetensors`](./blob/main/adapters/{i:07d}_adapters.safetensors) "
            f"| {val_by_iter.get(i, '—')} | {train_by_iter.get(i, '—')} |"
            for i in ckpt_iters]
    return {
        "val_first": vals[0] if vals else "—",
        "val_last": vals[-1] if vals else "—",
        "tps": f"{float(tps[-1]):.0f}" if tps else "—",
        "mem": max(mem, key=float) if mem else "—",
        "checkpoint_table": "\n".join(rows) if rows else "| — | — | — |",
    }


def main() -> int:
    user = whoami()["name"]
    repo_id = sys.argv[1] if len(sys.argv) > 1 else \
        f"{user}/Llama-3.2-1B-Instruct-Turkish-Alpaca-mlx"

    assert MERGED.is_dir(), f"merged model not found at {MERGED}"
    assert (MERGED / "config.json").exists(), "merged model incomplete (no config.json)"

    metrics = parse_metrics(Path("outputs/turkish-full/train.log"))
    card = CARD.format(repo_id=repo_id, **metrics)

    api = HfApi()
    print(f"[1/3] Creating repo {repo_id} …", flush=True)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    print(f"[2/3] Uploading merged model from {MERGED} …", flush=True)
    api.upload_folder(folder_path=str(MERGED), repo_id=repo_id,
                      commit_message="Merged 4-bit MLX model (Finetuner Studio)")
    if ADAPTERS.is_dir():
        print(f"      Uploading LoRA adapters from {ADAPTERS} …", flush=True)
        api.upload_folder(folder_path=str(ADAPTERS), repo_id=repo_id,
                          path_in_repo="adapters",
                          commit_message="LoRA adapters")

    print("[3/3] Writing model card …", flush=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo_id, commit_message="Model card")

    print(f"\nPUBLISHED ✅  https://huggingface.co/{repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
