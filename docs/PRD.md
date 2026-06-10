# Product Requirements — Finetuner Studio

*Owner: Product Manager*

## Problem

Fine-tuning on Apple Silicon is finally practical thanks to MLX and
[mlx-tune](https://github.com/ARahim3/mlx-tune), but it still requires writing
Python: loading models, shaping datasets into trainer-specific schemas,
wiring configs, parsing logs. Practitioners who *could* benefit most —
domain experts iterating on data — are blocked by exactly that boilerplate.

## Vision

**A local, low-code studio where "model + dataset + train button" is the whole
workflow**, and where graduating to code is one click, never a rewrite.

## Personas

- **The Tinkerer** — has an M-series Mac, wants a Turkish-instruction LLM this
  weekend. Needs: search Hub → pick dataset → train → chat.
- **The Data Owner** — has a proprietary `.csv`/`.jsonl`; doesn't know what
  "ShareGPT format" means. Needs: drop a file, have the format figured out.
- **The Engineer** — prototypes in the GUI, then wants a script for the GPU
  box. Needs: faithful code export (mlx-tune is Unsloth-compatible, so the
  script ports to cloud).

## Functional requirements

| # | Requirement | Status |
|---|---|---|
| F1 | Expose **every** mlx-tune trainer (SFT, DPO, ORPO, SimPO, KTO, GRPO, CPT, VLM-SFT, TTS, STT, Embedding, OCR) in the GUI | ✅ `core/registry.py` |
| F2 | Load models from the Hugging Face Hub (with search) **and** from local directories | ✅ `core/models.py` |
| F3 | Load datasets from the Hub (with search), local paths, and drag-drop upload (`.jsonl/.json/.csv/.tsv/.parquet`) | ✅ `core/data.py` |
| F4 | **Automatically detect** dataset format (Alpaca, ShareGPT, ChatML, preference, KTO, raw text, embedding pairs, audio+text, image+text, vision chat) and normalize to the trainer schema | ✅ `core/detector.py` |
| F5 | Full LoRA configuration (rank, alpha, dropout, target modules) | ✅ Model tab |
| F6 | Background training with live logs, loss curve, stop button | ✅ `core/jobs.py`, Monitor tab |
| F7 | Inference playground (chat) against the in-memory model | ✅ Playground tab |
| F8 | Export: adapters / merged fp16 / GGUF / push to Hub | ✅ `core/export.py` |
| F9 | One-click generation of an equivalent standalone Python script | ✅ `core/codegen.py` |
| F10 | Save/load complete run configurations as YAML recipes | ✅ `core/recipes.py` |

## Non-functional requirements

- **N1** Runs fully locally; no telemetry, no accounts (HF token only for Hub pushes).
- **N2** GUI stays responsive during training (worker threads).
- **N3** Survives mlx-tune API drift: configs are filtered against dataclass
  fields; trainer kwargs fall back between `tokenizer`/`processor`.
- **N4** Usable without MLX installed (planning mode) — enables CI and demos.

## Success metrics

- Zero lines of code from "open app" to "loss curve" for an Alpaca dataset.
- Format detection ≥ 90% precision on the common public dataset shapes.
- Generated script runs unmodified with `python train.py` on a Mac with mlx-tune.

## Out of scope (v0.x)

Multi-user/server deployments, distributed training, experiment databases,
hyperparameter sweeps (see [ROADMAP.md](ROADMAP.md)).
