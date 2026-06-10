# Test Plan — Finetuner Studio

*Owner: QA Engineer*

## Strategy

The backend boundary (mlx-tune, Apple-Silicon-only) splits testing into:

1. **Hermetic unit tests** (run anywhere, incl. CI): detection, normalization,
   codegen, recipes, job lifecycle. These cover all logic this repo owns.
2. **On-device smoke tests** (manual, Apple Silicon): real model load,
   a 10-step SFT run, playground generation, adapter export.

## Automated suite (`pytest`)

| File | Covers |
|---|---|
| `test_detector.py` | every canonical format, synonym matching, normalization output shapes, empty input |
| `test_codegen.py` | generated script for **all 12 tasks** parses with `ast.parse`, local vs Hub dataset blocks, epochs-vs-steps, DPO β default |
| `test_recipes.py` | YAML round-trip fidelity incl. `extra` dict |
| `test_jobs.py` | success path with loss parsing, failure capture, cooperative stop |

Run: `pip install -e '.[dev]' && pytest`

## Automated on-device checks (Apple Silicon)

- `scripts/smoke_sft.py` — headless E2E through the core stack: detection →
  model+LoRA → 10-step SFT via the job manager → generation.
- `scripts/gui_drive_turkish_sft.py` — Playwright drives the real GUI:
  loads Llama-3.2-1B-Instruct-4bit, pulls `TFLai/Turkish-Alpaca` from the Hub,
  verifies 95% Alpaca detection, trains 60 steps, checks the Monitor reaches
  `finished`, and chats in the Playground. Screenshots land in
  `docs/screenshots/`. (Requires `pip install playwright && playwright
  install chromium` and a running `finetuner` server.)

## Manual smoke checklist (Apple Silicon, before each release)

- [ ] `finetuner` launches; banner shows `🟢 mlx-tune ready`.
- [ ] Hub search returns models; `mlx-community/Llama-3.2-1B-Instruct-4bit`
      loads with LoRA in < 1 min.
- [ ] `examples/alpaca_sample.jsonl` upload → detected as Alpaca ≥ 90%.
- [ ] 10-step SFT run: Monitor shows logs and a descending loss curve; Stop works.
- [ ] Playground answers a prompt with the tuned model.
- [ ] Adapter export writes `lora_model/`; script export parses and matches the run.
- [ ] On a non-MLX machine: app launches in 🟡 GUI-only mode; dataset
      detection and codegen still work.

## Quality gates

- `ruff check` clean; `pytest` green on macOS + Linux (GUI-only path).
- New detector formats require a positive **and** a negative test.
