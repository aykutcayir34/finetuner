# Roadmap — Finetuner Studio

*Owner: Delivery Lead*

## v0.1 — Foundation (this release) ✅

- All 12 mlx-tune trainers in the registry and GUI
- Hub + local model/dataset loading, drag-drop upload
- Automatic dataset format detection & normalization
- Background jobs with live logs, loss chart, stop
- Playground chat, exports (adapters / merged / GGUF / Hub)
- Code generator and YAML recipes
- Hermetic test suite

## v0.2 — Trust & insight

- [ ] Train/validation split + eval loss in the Monitor chart
- [ ] Per-task dynamic hyperparameter panels (GRPO reward editor, embedding loss picker)
- [ ] Manual column-mapping editor when detection confidence < 70%
- [ ] Dataset statistics card (token length histogram, dedup warnings)
- [ ] Recipe gallery: curated starter recipes per task

## v0.3 — Scale & automation

- [ ] Job queue with sequential scheduling and run history persisted to disk
- [ ] Hyperparameter sweep (grid over lr / r / steps) with comparison chart
- [ ] Trackio/W&B logging integration
- [ ] Headless mode: `finetuner run recipe.yaml` (CLI parity with the GUI)

## v0.4 — Reach

- [ ] i18n (Turkish UI 🇹🇷 first)
- [ ] Hugging Face Space template (GUI-only planning mode)
- [ ] Model comparison playground (base vs tuned side by side)
- [ ] One-click Ollama handoff after GGUF export

## Risks

| Risk | Mitigation |
|---|---|
| mlx-tune API drift | registry + kwarg filtering seams (ADR-3); pin a tested version range when stable |
| GGUF export limits for 4-bit bases | warned inline; dequantization recipe planned |
| Gradio major-version changes | UI isolated in `finetuner/ui/*`; core has no Gradio imports |
