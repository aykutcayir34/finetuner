# UX Design — Finetuner Studio

*Owner: UX Designer*

## Design principles

1. **Left-to-right workflow.** Tabs are ordered as the mental model:
   Model → Dataset → Train → Monitor → Playground → Export. A first-time user
   can simply walk the tabs.
2. **Progressive disclosure.** Defaults work out of the box
   (Llama-3.2-1B-4bit, LoRA r=16, lr=2e-4, 100 steps). Power options live in
   accordions (script export, recipes) and grouped sliders.
3. **Detection is transparent, never silent.** The dataset panel shows the
   detected format, a 0–100% confidence bar, the exact column mapping
   (`text ← body`) and which trainers accept it. Wrong guesses are visible
   and overridable, not hidden.
4. **Errors are messages, not crashes.** Every handler catches exceptions and
   renders `❌ reason` in a status Markdown area; missing prerequisites give
   directions ("Load a model first — 🧠 Model tab").
5. **Escape hatch to code.** "🧾 Generate Python script" is a first-class
   button on the Train tab — the Studio teaches mlx-tune instead of hiding it.

## Tab specs

| Tab | Primary action | Key feedback |
|---|---|---|
| 🧠 Model | ⚡ Load model | task explainer card, load status with LoRA confirmation |
| 📚 Dataset | 📥 Load dataset | preview table (8 rows), detection card with confidence bar |
| 🚀 Train | 🏁 Start training | job id + sample count; jumps user to Monitor |
| 📈 Monitor | (passive, 2s auto-refresh) | status header, live log tail, loss `LinePlot`, ⏹ stop |
| 💬 Playground | chat submit | streaming-free simple chat, temperature/max-tokens sliders |
| 📦 Export | 4 export buttons | one status line per action; GGUF limitation warned inline |

## Visual identity

- Gradio `Soft` theme, **orange** primary (forge/heat metaphor), slate secondary.
- Emoji as functional iconography — instant recognition without icon assets.
- System health line under the banner: `🟢 mlx-tune ready` vs
  `🟡 GUI-only mode` so users immediately know what this machine can do.

## Localization note

The UI ships in English (open-source reach); example prompts and sample
datasets include Turkish to reflect the home audience. Full i18n is a
roadmap item.
