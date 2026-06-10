"""Finetuner Studio — application entry point."""

from __future__ import annotations

import gradio as gr

from .core.registry import TASKS, mlx_available
from .ui import tab_dataset, tab_export, tab_model, tab_monitor, tab_playground, tab_train

BANNER = """
# 🎛️ Finetuner Studio
**Low-code fine-tuning on Apple Silicon** · powered by
[mlx-tune](https://github.com/ARahim3/mlx-tune) · {n_tasks} training paradigms,
zero boilerplate
"""


def build_app() -> gr.Blocks:
    ok, reason = mlx_available()
    health = "🟢 mlx-tune ready" if ok else f"🟡 GUI-only mode — {reason}"

    with gr.Blocks(title="Finetuner Studio") as app:
        gr.Markdown(BANNER.format(n_tasks=len(TASKS)))
        gr.Markdown(f"`{health}`")
        with gr.Tabs():
            tab_model.build(app)
            tab_dataset.build(app)
            tab_train.build(app)
            tab_monitor.build(app)
            tab_playground.build(app)
            tab_export.build(app)
        gr.Markdown(
            "<center><small>Finetuner Studio · load a model → drop a dataset → "
            "press train. The generated Python script is yours to keep.</small></center>"
        )
    return app


def main():
    theme = gr.themes.Soft(primary_hue="orange", secondary_hue="slate")
    build_app().launch(theme=theme)


if __name__ == "__main__":
    main()
