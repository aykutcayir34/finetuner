"""Monitor tab: live logs, loss curve and job control, refreshed by a timer."""

from __future__ import annotations

import pandas as pd
import gradio as gr

from ..core.jobs import MANAGER

STATUS_ICONS = {"pending": "⏳", "running": "🏃", "finished": "✅",
                "failed": "❌", "stopped": "⏹"}


def _job_choices() -> list[tuple[str, int]]:
    return [(f"#{j.id} {STATUS_ICONS.get(j.status, '')} {j.name}", j.id)
            for j in reversed(MANAGER.all())]


def _snapshot(job_id):
    job = MANAGER.get(int(job_id)) if job_id else MANAGER.latest()
    if job is None:
        return ("*No jobs yet — start one from the 🚀 Train tab.*", "",
                pd.DataFrame({"step": [], "loss": []}))
    header = (f"**Job #{job.id}** · {job.name} · "
              f"{STATUS_ICONS.get(job.status, '')} **{job.status}** · "
              f"⏱ {job.elapsed:.0f}s · {len(job.metrics)} loss points")
    df = pd.DataFrame(job.metrics, columns=["step", "loss"]) if job.metrics \
        else pd.DataFrame({"step": [], "loss": []})
    return header, job.log_text(), df


def build(app):
    with gr.Tab("📈 Monitor", id="monitor"):
        with gr.Row():
            job_pick = gr.Dropdown(label="Job", choices=_job_choices(), scale=3)
            refresh_btn = gr.Button("🔄 Refresh list", scale=1)
            stop_btn = gr.Button("⏹ Stop job", variant="stop", scale=1)
        header = gr.Markdown("*No jobs yet — start one from the 🚀 Train tab.*")
        with gr.Row():
            with gr.Column(scale=1):
                loss_plot = gr.LinePlot(x="step", y="loss", label="Training loss",
                                        value=pd.DataFrame({"step": [], "loss": []}))
            with gr.Column(scale=1):
                logs = gr.Textbox(label="Live logs", lines=20, max_lines=20,
                                  autoscroll=True, interactive=False)

        timer = gr.Timer(2.0)
        timer.tick(_snapshot, job_pick, [header, logs, loss_plot])

        refresh_btn.click(lambda: gr.update(choices=_job_choices()), None, job_pick)
        job_pick.change(_snapshot, job_pick, [header, logs, loss_plot])

        def on_stop(job_id):
            job = MANAGER.get(int(job_id)) if job_id else MANAGER.latest()
            if job is None:
                return gr.update()
            MANAGER.stop(job.id)
            return gr.update(choices=_job_choices())

        stop_btn.click(on_stop, job_pick, job_pick)
