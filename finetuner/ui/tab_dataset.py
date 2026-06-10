"""Dataset tab: load from the Hub, a local path, or upload — then auto-detect the format."""

from __future__ import annotations

import pandas as pd
import gradio as gr

from ..core import data as datalib
from ..core.detector import detect
from ..core.registry import get_task
from ..core.state import STATE

PREVIEW_ROWS = 8


def _preview_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows[:PREVIEW_ROWS])
    return df.map(lambda v: str(v)[:300] if not isinstance(v, (int, float, bool)) else v)


def _detection_md() -> str:
    det = STATE.detection
    if det is None:
        return ""
    bar = "🟩" * round(det.confidence * 10) + "⬜" * (10 - round(det.confidence * 10))
    lines = [
        f"### 🔎 Detected format: **{det.label}**",
        f"Confidence: {bar} **{det.confidence:.0%}**",
    ]
    if det.mapping:
        mapped = ", ".join(f"`{k}` ← `{v}`" for k, v in det.mapping.items())
        lines.append(f"Column mapping: {mapped}")
    if det.suggested_tasks:
        tasks = ", ".join(f"**{get_task(t).label}**" for t in det.suggested_tasks)
        lines.append(f"Compatible trainers: {tasks}")
    for note in det.notes:
        lines.append(f"> 💡 {note}")
    return "\n\n".join(lines)


def _ingest(rows: list[dict], source: str, is_local: bool) -> tuple[str, pd.DataFrame, str]:
    STATE.raw_rows = rows
    STATE.detection = detect(rows)
    STATE.dataset_source = source
    STATE.dataset_is_local = is_local
    return (f"✅ Loaded **{len(rows)}** rows from `{source}`.",
            _preview_df(rows), _detection_md())


def build(app):
    with gr.Tab("📚 Dataset", id="dataset"):
        gr.Markdown("### Load a dataset — the format is detected automatically")
        source = gr.Radio(["Hugging Face Hub", "Local file", "Upload"],
                          value="Hugging Face Hub", label="Source")

        with gr.Group() as hub_group:
            with gr.Row():
                query = gr.Textbox(label="Search Hub datasets", placeholder="e.g. alpaca turkish", scale=3)
                search_btn = gr.Button("🔍 Search", scale=1)
            with gr.Row():
                ds_name = gr.Dropdown(label="Dataset", allow_custom_value=True, choices=[],
                                      info="Pick a result or type any dataset id.", scale=3)
                split = gr.Textbox(value="train", label="Split", scale=1)
                subset = gr.Textbox(value="", label="Config (optional)", scale=1)

        local_path = gr.Textbox(label="Local dataset path", visible=False,
                                placeholder="~/data/train.jsonl  (.jsonl/.json/.csv/.tsv/.parquet)")
        upload = gr.File(label="Upload dataset", visible=False,
                         file_types=[".jsonl", ".json", ".csv", ".tsv", ".parquet"])

        with gr.Row():
            max_rows = gr.Number(value=0, precision=0, label="Max rows (0 = all)")
            load_btn = gr.Button("📥 Load dataset", variant="primary", scale=2)

        status = gr.Markdown()
        detection_panel = gr.Markdown()
        preview = gr.Dataframe(label=f"Preview (first {PREVIEW_ROWS} rows)", interactive=False, wrap=True)

        # ----- events -------------------------------------------------------
        def on_source(src):
            return (gr.update(visible=src == "Hugging Face Hub"),
                    gr.update(visible=src == "Local file"),
                    gr.update(visible=src == "Upload"))

        source.change(on_source, source, [hub_group, local_path, upload])

        def on_search(q):
            results = datalib.search_hub_datasets(q)
            if not results:
                gr.Warning(f"No Hub datasets found for {q!r}")
                return gr.update()
            return gr.update(choices=results, value=results[0])

        search_btn.click(on_search, query, ds_name)
        query.submit(on_search, query, ds_name)

        def on_load(src, name, split_v, subset_v, path, file, n, progress=gr.Progress()):
            limit = int(n) or None
            try:
                if src == "Hugging Face Hub":
                    if not name:
                        return "❌ Choose a dataset first.", gr.update(), ""
                    progress(0.2, desc=f"Downloading {name} …")
                    rows = datalib.load_hub_dataset(name, split_v or "train", subset_v or None, limit)
                    return _ingest(rows, name, is_local=False)
                target = path if src == "Local file" else (file.name if file else "")
                if not target:
                    return "❌ Provide a file first.", gr.update(), ""
                rows = datalib.load_local_dataset(target, limit)
                return _ingest(rows, target, is_local=True)
            except Exception as exc:
                return f"❌ Failed to load dataset: {exc}", gr.update(), ""

        load_btn.click(on_load, [source, ds_name, split, subset, local_path, upload, max_rows],
                       [status, preview, detection_panel])
