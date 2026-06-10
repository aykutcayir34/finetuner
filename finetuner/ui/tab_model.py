"""Model tab: pick a task, find a model (Hub search or local path), load it, attach LoRA."""

from __future__ import annotations

import gradio as gr

from ..core import models
from ..core.registry import get_task, mlx_available, task_choices
from ..core.state import STATE


def _task_info(task_id: str) -> str:
    spec = get_task(task_id)
    lines = [f"**{spec.label}**", "", spec.description, "",
             f"- Backend: `mlx_tune.{spec.trainer}` + `{spec.config}`",
             f"- Dataset schema: `{', '.join(spec.dataset_schema)}`"]
    if spec.notes:
        lines.append(f"- ⚠️ {spec.notes}")
    return "\n".join(lines)


def build(app):
    with gr.Tab("🧠 Model", id="model"):
        gr.Markdown("### 1 · Choose a task and a base model")
        with gr.Row():
            with gr.Column(scale=1):
                task = gr.Dropdown(choices=task_choices(), value="sft", label="Training task",
                                   info="Every mlx-tune trainer is available here.")
                task_info = gr.Markdown(_task_info("sft"))
            with gr.Column(scale=2):
                source = gr.Radio(["Hugging Face Hub", "Local path"], value="Hugging Face Hub",
                                  label="Model source")
                with gr.Group() as hub_group:
                    with gr.Row():
                        query = gr.Textbox(label="Search the Hub",
                                           placeholder="e.g. llama 3.2 instruct 4bit", scale=3)
                        search_btn = gr.Button("🔍 Search", scale=1)
                    model_name = gr.Dropdown(label="Model", allow_custom_value=True,
                                             value=get_task("sft").default_model,
                                             choices=[get_task("sft").default_model],
                                             info="Pick a search result or type any repo id.")
                local_path = gr.Textbox(label="Local model directory", visible=False,
                                        placeholder="/path/to/converted-mlx-model")
                with gr.Row():
                    max_seq = gr.Slider(256, 32768, value=2048, step=256, label="Max sequence length")
                    four_bit = gr.Checkbox(value=True, label="Load in 4-bit")

        gr.Markdown("### 2 · LoRA adapters")
        with gr.Row():
            use_lora = gr.Checkbox(value=True, label="Attach LoRA", scale=1)
            lora_r = gr.Slider(1, 256, value=16, step=1, label="Rank (r)", scale=2)
            lora_alpha = gr.Slider(1, 256, value=16, step=1, label="Alpha", scale=2)
            lora_dropout = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="Dropout", scale=2)
        target_modules = gr.CheckboxGroup(
            choices=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            value=["q_proj", "k_proj", "v_proj", "o_proj"],
            label="Target modules (text models)")

        load_btn = gr.Button("⚡ Load model", variant="primary")
        status = gr.Markdown()

        # ----- events -------------------------------------------------------
        def on_task(task_id):
            spec = get_task(task_id)
            return (_task_info(task_id),
                    gr.update(value=spec.default_model, choices=[spec.default_model]),
                    gr.update(value=list(spec.default_target_modules)))

        task.change(on_task, task, [task_info, model_name, target_modules])

        def on_source(src):
            hub = src == "Hugging Face Hub"
            return gr.update(visible=hub), gr.update(visible=not hub)

        source.change(on_source, source, [hub_group, local_path])

        def on_search(q):
            results = models.search_hub_models(q)
            if not results:
                gr.Warning(f"No Hub models found for {q!r}")
                return gr.update()
            return gr.update(choices=results, value=results[0])

        search_btn.click(on_search, query, model_name)
        query.submit(on_search, query, model_name)

        def on_load(task_id, src, name, path, seq, fourbit,
                    lora, r, alpha, dropout, targets, progress=gr.Progress()):
            ok, reason = mlx_available()
            if not ok:
                return f"❌ {reason}"
            resolved = name
            try:
                if src == "Local path":
                    resolved = models.validate_local_model(path)
                progress(0.1, desc=f"Loading {resolved} …")
                model, tok = models.load_model(task_id, resolved, int(seq), bool(fourbit))
                if lora and get_task(task_id).peft_supported:
                    progress(0.7, desc="Attaching LoRA adapters …")
                    model = models.apply_lora(task_id, model, int(r), int(alpha),
                                              float(dropout), list(targets))
                STATE.model, STATE.tokenizer = model, tok
                STATE.model_name = resolved
                STATE.model_loaded_for_task = task_id
                STATE.lora_attached = bool(lora)
                cfg = STATE.config
                cfg.task, cfg.model_name = task_id, resolved
                cfg.max_seq_length, cfg.load_in_4bit = int(seq), bool(fourbit)
                cfg.use_lora, cfg.lora_r, cfg.lora_alpha = bool(lora), int(r), int(alpha)
                cfg.lora_dropout, cfg.target_modules = float(dropout), list(targets)
                return (f"✅ **{resolved}** loaded for **{get_task(task_id).label}**"
                        + (" with LoRA attached." if lora else "."))
            except Exception as exc:  # surfaced to the user, not crashed
                return f"❌ Load failed: {exc}"

        load_btn.click(
            on_load,
            [task, source, model_name, local_path, max_seq, four_bit,
             use_lora, lora_r, lora_alpha, lora_dropout, target_modules],
            status)

    return {"task": task}
