"""Export tab: adapters, merged weights, GGUF, and Hugging Face Hub upload."""

from __future__ import annotations

import gradio as gr

from ..core import export
from ..core.state import STATE


def _guard():
    if STATE.model is None:
        raise gr.Error("No model in memory — load and train one first.")


def build(app):
    with gr.Tab("📦 Export", id="export"):
        gr.Markdown("### Save or publish the fine-tuned model")

        with gr.Group():
            gr.Markdown("**LoRA adapters** — small, fast to share")
            with gr.Row():
                adapter_path = gr.Textbox(value="lora_model", label="Directory", scale=3)
                adapter_btn = gr.Button("💾 Save adapters", scale=1)

        with gr.Group():
            gr.Markdown("**Merged model** — base weights + adapters fused to 16-bit")
            with gr.Row():
                merged_path = gr.Textbox(value="merged", label="Directory", scale=3)
                merged_btn = gr.Button("🔗 Save merged", scale=1)

        with gr.Group():
            gr.Markdown("**GGUF** — for llama.cpp / Ollama. "
                        "⚠️ mlx-lm's GGUF writer is currently unreliable (version drift, "
                        "quantized bases unsupported). If this fails, use the proven recipe: "
                        "`mlx_lm.fuse --dequantize` + llama.cpp's `convert_hf_to_gguf.py` "
                        "(see README).")
            with gr.Row():
                gguf_path = gr.Textbox(value="model_gguf", label="Directory", scale=3)
                gguf_btn = gr.Button("🦙 Export GGUF", scale=1)

        with gr.Group():
            gr.Markdown("**Hugging Face Hub** — publish the model to your account")
            with gr.Row():
                repo_id = gr.Textbox(label="Repo id", placeholder="username/my-finetuned-model", scale=2)
                hf_token = gr.Textbox(label="HF token (optional if logged in)", type="password", scale=2)
                push_btn = gr.Button("🤗 Push to Hub", variant="primary", scale=1)

        status = gr.Markdown()

        def run(fn, *args):
            _guard()
            try:
                return f"✅ {fn(*args)}"
            except Exception as exc:
                return f"❌ {exc}"

        adapter_btn.click(lambda p: run(export.save_adapters, STATE.model, p),
                          adapter_path, status)
        merged_btn.click(lambda p: run(export.save_merged, STATE.model, STATE.tokenizer, p),
                         merged_path, status)
        gguf_btn.click(lambda p: run(export.save_gguf, STATE.model, STATE.tokenizer, p),
                       gguf_path, status)
        push_btn.click(lambda r, t: run(export.push_to_hub, STATE.model, r, t or None),
                       [repo_id, hf_token], status)
