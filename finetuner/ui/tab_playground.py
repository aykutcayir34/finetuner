"""Playground tab: chat with the currently loaded (and freshly tuned) model."""

from __future__ import annotations

import gradio as gr

from ..core.state import STATE


def _generate(message: str, history: list[dict], max_tokens: int, temperature: float) -> str:
    if STATE.model is None or STATE.tokenizer is None:
        return "⚠️ No model loaded — load one in the 🧠 Model tab first."

    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": message})

    tok = STATE.tokenizer
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(m["content"] for m in messages)

    # mlx-tune models are mlx-lm compatible; prefer its generate().
    try:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
        return mlx_generate(STATE.model, tok, prompt=prompt, max_tokens=int(max_tokens),
                            sampler=make_sampler(temp=float(temperature)), verbose=False)
    except Exception:
        pass
    try:  # older mlx-lm signature
        from mlx_lm import generate as mlx_generate
        return mlx_generate(STATE.model, tok, prompt=prompt,
                            max_tokens=int(max_tokens), verbose=False)
    except Exception as exc:
        return f"⚠️ Generation failed: {exc}"


def build(app):
    with gr.Tab("💬 Playground", id="playground"):
        gr.Markdown("### Test the loaded model — before and after fine-tuning")
        with gr.Row():
            max_tokens = gr.Slider(16, 4096, value=512, step=16, label="Max new tokens")
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
        gr.ChatInterface(
            fn=_generate,
            additional_inputs=[max_tokens, temperature],
            examples=[["Merhaba! Kendini tanıtır mısın?"],
                      ["Explain LoRA fine-tuning in two sentences."]],
        )
