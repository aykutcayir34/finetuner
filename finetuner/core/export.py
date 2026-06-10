"""Export trained models: adapters, merged weights, GGUF, or push to the Hub."""

from __future__ import annotations


def save_adapters(model, path: str) -> str:
    model.save_pretrained(path)
    return f"LoRA adapters saved to {path}"


def save_merged(model, tokenizer, path: str) -> str:
    model.save_pretrained_merged(path, tokenizer)
    return f"Merged 16-bit model saved to {path}"


def save_gguf(model, tokenizer, path: str) -> str:
    # mlx-lm limitation: GGUF export requires a non-quantized base model.
    model.save_pretrained_gguf(path, tokenizer)
    return f"GGUF model saved to {path}"


def push_to_hub(model, repo_id: str, token: str | None = None) -> str:
    kwargs = {"token": token} if token else {}
    try:
        model.push_to_hub(repo_id, **kwargs)
    except TypeError:
        model.push_to_hub(repo_id)
    return f"Pushed to https://huggingface.co/{repo_id}"
