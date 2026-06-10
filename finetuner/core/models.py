"""Model discovery (Hugging Face Hub) and loading via mlx-tune."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

from .registry import get_task, resolve


def search_hub_models(query: str, limit: int = 20, mlx_only: bool = True) -> list[str]:
    """Search the Hub; by default biased to mlx-community / MLX-tagged models."""
    if not query.strip():
        return []
    api = HfApi()
    kwargs: dict = {"search": query, "limit": limit, "sort": "downloads"}
    if mlx_only:
        kwargs["library"] = "mlx"
    results = [m.id for m in api.list_models(**kwargs)]
    if not results and mlx_only:  # fall back to an unrestricted search
        results = [m.id for m in api.list_models(search=query, limit=limit, sort="downloads")]
    return results


def validate_local_model(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(f"Not a directory: {p}")
    if not (p / "config.json").exists():
        raise ValueError(f"{p} does not look like a model directory (missing config.json).")
    return str(p)


def load_model(task_id: str, model_name: str, max_seq_length: int = 2048,
               load_in_4bit: bool = True):
    """Load a model + tokenizer/processor through the task's mlx-tune loader."""
    spec = get_task(task_id)
    loader = resolve(spec.loader)
    kwargs: dict = {}
    if spec.modality == "text":
        kwargs["max_seq_length"] = max_seq_length
        kwargs["load_in_4bit"] = load_in_4bit
    return loader.from_pretrained(model_name, **kwargs)


def apply_lora(task_id: str, model, r: int = 16, lora_alpha: int = 16,
               lora_dropout: float = 0.0, target_modules: list[str] | None = None,
               **extra):
    spec = get_task(task_id)
    loader = resolve(spec.loader)
    kwargs: dict = {"r": r, "lora_alpha": lora_alpha}
    if lora_dropout:
        kwargs["lora_dropout"] = lora_dropout
    if spec.modality == "text" and target_modules:
        kwargs["target_modules"] = list(target_modules)
    kwargs.update(extra)
    try:
        return loader.get_peft_model(model, **kwargs)
    except TypeError:
        # Older mlx-tune versions may not accept every kwarg (e.g. lora_dropout).
        kwargs.pop("lora_dropout", None)
        return loader.get_peft_model(model, **kwargs)
