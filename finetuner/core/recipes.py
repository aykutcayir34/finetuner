"""Recipes: save/load complete run configurations as shareable YAML files."""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from .training import RunConfig

RECIPE_DIR = Path("recipes")


def save_recipe(cfg: RunConfig, name: str = "", dataset_source: str = "",
                dataset_is_local: bool = False) -> Path:
    RECIPE_DIR.mkdir(exist_ok=True)
    slug = (name.strip() or f"{cfg.task}-{time.strftime('%Y%m%d-%H%M%S')}").replace(" ", "-")
    path = RECIPE_DIR / f"{slug}.yaml"
    payload = {
        "finetuner_recipe": 1,
        "dataset": {"source": dataset_source, "local": dataset_is_local},
        "run": cfg.to_dict(),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return path


def load_recipe(path: str) -> tuple[RunConfig, str, bool]:
    data = yaml.safe_load(Path(path).expanduser().read_text())
    if not isinstance(data, dict) or "run" not in data:
        raise ValueError("Not a Finetuner recipe (missing `run` section).")
    ds = data.get("dataset", {})
    return RunConfig.from_dict(data["run"]), ds.get("source", ""), bool(ds.get("local", False))


def list_recipes() -> list[str]:
    if not RECIPE_DIR.exists():
        return []
    return sorted(str(p) for p in RECIPE_DIR.glob("*.yaml"))
