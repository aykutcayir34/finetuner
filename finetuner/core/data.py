"""Dataset loading from the Hugging Face Hub or local files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from huggingface_hub import HfApi


def search_hub_datasets(query: str, limit: int = 20) -> list[str]:
    if not query.strip():
        return []
    api = HfApi()
    return [d.id for d in api.list_datasets(search=query, limit=limit, sort="downloads")]


def load_hub_dataset(name: str, split: str = "train", config: str | None = None,
                     max_rows: int | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(name, config or None, split=split)
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    return [dict(r) for r in ds]


def load_local_dataset(path: str, max_rows: int | None = None) -> list[dict]:
    """Load a local dataset file: .jsonl, .json, .csv, .tsv or .parquet."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    suffix = p.suffix.lower()

    rows: list[dict]
    if suffix == ".jsonl":
        rows = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                if max_rows and len(rows) >= max_rows:
                    break
    elif suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict):  # e.g. {"data": [...]} wrappers
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of records.")
        rows = data
    elif suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f, delimiter=delim))
    elif suffix == ".parquet":
        import pandas as pd
        rows = pd.read_parquet(p).to_dict("records")
    else:
        raise ValueError(f"Unsupported dataset format: {suffix} "
                         "(supported: .jsonl, .json, .csv, .tsv, .parquet)")

    if max_rows:
        rows = rows[:max_rows]
    return rows
