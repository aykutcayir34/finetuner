"""Shared application state for the (single-user, local) Studio session."""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection
from .training import RunConfig


@dataclass
class AppState:
    # Model
    model = None
    tokenizer = None
    model_name: str = ""
    model_loaded_for_task: str = ""
    lora_attached: bool = False
    # Dataset
    raw_rows: list[dict] = field(default_factory=list)
    detection: Detection | None = None
    dataset_source: str = ""
    dataset_is_local: bool = False
    # Run configuration
    config: RunConfig = field(default_factory=RunConfig)

    def reset_model(self):
        self.model = None
        self.tokenizer = None
        self.model_name = ""
        self.model_loaded_for_task = ""
        self.lora_attached = False


STATE = AppState()
