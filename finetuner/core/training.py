"""Build and run mlx-tune trainers from a flat GUI config dict."""

from __future__ import annotations

from dataclasses import dataclass, field

from .jobs import Job
from .registry import get_task, resolve


@dataclass
class RunConfig:
    """Everything the GUI collects, flattened into one serializable object."""
    task: str = "sft"
    model_name: str = ""
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    # Training hyperparameters
    output_dir: str = "outputs"
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    max_steps: int = 100
    num_train_epochs: float | None = None
    warmup_steps: int = 5
    gradient_checkpointing: bool = False
    seed: int = 42
    # Task-specific extras (beta for DPO, temperature for embeddings, ...)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _filtered_kwargs(config_cls, kwargs: dict) -> dict:
    """Drop kwargs the dataclass config doesn't accept (version drift safety)."""
    fields = getattr(config_cls, "__dataclass_fields__", None)
    if fields is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in fields}


def build_trainer_args(cfg: RunConfig) -> dict:
    spec = get_task(cfg.task)
    args: dict = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "warmup_steps": cfg.warmup_steps,
        "seed": cfg.seed,
    }
    if cfg.num_train_epochs:
        args["num_train_epochs"] = cfg.num_train_epochs
    else:
        args["max_steps"] = cfg.max_steps
    if cfg.gradient_checkpointing:
        args["gradient_checkpointing"] = True
    args.update(spec.extra_config_defaults)
    args.update(cfg.extra)
    return args


def run_training(job: Job, cfg: RunConfig, model, tokenizer, dataset: list[dict]):
    """Job target: construct the task's trainer and train. Runs on a worker thread."""
    spec = get_task(cfg.task)
    trainer_cls = resolve(spec.trainer)
    config_cls = resolve(spec.config, spec.config_module)

    args = config_cls(**_filtered_kwargs(config_cls, build_trainer_args(cfg)))

    trainer_kwargs: dict = {"model": model, "train_dataset": dataset, "args": args}
    if spec.modality == "vision" or spec.id == "ocr_sft":
        trainer_kwargs["processor"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if spec.collator:
        collator_cls = resolve(spec.collator)
        trainer_kwargs["data_collator"] = collator_cls(model, tokenizer)

    try:
        trainer = trainer_cls(**trainer_kwargs)
    except TypeError:
        # Some trainers take `processor` instead of `tokenizer` or vice versa.
        if "tokenizer" in trainer_kwargs:
            trainer_kwargs["processor"] = trainer_kwargs.pop("tokenizer")
        else:
            trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processor")
        trainer = trainer_cls(**trainer_kwargs)

    job.add_log(f"▶ {spec.label} started — {len(dataset)} samples, output → {cfg.output_dir}")
    trainer.train()
    job.add_log("✅ Training finished.")
