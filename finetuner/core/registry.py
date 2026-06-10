"""Task registry: one entry per mlx-tune training paradigm.

Every public trainer interface that mlx-tune exposes is described here so the
GUI, the code generator and the recipe system all share a single source of
truth. mlx-tune itself is imported lazily — the Studio runs (for planning,
dataset inspection, recipe authoring and script generation) even on machines
where MLX is not installed.
"""

from __future__ import annotations

import importlib
import platform
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    id: str
    label: str
    description: str
    loader: str                      # mlx_tune Fast*Model class name
    trainer: str                     # mlx_tune trainer class name
    config: str                      # mlx_tune config class name
    config_module: str = "mlx_tune"  # module to import the config from
    collator: str | None = None      # optional data collator class name
    dataset_schema: tuple[str, ...] = ()   # canonical required fields
    detector_formats: tuple[str, ...] = ()  # detector format ids that map to this task
    default_model: str = ""
    default_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    peft_supported: bool = True
    extra_config_defaults: dict = field(default_factory=dict)
    modality: str = "text"           # text | vision | audio | image
    notes: str = ""


FULL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

TASKS: dict[str, TaskSpec] = {
    "sft": TaskSpec(
        id="sft",
        label="SFT — Supervised Fine-Tuning",
        description="Instruction tuning of chat/completion LLMs on text or conversation data.",
        loader="FastLanguageModel",
        trainer="SFTTrainer",
        config="SFTConfig",
        dataset_schema=("text",),
        detector_formats=("alpaca", "sharegpt", "chatml", "prompt_completion", "text"),
        default_model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    ),
    "dpo": TaskSpec(
        id="dpo",
        label="DPO — Direct Preference Optimization",
        description="Align a model with human preferences from chosen/rejected pairs.",
        loader="FastLanguageModel",
        trainer="DPOTrainer",
        config="DPOConfig",
        dataset_schema=("prompt", "chosen", "rejected"),
        detector_formats=("preference",),
        default_model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        extra_config_defaults={"beta": 0.1},
    ),
    "orpo": TaskSpec(
        id="orpo",
        label="ORPO — Odds Ratio Preference Optimization",
        description="Reference-free preference optimization; combines SFT and alignment in one pass.",
        loader="FastLanguageModel",
        trainer="ORPOTrainer",
        config="ORPOConfig",
        dataset_schema=("prompt", "chosen", "rejected"),
        detector_formats=("preference",),
        default_model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        extra_config_defaults={"beta": 0.1},
    ),
    "simpo": TaskSpec(
        id="simpo",
        label="SimPO — Simple Preference Optimization",
        description="Length-normalized, reference-free preference optimization.",
        loader="FastLanguageModel",
        trainer="SimPOTrainer",
        config="SimPOConfig",
        dataset_schema=("prompt", "chosen", "rejected"),
        detector_formats=("preference",),
        default_model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    ),
    "kto": TaskSpec(
        id="kto",
        label="KTO — Kahneman-Tversky Optimization",
        description="Alignment from simple binary thumbs-up/down feedback (no pairs needed).",
        loader="FastLanguageModel",
        trainer="KTOTrainer",
        config="KTOConfig",
        dataset_schema=("prompt", "completion", "label"),
        detector_formats=("kto",),
        default_model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    ),
    "grpo": TaskSpec(
        id="grpo",
        label="GRPO — Group Relative Policy Optimization",
        description="Online RL with programmable reward functions (reasoning, math, code).",
        loader="FastLanguageModel",
        trainer="GRPOTrainer",
        config="GRPOConfig",
        dataset_schema=("prompt",),
        detector_formats=("grpo",),
        default_model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        notes="Reward functions are plain Python callables; edit them in the generated script.",
    ),
    "cpt": TaskSpec(
        id="cpt",
        label="CPT — Continual Pretraining",
        description="Inject domain knowledge by continuing pretraining on raw text.",
        loader="FastLanguageModel",
        trainer="CPTTrainer",
        config="CPTConfig",
        dataset_schema=("text",),
        detector_formats=("text",),
        default_model="mlx-community/SmolLM2-360M-Instruct",
        default_target_modules=FULL_TARGETS,
        extra_config_defaults={"embedding_learning_rate": 5e-6, "include_embeddings": True},
    ),
    "vlm_sft": TaskSpec(
        id="vlm_sft",
        label="Vision SFT — Vision-Language Models",
        description="Fine-tune VLMs (Qwen-VL, LLaVA-style) on image + conversation data.",
        loader="FastVisionModel",
        trainer="VLMSFTTrainer",
        config="VLMSFTConfig",
        config_module="mlx_tune.vlm",
        dataset_schema=("images", "messages"),
        detector_formats=("vision_chat",),
        default_model="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        modality="vision",
    ),
    "tts_sft": TaskSpec(
        id="tts_sft",
        label="TTS SFT — Text-to-Speech",
        description="Fine-tune speech synthesis models (Orpheus, OuteTTS, CSM…) on audio+text pairs.",
        loader="FastTTSModel",
        trainer="TTSSFTTrainer",
        config="TTSSFTConfig",
        collator="TTSDataCollator",
        dataset_schema=("audio", "text"),
        detector_formats=("audio_text",),
        default_model="mlx-community/orpheus-3b-0.1-ft-bf16",
        modality="audio",
        notes="Audio training currently supports batch_size=1 (mlx-tune limitation).",
    ),
    "stt_sft": TaskSpec(
        id="stt_sft",
        label="STT SFT — Speech-to-Text",
        description="Fine-tune ASR models (Whisper, Parakeet, Canary…) on audio+transcription pairs.",
        loader="FastSTTModel",
        trainer="STTSFTTrainer",
        config="STTSFTConfig",
        collator="STTDataCollator",
        dataset_schema=("audio", "text"),
        detector_formats=("audio_text",),
        default_model="mlx-community/whisper-tiny-asr-fp16",
        modality="audio",
        notes="Audio training currently supports batch_size=1 (mlx-tune limitation).",
    ),
    "embedding": TaskSpec(
        id="embedding",
        label="Embedding SFT — Sentence Embeddings",
        description="Contrastive fine-tuning of embedding models (anchor/positive pairs).",
        loader="FastEmbeddingModel",
        trainer="EmbeddingSFTTrainer",
        config="EmbeddingSFTConfig",
        dataset_schema=("anchor", "positive"),
        detector_formats=("embedding_pairs",),
        default_model="mlx-community/all-MiniLM-L6-v2-bf16",
        extra_config_defaults={"loss_type": "infonce", "temperature": 0.05},
    ),
    "ocr_sft": TaskSpec(
        id="ocr_sft",
        label="OCR SFT — Optical Character Recognition",
        description="Fine-tune OCR models (DeepSeek-OCR, olmOCR…) on image + ground-truth text.",
        loader="FastOCRModel",
        trainer="OCRSFTTrainer",
        config="OCRSFTConfig",
        dataset_schema=("image", "text"),
        detector_formats=("image_text",),
        default_model="mlx-community/DeepSeek-OCR-8bit",
        modality="image",
        extra_config_defaults={"learning_rate": 5e-5},
    ),
}


def get_task(task_id: str) -> TaskSpec:
    return TASKS[task_id]


def task_choices() -> list[tuple[str, str]]:
    """(label, id) pairs for a Gradio dropdown."""
    return [(spec.label, spec.id) for spec in TASKS.values()]


def tasks_for_format(format_id: str) -> list[TaskSpec]:
    return [spec for spec in TASKS.values() if format_id in spec.detector_formats]


# ---------------------------------------------------------------------------
# Lazy mlx-tune access
# ---------------------------------------------------------------------------

def mlx_available() -> tuple[bool, str]:
    """Whether mlx-tune is importable on this machine, plus a human reason."""
    if platform.machine() != "arm64" or platform.system() != "Darwin":
        return False, "mlx-tune requires an Apple Silicon Mac (arm64/macOS)."
    if importlib.util.find_spec("mlx_tune") is None:
        return False, "mlx-tune is not installed. Run: pip install 'finetuner[mlx]'"
    return True, "mlx-tune is available."


def resolve(name: str, module: str = "mlx_tune"):
    """Import `name` from an mlx_tune module, raising a friendly error if missing."""
    ok, reason = mlx_available()
    if not ok:
        raise RuntimeError(reason)
    mod = importlib.import_module(module)
    try:
        return getattr(mod, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"`{name}` not found in `{module}`. Your mlx-tune version may be too old; "
            "try: pip install -U mlx-tune"
        ) from exc
