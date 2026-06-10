"""Automatic dataset format detection and normalization.

Given a few sample rows, classify the dataset into one of the canonical
formats mlx-tune trainers understand, propose a column mapping, and convert
rows into the exact shape the selected trainer expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Column-name synonyms (English + Turkish), matched case-insensitively after
# stripping whitespace — real-world CSV headers are messy.
SYNONYMS = {
    "instruction": {"instruction", "question", "query", "instruct", "task",
                    "talimat", "soru", "görev"},
    "input": {"input", "context", "system", "giriş", "girdi", "bağlam"},
    "output": {"output", "response", "answer", "completion", "target",
               "çıktı", "cevap", "yanıt"},
    "prompt": {"prompt", "question", "query", "instruction", "istem", "soru"},
    "chosen": {"chosen", "preferred", "accepted", "good", "seçilen", "tercih"},
    "rejected": {"rejected", "dispreferred", "bad", "reddedilen"},
    "completion": {"completion", "response", "output", "answer", "cevap", "yanıt"},
    "label": {"label", "thumbs_up", "is_good", "score", "etiket"},
    "conversations": {"conversations", "messages", "dialogue", "dialog", "chat",
                      "turns", "konuşmalar", "mesajlar"},
    "text": {"text", "content", "document", "body", "metin", "içerik"},
    "anchor": {"anchor", "query", "sentence1", "question", "soru"},
    "positive": {"positive", "passage", "sentence2", "answer", "document"},
    "audio": {"audio", "audio_path", "audio_filepath", "wav", "file", "path", "ses"},
    "transcription": {"text", "transcription", "transcript", "sentence", "caption",
                      "metin", "cümle"},
    "image": {"image", "images", "image_path", "img", "picture", "görsel", "resim"},
}

FORMAT_LABELS = {
    "alpaca": "Alpaca (instruction / input / output)",
    "sharegpt": "ShareGPT (conversations with from/value turns)",
    "chatml": "ChatML / OpenAI messages (role/content turns)",
    "prompt_completion": "Prompt–completion pairs",
    "preference": "Preference pairs (prompt / chosen / rejected)",
    "kto": "KTO binary feedback (prompt / completion / label)",
    "grpo": "GRPO prompts (prompt, optional answer)",
    "text": "Raw text corpus",
    "embedding_pairs": "Embedding pairs (anchor / positive)",
    "audio_text": "Audio + text (TTS / STT)",
    "vision_chat": "Image + conversation (VLM)",
    "image_text": "Image + ground-truth text (OCR)",
    "unknown": "Unknown — manual mapping required",
}


@dataclass
class Detection:
    format: str
    confidence: float                      # 0..1
    mapping: dict[str, str] = field(default_factory=dict)   # canonical -> actual column
    suggested_tasks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return FORMAT_LABELS.get(self.format, self.format)


def _find(columns: list[str], canonical: str) -> str | None:
    """Find the actual column matching a canonical field name."""
    lowered = {c.strip().lower(): c for c in columns}
    if canonical in lowered:
        return lowered[canonical]
    for syn in SYNONYMS.get(canonical, ()):
        if syn in lowered:
            return lowered[syn]
    return None


def _is_turn_list(value) -> str | None:
    """Classify a list-of-dicts column as 'sharegpt' or 'chatml' turns."""
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        return None
    keys = set(value[0].keys())
    if {"from", "value"} <= keys:
        return "sharegpt"
    if {"role", "content"} <= keys:
        return "chatml"
    return None


def _looks_like_path(value, exts: tuple[str, ...]) -> bool:
    return isinstance(value, str) and value.lower().endswith(exts)


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")


def detect(rows: list[dict]) -> Detection:
    """Detect the dataset format from sample rows (a handful is enough)."""
    if not rows:
        return Detection("unknown", 0.0, notes=["Dataset is empty."])

    row = rows[0]
    columns = list(row.keys())
    from .registry import tasks_for_format  # local import to avoid cycles

    def done(fmt: str, conf: float, mapping: dict[str, str], notes: list[str] | None = None) -> Detection:
        tasks = [t.id for t in tasks_for_format(fmt)]
        return Detection(fmt, conf, mapping, tasks, notes or [])

    # --- multimodal first: presence of media columns dominates -------------
    img_col = _find(columns, "image")
    audio_col = _find(columns, "audio")
    conv_col = _find(columns, "conversations")

    if img_col is not None:
        sample = row.get(img_col)
        media_like = _looks_like_path(sample, IMAGE_EXTS) or not isinstance(sample, str)
        if conv_col is not None:
            return done("vision_chat", 0.9 if media_like else 0.6,
                        {"images": img_col, "messages": conv_col})
        text_col = _find([c for c in columns if c != img_col], "text")
        if text_col is not None:
            return done("image_text", 0.85 if media_like else 0.55,
                        {"image": img_col, "text": text_col},
                        ["Image + text detected — suitable for OCR SFT or vision tasks."])

    if audio_col is not None:
        sample = row.get(audio_col)
        if _looks_like_path(sample, AUDIO_EXTS) or isinstance(sample, dict):
            text_col = _find([c for c in columns if c != audio_col], "transcription")
            if text_col is not None:
                return done("audio_text", 0.9, {"audio": audio_col, "text": text_col},
                            ["Audio + text detected — choose TTS (synthesis) or STT (recognition)."])

    # --- conversation formats ----------------------------------------------
    if conv_col is not None:
        kind = _is_turn_list(row.get(conv_col))
        if kind == "sharegpt":
            return done("sharegpt", 0.95, {"conversations": conv_col})
        if kind == "chatml":
            return done("chatml", 0.95, {"messages": conv_col})

    # --- preference / feedback ----------------------------------------------
    chosen = _find(columns, "chosen")
    rejected = _find(columns, "rejected")
    prompt = _find(columns, "prompt")
    if chosen and rejected:
        mapping = {"chosen": chosen, "rejected": rejected}
        if prompt:
            mapping["prompt"] = prompt
            return done("preference", 0.95, mapping)
        return done("preference", 0.75, mapping,
                    ["No explicit prompt column; chosen/rejected may embed the prompt."])

    label = _find(columns, "label")
    completion = _find(columns, "completion")
    if prompt and completion and label is not None:
        if isinstance(row.get(label), (bool, int)):
            return done("kto", 0.9, {"prompt": prompt, "completion": completion, "label": label})

    # --- instruction tuning ---------------------------------------------------
    instruction = _find(columns, "instruction")
    output = _find(columns, "output")
    if instruction and output:
        mapping = {"instruction": instruction, "output": output}
        inp = _find([c for c in columns if c not in (instruction, output)], "input")
        if inp:
            mapping["input"] = inp
        return done("alpaca", 0.95, mapping)

    if prompt and completion:
        return done("prompt_completion", 0.9, {"prompt": prompt, "completion": completion})

    # --- embeddings -----------------------------------------------------------
    anchor = _find(columns, "anchor")
    positive = _find(columns, "positive")
    if anchor and positive and anchor != positive:
        return done("embedding_pairs", 0.8, {"anchor": anchor, "positive": positive},
                    ["Anchor/positive pair detected — embedding contrastive training."])

    # --- GRPO: bare prompts -----------------------------------------------------
    if prompt and len(columns) <= 2:
        return done("grpo", 0.6, {"prompt": prompt},
                    ["Bare prompts — usable for GRPO with a custom reward function."])

    # --- raw text ----------------------------------------------------------------
    text = _find(columns, "text")
    if text:
        return done("text", 0.85, {"text": text},
                    ["Raw text — suitable for CPT or completion-style SFT."])

    # --- single string column fallback ---------------------------------------------
    str_cols = [c for c in columns if isinstance(row.get(c), str)]
    if len(str_cols) == 1:
        return done("text", 0.5, {"text": str_cols[0]},
                    [f"Single text column `{str_cols[0]}` assumed to be raw text."])

    return Detection("unknown", 0.0, {},
                     notes=[f"Could not classify columns: {columns}. Map fields manually."])


# ---------------------------------------------------------------------------
# Normalization: convert detected rows into trainer-ready shape
# ---------------------------------------------------------------------------

def _format_chat(turns: list[dict], tokenizer=None) -> str:
    """Render chat turns to text via the tokenizer's chat template when possible."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return "\n".join(f"<|{t['role']}|>\n{t['content']}" for t in turns) + "\n"


SHAREGPT_ROLES = {"human": "user", "user": "user", "gpt": "assistant",
                  "assistant": "assistant", "system": "system"}


def to_messages(row: dict, detection: Detection) -> list[dict]:
    """Convert a row of any chat-like format into ChatML messages."""
    m = detection.mapping
    fmt = detection.format
    if fmt == "chatml":
        return row[m["messages"]]
    if fmt == "sharegpt":
        return [{"role": SHAREGPT_ROLES.get(t["from"], "user"), "content": t["value"]}
                for t in row[m["conversations"]]]
    if fmt == "alpaca":
        user = row[m["instruction"]]
        if "input" in m and row.get(m["input"]):
            user = f"{user}\n\n{row[m['input']]}"
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": row[m["output"]]}]
    if fmt == "prompt_completion":
        return [{"role": "user", "content": row[m["prompt"]]},
                {"role": "assistant", "content": row[m["completion"]]}]
    raise ValueError(f"Cannot build messages from format {fmt!r}")


def normalize(rows: list[dict], detection: Detection, task_id: str, tokenizer=None) -> list[dict]:
    """Convert raw rows into the schema the chosen task's trainer expects."""
    m = detection.mapping
    fmt = detection.format

    if task_id in ("sft",):
        if fmt == "text":
            return [{"text": r[m["text"]]} for r in rows]
        return [{"text": _format_chat(to_messages(r, detection), tokenizer)} for r in rows]

    if task_id == "cpt":
        col = m.get("text")
        if col is None:
            raise ValueError("CPT needs a raw text column.")
        return [{"text": r[col]} for r in rows]

    if task_id in ("dpo", "orpo", "simpo"):
        out = []
        for r in rows:
            item = {"chosen": r[m["chosen"]], "rejected": r[m["rejected"]]}
            item["prompt"] = r[m["prompt"]] if "prompt" in m else ""
            out.append(item)
        return out

    if task_id == "kto":
        return [{"prompt": r[m["prompt"]], "completion": r[m["completion"]],
                 "label": bool(r[m["label"]])} for r in rows]

    if task_id == "grpo":
        return [{"prompt": r[m["prompt"]]} for r in rows]

    if task_id == "embedding":
        return [{"anchor": r[m["anchor"]], "positive": r[m["positive"]]} for r in rows]

    if task_id in ("tts_sft", "stt_sft"):
        return [{"audio": r[m["audio"]], "text": r[m["text"]]} for r in rows]

    if task_id == "ocr_sft":
        return [{"image": r[m["image"]], "text": r[m["text"]]} for r in rows]

    if task_id == "vlm_sft":
        out = []
        for r in rows:
            sub = {"images": r[m["images"]], "messages": r[m["messages"]]}
            if isinstance(sub["messages"], list) and sub["messages"] \
                    and "from" in (sub["messages"][0] or {}):
                sub["messages"] = [{"role": SHAREGPT_ROLES.get(t["from"], "user"),
                                    "content": t["value"]} for t in sub["messages"]]
            out.append(sub)
        return out

    raise ValueError(f"Unknown task {task_id!r}")
