# Dataset Format Detection — Specification

*Owner: ML/Data Engineer*

## Goal

Given arbitrary rows (from the Hub, a local file, or an upload), decide
**what kind of dataset this is**, **which mlx-tune trainers can consume it**,
and **how to reshape it** — without asking the user to know format names.

## Canonical formats

| Format id | Signature | Normalizes to | Trainers |
|---|---|---|---|
| `alpaca` | `instruction` + `output` (+ optional `input`) | chat-templated `text` | SFT |
| `sharegpt` | list column with `{from, value}` turns | chat-templated `text` | SFT |
| `chatml` | list column with `{role, content}` turns | chat-templated `text` | SFT |
| `prompt_completion` | `prompt` + `completion` | chat-templated `text` | SFT |
| `preference` | `chosen` + `rejected` (+ `prompt`) | as-is | DPO, ORPO, SimPO |
| `kto` | `prompt` + `completion` + bool/int `label` | `label` cast to bool | KTO |
| `grpo` | bare `prompt` (≤2 columns) | `prompt` only | GRPO |
| `text` | `text`-like column | as-is | CPT, SFT |
| `embedding_pairs` | `anchor` + `positive` | as-is | Embedding |
| `audio_text` | audio path/dict + transcript | `{audio, text}` | TTS, STT |
| `image_text` | image path/object + `text` | `{image, text}` | OCR |
| `vision_chat` | image column + conversation column | `{images, messages}` | VLM-SFT |

## Detection algorithm (`detector.detect`)

Ordered rules — first match wins; order encodes specificity:

1. **Media columns dominate**: an image column with conversations →
   `vision_chat`; with text → `image_text`; audio path + transcript →
   `audio_text`. Value shape (file extension or non-string object) modulates
   confidence (0.55–0.9).
2. **Turn lists**: list-of-dicts with `from/value` → `sharegpt`;
   `role/content` → `chatml` (0.95).
3. **Preference**: `chosen`+`rejected` (0.95 with prompt, 0.75 without —
   noted that the prompt may be embedded).
4. **KTO**: prompt+completion+boolean-ish label (0.9).
5. **Alpaca** (0.95) → **prompt/completion** (0.9).
6. **Embedding pairs** (0.8), **GRPO bare prompts** (0.6).
7. **Raw text** (0.85), single-string-column fallback (0.5).
8. Otherwise `unknown` (0.0) with the column list in the notes.

### Synonym table

Column names are matched case-insensitively against synonym sets, e.g.
`output ∈ {output, response, answer, completion, target}`,
`conversations ∈ {conversations, messages, dialogue, chat, turns}`,
`audio ∈ {audio, audio_path, audio_filepath, wav, …}`. This is what lets a
`question/answer` CSV "just work".

## Normalization (`detector.normalize`)

- Chat-like rows are converted to ChatML messages (`to_messages`), then
  rendered with the **live tokenizer's chat template** when one is loaded,
  falling back to a generic `<|role|>` rendering. This guarantees the text
  matches what the model expects.
- ShareGPT roles are mapped (`human→user`, `gpt→assistant`) — including
  inside `vision_chat` messages.
- KTO labels are cast to `bool`; preference rows always carry a `prompt` key
  (empty string when absent) for trainer compatibility.

## Known limitations / future work

- Multi-positive or hard-negative embedding datasets (only anchor/positive
  pairs today).
- HF `Audio`/`Image` feature decoding relies on the underlying `datasets`
  objects; raw bytes columns are classified with reduced confidence.
- No statistical content sniffing yet (e.g. detecting code vs prose) — column
  names and value shapes only.
