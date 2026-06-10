from finetuner.core.detector import detect, normalize


def test_alpaca():
    rows = [{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}]
    det = detect(rows)
    assert det.format == "alpaca"
    assert det.mapping == {"instruction": "instruction", "output": "output", "input": "input"}
    assert "sft" in det.suggested_tasks
    norm = normalize(rows, det, "sft")
    assert "Bonjour" in norm[0]["text"]


def test_alpaca_synonyms():
    rows = [{"question": "2+2?", "answer": "4"}]
    det = detect(rows)
    assert det.format in ("alpaca", "prompt_completion")
    assert det.confidence >= 0.8


def test_sharegpt():
    rows = [{"conversations": [{"from": "human", "value": "hi"},
                               {"from": "gpt", "value": "hello"}]}]
    det = detect(rows)
    assert det.format == "sharegpt"
    msgs = normalize(rows, det, "sft")
    assert "hello" in msgs[0]["text"]


def test_chatml():
    rows = [{"messages": [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "hello"}]}]
    det = detect(rows)
    assert det.format == "chatml"


def test_preference():
    rows = [{"prompt": "Q", "chosen": "good", "rejected": "bad"}]
    det = detect(rows)
    assert det.format == "preference"
    assert {"dpo", "orpo", "simpo"} <= set(det.suggested_tasks)
    norm = normalize(rows, det, "dpo")
    assert norm[0] == {"chosen": "good", "rejected": "bad", "prompt": "Q"}


def test_kto():
    rows = [{"prompt": "Q", "completion": "A", "label": True}]
    det = detect(rows)
    assert det.format == "kto"
    assert normalize(rows, det, "kto")[0]["label"] is True


def test_raw_text():
    rows = [{"text": "Some corpus."}]
    det = detect(rows)
    assert det.format == "text"
    assert "cpt" in det.suggested_tasks


def test_embedding_pairs():
    rows = [{"anchor": "query", "positive": "doc"}]
    det = detect(rows)
    assert det.format == "embedding_pairs"
    assert normalize(rows, det, "embedding")[0] == {"anchor": "query", "positive": "doc"}


def test_audio_text():
    rows = [{"audio": "clip1.wav", "sentence": "merhaba"}]
    det = detect(rows)
    assert det.format == "audio_text"
    norm = normalize(rows, det, "stt_sft")
    assert norm[0] == {"audio": "clip1.wav", "text": "merhaba"}


def test_image_text_ocr():
    rows = [{"image": "scan.png", "text": "Invoice #42"}]
    det = detect(rows)
    assert det.format == "image_text"
    assert "ocr_sft" in det.suggested_tasks


def test_vision_chat():
    rows = [{"images": "cat.jpg",
             "messages": [{"role": "user", "content": "What is this?"}]}]
    det = detect(rows)
    assert det.format == "vision_chat"


def test_empty():
    det = detect([])
    assert det.format == "unknown"
    assert det.confidence == 0.0
