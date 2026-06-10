"""Drive the Finetuner Studio GUI end-to-end with Playwright:
Llama-3.2-1B-Instruct-4bit + TFLai/Turkish-Alpaca, 60-step LoRA SFT.

Assumes the Studio is already running at http://127.0.0.1:7860.
Saves screenshots to docs/screenshots/.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
SHOTS = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)

MODEL_ID = "mlx-community/Llama-3.2-1B-Instruct-4bit"
DATASET_ID = "TFLai/Turkish-Alpaca"
MAX_ROWS = 300
MAX_STEPS = 60


def shot(page, name: str):
    page.screenshot(path=str(SHOTS / name), full_page=True)
    print(f"  📸 {name}")


def wait_for_text(page, pattern: str, timeout_s: int, poll_s: float = 2.0) -> str:
    """Poll the page body until a regex matches; return the matching text."""
    rx = re.compile(pattern)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = page.inner_text("body")
        m = rx.search(body)
        if m:
            return m.group(0)
        time.sleep(poll_s)
    raise TimeoutError(f"pattern {pattern!r} not found within {timeout_s}s")


def set_dropdown(page, label: str, value: str):
    """Set a Gradio dropdown (allow_custom_value) by typing into its input."""
    box = page.get_by_label(label, exact=True)
    box.click()
    box.fill(value)
    box.press("Enter")
    time.sleep(0.5)
    actual = box.input_value()
    assert actual == value, f"dropdown {label!r} holds {actual!r}, wanted {value!r}"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(URL)
        page.wait_for_selector("text=Finetuner Studio", timeout=30_000)
        print("[1/6] Studio loaded")
        shot(page, "01_home.png")

        # ---- Model tab: default model is already the target — verify, load ----
        page.get_by_role("tab", name="Model").click()
        model_box = page.get_by_label("Model", exact=True)
        if model_box.input_value() != MODEL_ID:
            set_dropdown(page, "Model", MODEL_ID)
        print(f"[2/6] Model selected: {model_box.input_value()}")
        page.get_by_role("button", name="Load model").click()
        msg = wait_for_text(page, r"(✅.*loaded for.*|❌ .*)", 1800, poll_s=5)
        print(f"      {msg[:120]}")
        assert msg.startswith("✅"), f"model load failed: {msg}"
        shot(page, "02_model_loaded.png")

        # ---- Dataset tab: type repo id → load → auto-detection ---------------
        page.get_by_role("tab", name="Dataset").click()
        set_dropdown(page, "Dataset", DATASET_ID)
        page.get_by_label("Max rows (0 = all)").fill(str(MAX_ROWS))
        page.get_by_role("button", name="Load dataset").click()
        msg = wait_for_text(page, r"(✅ Loaded.*rows.*|❌ .*)", 600, poll_s=3)
        print(f"[3/6] {msg[:120]}")
        assert msg.startswith("✅"), f"dataset load failed: {msg}"
        det = wait_for_text(page, r"Detected format: .*", 30)
        conf = wait_for_text(page, r"Confidence: .*", 10)
        print(f"      {det[:80]} | {conf[:60]}")
        shot(page, "03_dataset_detected.png")

        # ---- Train tab: set steps → start --------------------------------------
        page.get_by_role("tab", name="Train").click()
        # Gradio sliders expose their number box as "number input for <label>".
        steps_box = page.get_by_label("number input for Max steps")
        steps_box.fill(str(MAX_STEPS))
        steps_box.press("Enter")
        page.get_by_role("button", name="Start training").click()
        msg = wait_for_text(page, r"(🏃.*Job #\d+.*|❌ .*)", 120, poll_s=2)
        print(f"[4/6] {msg[:120]}")
        assert "Job #" in msg, f"training did not start: {msg}"
        shot(page, "04_training_started.png")

        # ---- Monitor: wait for completion ----------------------------------------
        page.get_by_role("tab", name="Monitor").click()
        wait_for_text(page, r"(finished|failed|stopped)", 3600, poll_s=10)
        status = wait_for_text(page, r"Job #\d+.*", 10)
        print(f"[5/6] {status[:140]}")
        time.sleep(3)  # let the final chart tick render
        shot(page, "05_monitor_loss.png")
        assert "finished" in page.inner_text("body"), "training did not finish cleanly"

        # ---- Playground: Turkish question ------------------------------------------
        page.get_by_role("tab", name="Playground").click()
        chat = page.locator("textarea").last
        chat.fill("Türkiye'nin başkenti neresidir? Tek cümleyle cevapla.")
        chat.press("Enter")
        time.sleep(2)
        wait_for_text(page, r"Ankara|başkent", 300, poll_s=3)
        print("[6/6] Playground answered in Turkish")
        shot(page, "06_playground.png")

        browser.close()
        print("\nGUI DRIVE PASSED ✅")


if __name__ == "__main__":
    main()
