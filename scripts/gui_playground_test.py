"""Drive the GUI: load the fully-trained merged model from a LOCAL PATH,
then ask Turkish questions in the Playground. One-off session script."""

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
MERGED = str(Path("outputs/turkish-full/merged").resolve())
SHOTS = Path("docs/screenshots")

QUESTIONS = [
    "Sağlıklı yaşam için üç öneri ver.",
    "Aşağıdaki cümleyi İngilizceye çevir: Bugün hava çok güzel.",
]


def wait_for_text(page, pattern, timeout_s, poll_s=2.0):
    rx = re.compile(pattern, re.DOTALL)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = rx.search(page.inner_text("body"))
        if m:
            return m.group(0)
        time.sleep(poll_s)
    raise TimeoutError(pattern)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(URL)
        page.wait_for_selector("text=Finetuner Studio", timeout=30_000)

        # Model tab: switch to local path, uncheck LoRA (inference only)
        page.get_by_role("tab", name="Model").click()
        page.get_by_role("radio", name="Local path").check()
        page.get_by_label("Local model directory").fill(MERGED)
        lora = page.get_by_role("checkbox", name="Attach LoRA")
        if lora.is_checked():
            lora.uncheck()
        page.get_by_role("button", name="Load model").click()
        msg = wait_for_text(page, r"(✅.*loaded for.*|❌ .*)", 600, poll_s=5)
        print(f"[load] {msg[:120]}", flush=True)
        assert msg.startswith("✅"), msg
        page.screenshot(path=str(SHOTS / "07_local_model_loaded.png"), full_page=True)

        page.get_by_role("tab", name="Playground").click()
        for i, q in enumerate(QUESTIONS, 1):
            chat = page.locator("textarea").last
            chat.fill(q)
            chat.press("Enter")
            time.sleep(3)
            # wait until a new bot bubble appears with >10 chars after our question
            deadline = time.time() + 300
            answer = ""
            while time.time() < deadline:
                bubbles = page.locator(".bot, .message.bot, [data-testid='bot']").all()
                if bubbles:
                    answer = bubbles[-1].inner_text().strip()
                    if len(answer) > 10:
                        break
                time.sleep(3)
            print(f"[Q{i}] {q}", flush=True)
            print(f"[A{i}] {answer[:300]}", flush=True)
        page.screenshot(path=str(SHOTS / "08_playground_full_model.png"), full_page=True)
        browser.close()
        print("PLAYGROUND TEST DONE ✅", flush=True)


if __name__ == "__main__":
    sys.exit(main())
