"""Drive the Export tab: load SmolLM2-135M+LoRA, then exercise the local
export paths (adapters / merged / GGUF). Hub push tested separately."""

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
MODEL = "mlx-community/SmolLM2-135M-Instruct"
SHOTS = Path("docs/screenshots")


def last_status(page):
    """Return the last ✅/❌ status line currently on the page."""
    matches = re.findall(r"[✅❌][^\n]*", page.inner_text("body"))
    return matches[-1] if matches else ""


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(URL)
        page.wait_for_selector("text=Finetuner Studio", timeout=30_000)

        # Load small model WITH LoRA so adapter export is meaningful
        page.get_by_role("tab", name="Model").click()
        page.get_by_role("radio", name="Hugging Face Hub").check()
        box = page.get_by_label("Model", exact=True)
        box.click()
        box.fill(MODEL)
        box.press("Enter")
        lora = page.get_by_role("checkbox", name="Attach LoRA")
        if not lora.is_checked():
            lora.check()
        page.get_by_role("button", name="Load model").click()
        deadline = time.time() + 600
        while time.time() < deadline:
            s = last_status(page)
            if "loaded for" in s or s.startswith("❌"):
                break
            time.sleep(3)
        print(f"[load] {last_status(page)[:110]}", flush=True)

        page.get_by_role("tab", name="Export").click()
        results = {}
        for label, field, value, btn in [
            ("adapters", "Directory", "outputs/export-test/adapters", "Save adapters"),
            ("merged", "Directory", "outputs/export-test/merged", "Save merged"),
            ("gguf", "Directory", "outputs/export-test/gguf", "Export GGUF"),
        ]:
            # Each group has its own Directory textbox; match by current value order
            boxes = page.get_by_label("Directory").all()
            idx = {"adapters": 0, "merged": 1, "gguf": 2}[label]
            boxes[idx].fill(value)
            before = last_status(page)
            page.get_by_role("button", name=btn).click()
            deadline = time.time() + 600
            while time.time() < deadline:
                s = last_status(page)
                if s and s != before:
                    break
                time.sleep(2)
            results[label] = last_status(page)
            print(f"[{label}] {results[label][:160]}", flush=True)

        page.screenshot(path=str(SHOTS / "09_export_tests.png"), full_page=True)
        browser.close()
        print("EXPORT LOCAL TESTS DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
