"""Render a branding page to an image with headless Chromium.

The page is opened from disk, so its fonts and panels load by relative path, and the first element matching the
selector is saved. It needs Playwright with its Chromium (`uv run --with playwright python -m playwright install
chromium` once).

Usage: uv run --with playwright python branding/render.py branding/poster/poster.html out.png [--selector .poster]
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("page")
    ap.add_argument("out")
    ap.add_argument("--selector", default=".poster")
    args = ap.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--allow-file-access-from-files"])
        page = browser.new_page(viewport={"width": 1500, "height": 2100})
        page.goto(Path(args.page).resolve().as_uri())
        page.wait_for_load_state("networkidle")
        missing = page.evaluate("[...document.images].filter(i => !i.naturalWidth).map(i => i.src)")
        if missing:
            raise SystemExit(f"images that did not load: {missing}")
        page.locator(args.selector).screenshot(path=args.out, quality=90 if args.out.endswith(".jpg") else None)
        browser.close()


if __name__ == "__main__":
    main()
