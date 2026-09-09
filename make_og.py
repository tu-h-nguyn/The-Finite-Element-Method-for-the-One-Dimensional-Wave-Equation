#!/usr/bin/env python3
"""Render docs/og.png — the 1200x630 social-share card for the project page.

Drawn in the browser rather than in matplotlib so the card uses the page's own
typography and palette. Fonts are named with concrete fallbacks that exist on a
bare Linux box, so the render is deterministic without network access.

    pip install playwright && python3 make_og.py
"""
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "docs" / "og.png"

CARD = """
<style>
  @page { margin:0 }
  *{box-sizing:border-box; margin:0}
  body{width:1200px; height:630px; background:#0c1620; color:#e8eef4;
       font-family:"Liberation Sans","DejaVu Sans",system-ui,sans-serif;
       display:flex; flex-direction:column; justify-content:space-between;
       padding:70px 78px 62px; position:relative; overflow:hidden}
  .glow{position:absolute; width:780px; height:780px; right:-300px; top:-340px; border-radius:50%;
        background:radial-gradient(circle,rgba(56,152,236,.20),rgba(56,152,236,0) 68%)}
  .eyebrow{font-family:"DejaVu Sans Mono",monospace; font-size:19px;
           letter-spacing:.16em; text-transform:uppercase; color:#8398ad}
  h1{font-weight:700; font-size:78px; line-height:1.06; letter-spacing:-.03em;
     color:#fff; margin-top:16px}
  .deck{font-size:25px; color:#b9c8d6; max-width:36ch; margin-top:24px; line-height:1.45}
  .eq{font-family:"DejaVu Sans Mono",monospace; font-size:27px; color:#6cb6f5;
      margin-top:26px; letter-spacing:-.01em}
  .stats{display:flex; gap:52px; align-items:flex-end;
         border-top:1px solid #22303e; padding-top:24px}
  .n{font-family:"DejaVu Sans Mono",monospace; font-size:34px; font-weight:700; color:#6cb6f5}
  .l{font-size:16px; color:#8398ad; margin-top:5px}
  .by{margin-left:auto; text-align:right; font-size:16px; color:#8398ad; line-height:1.5}
</style>
<div class="glow"></div>
<div>
  <p class="eyebrow">Numerical analysis &middot; FEM for the wave equation</p>
  <h1>&Delta;t &le; h/c is not enough</h1>
  <p class="deck">The consistent mass matrix needs a step &radic;3 times smaller
     than the textbook bound &mdash; and the exact spectrum says so.</p>
  <p class="eq">&Delta;t &lt; h/(c&radic;3) &nbsp;full mass&nbsp; &middot; &nbsp;&Delta;t &lt; h/c &nbsp;lumped</p>
</div>
<div class="stats">
  <div><div class="n">4 s.f.</div><div class="l">theory vs. measured threshold</div></div>
  <div><div class="n">2.00 / 1.00</div><div class="l">observed order, L&sup2; / H&sup1;</div></div>
  <div><div class="n">4 / 4</div><div class="l">tables machine-verified</div></div>
  <div class="by">Nguy&#7877;n Ho&agrave;ng T&uacute;<br>July 2026</div>
</div>
"""


def chromium():
    for p in sorted(pathlib.Path("/opt/pw-browsers").glob("chromium*/chrome-linux/chrome")):
        return str(p)
    direct = pathlib.Path("/opt/pw-browsers/chromium")
    if direct.exists():
        return str(direct)
    return shutil.which("chromium") or shutil.which("google-chrome")


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright is not installed — pip install playwright")

    with sync_playwright() as pw:
        exe = chromium()
        browser = pw.chromium.launch(executable_path=exe) if exe else pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 630},
                                device_scale_factor=1)
        page.set_content(CARD, wait_until="load")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(OUT))
        browser.close()

    print(f"{OUT.relative_to(ROOT)}  {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
