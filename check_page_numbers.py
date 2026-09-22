"""
Check that every number in the tables of docs/index.html matches the solver.

The figures on the page are generated, so they cannot drift. The tables are
hand-written HTML, so they can — a typo there is a wrong claim on a public
page. This parses the four data tables out of the page and compares every cell
against a fresh computation from fem_wave.py.

    python3 check_page_numbers.py       (from the repo root)

Exits non-zero on any mismatch.
"""
import os
import re
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reproduce_tables as rt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "docs", "index.html")

RTOL = 1e-3          # the page prints 5 significant figures
ORDER_ATOL = 0.005   # orders are printed to 2 decimals


def norm(text):
    """HTML cell -> plain text, with typographic minus and entities resolved."""
    text = re.sub(r"<[^>]+>", "", text)
    text = (text.replace("&minus;", "-").replace("&nbsp;", " ")
                .replace("−", "-").replace("—", "—"))
    return unicodedata.normalize("NFKC", text).strip()


def num(cell):
    """Parse a numeric cell; None for an em-dash placeholder or a non-number."""
    t = norm(cell)
    if t in {"—", "-", ""}:
        return None
    t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


# The caption sits in a <p class="tcap"> immediately before the scroll box, not
# in a <caption> inside the table: inside, it inherited the table's width and
# was clipped at the viewport edge on a phone. This pairs the two back up.
TABLE_RE = re.compile(
    r'<p class="tcap">(?P<cap>.*?)</p>\s*'
    r'<div class="table-scroll">\s*<table>(?P<tbl>.*?)</table>',
    re.S,
)


def tables(html):
    """Yield (caption, [[cell,...],...]) for every captioned table on the page."""
    for m in TABLE_RE.finditer(html):
        body = re.search(r"<tbody>(.*?)</tbody>", m.group("tbl"), re.S)
        if not body:
            continue
        # the mobile-only "swipe the table sideways" hint is not part of the caption
        cap = re.sub(r'<span class="swipe">.*?</span>', "", m.group("cap"), flags=re.S)
        rows = []
        for tr in re.findall(r"<tr>(.*?)</tr>", body.group(1), re.S):
            rows.append(re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S))
        yield norm(cap), rows


def main():
    html = open(PAGE, encoding="utf-8").read()
    found = list(tables(html))
    fails = []
    checked = 0

    def cmp(label, got, want, atol=None):
        nonlocal checked
        if got is None:
            fails.append(f"{label}: page cell is not a number")
            return
        ok = (abs(got - want) <= atol) if atol is not None else \
             (abs(got - want) <= RTOL * abs(want))
        checked += 1
        if not ok:
            fails.append(f"{label}: page says {got:.6g}, solver says {float(want):.6g}")

    # ---- identify the four tables by their caption ------------------------
    by_kind = {}
    for cap, rows in found:
        low = cap.lower()
        if "full mass matrix, " in low and "0.5" in low:
            by_kind["spatial"] = rows
        elif "n = 200 fixed" in low:
            by_kind["temporal"] = rows
        elif "amplitude max" in low:
            by_kind["stability"] = rows
        elif "lumped mass matrix" in low and "0.9" in low:
            by_kind["lumped"] = rows
        elif "closed-form spectra" in low:
            by_kind["spectra"] = rows

    for want_key in ("spatial", "temporal", "stability", "lumped"):
        if want_key not in by_kind:
            fails.append(f"table '{want_key}' not found on the page")
    if fails:
        print("\n".join(fails))
        sys.exit(1)

    # ---- Table 1: N, h, dt, L2, order, H1, order --------------------------
    for row, (N, h, dt, l2, p, h1, q) in zip(by_kind["spatial"], rt.table1(), strict=True):
        cmp(f"T1 N={N} h", num(row[1]), h)
        cmp(f"T1 N={N} dt", num(row[2]), dt)
        cmp(f"T1 N={N} L2", num(row[3]), l2)
        cmp(f"T1 N={N} H1", num(row[5]), h1)
        if p is not None:
            cmp(f"T1 N={N} L2 order", num(row[4]), p, atol=ORDER_ATOL)
            cmp(f"T1 N={N} H1 order", num(row[6]), q, atol=ORDER_ATOL)

    # ---- Table 2: theta, dt, diff, order ----------------------------------
    for row, (t, dt, d, p) in zip(by_kind["temporal"], rt.table2(), strict=True):
        cmp(f"T2 theta={t} dt", num(row[1]), dt)
        cmp(f"T2 theta={t} diff", num(row[2]), d)
        if p is not None:
            cmp(f"T2 theta={t} order", num(row[3]), p, atol=ORDER_ATOL)

    # ---- Table 3: theta, N=80, N=160 --------------------------------------
    for row, (t, g80, g160) in zip(by_kind["stability"], rt.table3(), strict=True):
        cmp(f"T3 theta={t} N=80", num(row[1]), g80)
        cmp(f"T3 theta={t} N=160", num(row[2]), g160)

    # ---- Table 4: N, h, L2, order, H1, order ------------------------------
    for row, (N, h, l2, p, h1, q) in zip(by_kind["lumped"], rt.table4(), strict=True):
        cmp(f"T4 N={N} h", num(row[1]), h)
        cmp(f"T4 N={N} L2", num(row[2]), l2)
        cmp(f"T4 N={N} H1", num(row[4]), h1)
        if p is not None:
            cmp(f"T4 N={N} L2 order", num(row[3]), p, atol=ORDER_ATOL)
            cmp(f"T4 N={N} H1 order", num(row[5]), q, atol=ORDER_ATOL)

    # ---- the thresholds quoted in prose and in the stability caption ------
    th = rt.thresholds()
    for label, want in [("theta*(N=80)", th[80]), ("theta*(N=160)", th[160])]:
        want_s = f"{want:.6f}"
        if want_s not in html:
            fails.append(f"{label}: page does not carry the computed value {want_s}")
        else:
            checked += 1

    inv_sqrt3 = f"{1 / np.sqrt(3):.6f}"
    if inv_sqrt3 not in html:
        fails.append(f"1/sqrt(3) limit {inv_sqrt3} not quoted on the page")
    else:
        checked += 1

    if fails:
        print(f"{len(fails)} MISMATCH(ES) between docs/index.html and the solver:\n")
        for f in fails:
            print("  -", f)
        sys.exit(1)

    print(f"docs/index.html: {checked} numbers checked against the solver, all agree.")


if __name__ == "__main__":
    main()
