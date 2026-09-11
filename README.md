# FEM for the 1D wave equation — sharp CFL analysis, reproduced

P1 finite elements + leapfrog for `u_tt - c² u_xx = 0` on `(0,1)`, homogeneous
Dirichlet. The point of the project is not the scheme — it is the **sharp**
stability constant: the familiar `Δt ≤ h/c` is *not* sufficient when you use a
consistent (full) mass matrix, and the analysis says exactly what is.

**Read the write-up:** [project page](https://tu-h-nguyn.github.io/The-Finite-Element-Method-for-the-One-Dimensional-Wave-Equation/) ·
[full report, 19 pages, Vietnamese (PDF)](docs/report-fem-wave-vi.pdf)

Joint coursework — *Numerical Analysis for PDEs*, Faculty of Mathematics and
Computer Science, VNU-HCM University of Science, July 2026. Report written with
Nông Thanh Toàn.

## The result

Diagonalising the generalised eigenproblem for the pair `(S, T)` (stiffness,
mass) gives a necessary **and sufficient** stability condition

```
Δt < 2 / (c · sqrt(λ_max(T⁻¹S)))
```

and `λ_max` has a closed form for both mass matrices, so the CFL constant falls
out exactly:

| Mass matrix | λ_max | CFL bound | Threshold θ* = Δt/h |
|---|---|---|---|
| Full (consistent) | `(6/h²)·max(1−cos θ_m)/(2+cos θ_m)` | `Δt < h/(c√3)` | 0.5774 |
| Lumped | `(2/h²)·max(1−cos θ_m)` | `Δt < h/c` | 1.0000 |

A factor of `√3 ≈ 1.73` between them — which is why `Δt ≤ h/c` blows up on the
full mass matrix, and why lumping is the better choice here: it keeps the
`O(h²)` order, gives slightly *smaller* L² error, and turns each solve into a
scalar division.

Measured against the analysis: the empirical threshold matches θ* to **four
significant figures** (§7.3), and the observed orders are `O(h²)` in L² and
`O(h)` in the H¹ semi-norm, on the nose.

## Reproduce it

```bash
pip install numpy matplotlib

python3 reproduce_tables.py    # recomputes Tables 1-4, checks them, writes results/tables.md
python3 make_figs.py           # figures/*.pdf  — the report's figures (Vietnamese labels)
python3 make_figs.py --web     # docs/figures/*.svg — the web page's figures
```

`reproduce_tables.py` recomputes every number in the report's four tables from
`fem_wave.py`, compares each against the value printed in the PDF, and **exits
non-zero on a mismatch**. Current state: all four tables agree to every printed
digit. Its output is checked in at [`results/tables.md`](results/tables.md).

### One gap it caught

The first run reproduced Tables 1, 2 and 4 exactly but *not* Table 3 — the
stability sweep stayed bounded at `θ = 0.578` and `θ = 0.580`, where the report
records divergence. The cause: §7.3 seeds the highest eigenmode with amplitude
`1e-8` so the unstable mode is visible immediately, but the `growth()` listing
in Appendix A did not do that. Initial data `sin(2πx)` contains almost none of
mode `m = N−1`, so without the seed you have to wait for round-off to amplify
it. `growth()` now takes `seed=1e-8` (set `seed=0.0` for the old behaviour), and
all six rows of Table 3 reproduce to four digits. The appendix pulls the file
with `\lstinputlisting`, so the report picks the fix up on its next compile.

## Files

```
baocao.tex            report source, 19 pages (Vietnamese)
fem_wave.py           the solver: assembly, Thomas O(N), errors, stability sweep
make_figs.py          the four figures — PDF for the report, SVG for the web page
reproduce_tables.py   recompute + verify Tables 1-4
check_page_numbers.py cross-check every number on the page against the solver
figures/*.pdf         report figures (regenerable)
results/tables.md     verified output of reproduce_tables.py
make_og.py            renders docs/og.png, the 1200x630 social-share card
docs/                 the published page: index.html, figures/*.svg, og.png,
                      sitemap.xml, robots.txt, the report PDF
```

CI (`.github/workflows/verify.yml`) runs both checkers on every push, so a
change that breaks the numbers fails the build rather than sitting unnoticed.
The page's figures are generated from the solver, but its tables are written by
hand — `check_page_numbers.py` re-derives all 91 of those cells, which is the
one place a wrong public claim could otherwise creep in.

The compiled PDF is served from the site at
[`docs/report-fem-wave-vi.pdf`](docs/report-fem-wave-vi.pdf); a local `xelatex`
build writes `baocao.pdf` at the repo root, which is gitignored.

## Building the report

Requires **XeLaTeX** (not pdfLaTeX) and the **Liberation Serif** font.

```bash
xelatex baocao.tex && xelatex baocao.tex && xelatex baocao.tex   # 3 passes for the TOC
# Ubuntu/Debian: sudo apt-get install texlive-xetex texlive-lang-other \
#                                     texlive-science fonts-liberation
```

Two preamble choices are **deliberate** — don't revert them:

1. **`polyglossia`, not `babel`.** Under XeLaTeX,
   `\usepackage[vietnamese]{babel}` fails with `Unknown option`.
2. **No `unicode-math`.** It remaps the maths font family and breaks the
   extensible brace glyphs in `\underbrace` (they render as black boxes). The
   classic `amsmath + amssymb + bm` stack is fully compatible.

`cleveref` was removed for the same reason — it does not get along with
Vietnamese polyglossia; cross-references are written out ("Định lý~\ref{...}").

## Where it stops

One space dimension, uniform mesh, constant `c`, homogeneous Dirichlet data, P1
elements, explicit leapfrog. The sharp constant is exact under those
assumptions and nothing here claims more: a variable coefficient, a non-uniform
mesh or higher-order elements each change `λ_max`, so the constant would have to
be recomputed — the method for getting it does carry over.

---

## Tiếng Việt — tóm tắt

Báo cáo môn *Giải tích số cho Phương trình Đạo hàm riêng* (ĐH KHTN, ĐHQG-HCM),
FEM `P1` + leapfrog cho phương trình sóng một chiều. Đóng góp chính là **phân
tích ổn định sắc nét**: tính tường minh phổ của cặp `(S, T)` để thu được điều
kiện ổn định cần và đủ, từ đó ra hai hằng số CFL khác nhau — `Δt < h/(c√3)` cho
khối lượng đầy đủ và `Δt < h/c` cho khối lượng gộp.

Phần bổ sung khi đưa dự án lên portfolio: `reproduce_tables.py` tính lại toàn bộ
số liệu của 4 bảng từ `fem_wave.py` và **đối chiếu với giá trị in trong PDF** —
sai lệch là báo lỗi. Nhờ đó phát hiện Bảng 3 không tái lập được do listing
`growth()` ở Phụ lục A thiếu bước kích thích mode `1e-8` mà Mục 7.3 có mô tả;
đã thêm tham số `seed` và cả 4 bảng khớp tới từng chữ số.
