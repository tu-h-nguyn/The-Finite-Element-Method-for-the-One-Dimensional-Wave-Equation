"""
Generate the four figures of the report.

    python3 make_figs.py          -> figures/*.pdf, Vietnamese labels (for baocao.tex)
    python3 make_figs.py --web    -> docs/figures/*.svg, English labels

Both modes plot exactly the same computed data; only the text and the container
format differ, so the web page cannot drift away from the report.
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fem_wave import assemble, growth, solve, thomas, tri_mul  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, RED, GREEN = "#1f4e9c", "#c0392b", "#1e8449"

HERE = os.path.dirname(os.path.abspath(__file__))

# Every string that differs between the report and the web page.
VI = {
    "l2err": r"Sai số $L^2$", "h1err": r"Sai số $H^1$",
    "l2title": r"Chuẩn $L^2$ — bậc 2", "h1title": r"Nửa chuẩn $H^1$ — bậc 1",
    "exact": "Chính xác", "stable": "ổn định",
    "energy_title": "Bao trên của năng lượng rời rạc ($N=80$)",
    "energy_y": r"$\max_{m\leq n}\,|\mathcal{E}^{m+1/2}|\,/\,|\mathcal{E}^{1/2}|$",
    "full": "Khối lượng đầy đủ", "lumped": "Khối lượng gộp",
    "stab_title": r"Ngưỡng ổn định thực nghiệm ($N=160$, $c=1$)",
    "anim_early": "Hai chu kỳ đầu", "anim_late": "Sau 18 chu kỳ",
    "anim_env": "biên độ bảo toàn: đỉnh vẫn chạm $\\pm 1$",
}
EN = {
    "l2err": r"$L^2$ error", "h1err": r"$H^1$ error",
    "l2title": r"$L^2$ norm — order 2", "h1title": r"$H^1$ semi-norm — order 1",
    "exact": "Exact", "stable": "stable",
    "energy_title": "Discrete energy envelope ($N=80$)",
    "energy_y": r"$\max_{m\leq n}\,|\mathcal{E}^{m+1/2}|\,/\,|\mathcal{E}^{1/2}|$",
    "full": "Full mass", "lumped": "Lumped mass",
    "stab_title": r"Measured stability threshold ($N=160$, $c=1$)",
    "anim_early": "First two periods", "anim_late": "After eighteen periods",
    "anim_env": "amplitude conserved: the peak still reaches $\\pm 1$",
}


def snapshots(N, theta, times, c=1.0):
    (Td, Tl), (Sd, Sl), h = assemble(N, False)
    dt = theta * h
    x = np.linspace(0, 1, N + 1)
    U0 = np.sin(2 * np.pi * x[1:-1])
    A0 = thomas(Td, Tl, -(c ** 2) * tri_mul(Sd, Sl, U0))
    Um, Uc = U0, U0 + 0.5 * dt ** 2 * A0
    out, k = {}, 1
    targets = sorted(times)
    if 0.0 in targets:
        out[0.0] = np.concatenate(([0], U0, [0]))
    while targets and k < 10 ** 7:
        Un = 2 * Uc - Um + dt ** 2 * thomas(Td, Tl, -(c ** 2) * tri_mul(Sd, Sl, Uc))
        Um, Uc = Uc, Un
        k += 1
        t = k * dt
        while targets and t >= targets[0] - 1e-12:
            out[targets.pop(0)] = np.concatenate(([0], Uc, [0]))
        if not targets:
            break
    return x, out


def energy_hist(N, theta, T=20.0, c=1.0):
    (Td, Tl), (Sd, Sl), h = assemble(N, False)
    dt = theta * h
    M = int(T / dt)
    x = np.linspace(0, 1, N + 1)[1:-1]
    U0 = np.sin(2 * np.pi * x)
    A0 = thomas(Td, Tl, -(c ** 2) * tri_mul(Sd, Sl, U0))
    Um, Uc = U0, U0 + 0.5 * dt ** 2 * A0
    E, tt = [], []
    for k in range(1, M):
        D = (Uc - Um) / dt
        e = D @ tri_mul(Td, Tl, D) + c ** 2 * (Uc @ tri_mul(Sd, Sl, Um))
        if not np.isfinite(e) or abs(e) > 1e30:
            break
        E.append(e)
        tt.append(k * dt)
        Un = 2 * Uc - Um + dt ** 2 * thomas(Td, Tl, -(c ** 2) * tri_mul(Sd, Sl, Uc))
        Um, Uc = Uc, Un
        if not np.isfinite(Uc).all():
            break
    return np.array(tt), np.array(E)


def fig_convergence(L, save):
    Ns = np.array([10, 20, 40, 80, 160, 320])
    hs = 1.0 / Ns
    L2, H1 = [], []
    for N in Ns:
        a, b = solve(N, theta=0.5)
        L2.append(a)
        H1.append(b)
    L2, H1 = np.array(L2), np.array(H1)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax[0].loglog(hs, L2, "o-", color=BLUE, mfc="w", label=r"$\|u-\hat U\|_{L^2}$")
    ax[0].loglog(hs, L2[0] * (hs / hs[0]) ** 2, "k--", lw=1, label=r"$\mathcal{O}(h^2)$")
    ax[0].set_xlabel("$h$")
    ax[0].set_ylabel(L["l2err"])
    ax[0].set_title(L["l2title"], fontsize=9)
    ax[1].loglog(hs, H1, "s-", color=RED, mfc="w", label=r"$|u-\hat U|_{H^1}$")
    ax[1].loglog(hs, H1[0] * (hs / hs[0]), "k--", lw=1, label=r"$\mathcal{O}(h)$")
    ax[1].set_xlabel("$h$")
    ax[1].set_ylabel(L["h1err"])
    ax[1].set_title(L["h1title"], fontsize=9)
    for a in ax:
        a.legend(frameon=False, fontsize=8)
        a.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_convergence")
    plt.close(fig)
    return L2, H1


def fig_solution(L, save):
    ts = [0.0, 0.125, 0.5, 20.0]
    x, snap = snapshots(20, 0.5, ts)
    xf = np.linspace(0, 1, 400)
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.0), sharey=True)
    for a, t in zip(axs, ts):
        a.plot(xf, np.cos(2 * np.pi * t) * np.sin(2 * np.pi * xf), "--", color=RED, lw=1.4,
               label=L["exact"])
        a.plot(x, snap[t], "-o", color=BLUE, ms=2.5, lw=1.2, label="FEM")
        a.set_title(f"$t={t}$", fontsize=9)
        a.set_xlabel("$x$")
        a.set_ylim(-1.35, 1.35)
    axs[0].set_ylabel("$u(x,t)$")
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "fig_solution")
    plt.close(fig)


def fig_energy(L, save):
    # The running maximum, not the raw ratio. Once the scheme goes unstable the
    # dominant mode flips sign every step, so the cross term in E cancels the
    # kinetic term exactly at some steps and E passes through zero -- and a zero
    # on a log axis is a full-height vertical stroke. Hundreds of those render as
    # a solid block. The envelope carries the same claim and stays readable.
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    for th, col, ls, lab in [(0.50, BLUE, "-", r"$\theta=0.50$ (%s)" % L["stable"]),
                             (0.5770, GREEN, "--", r"$\theta=0.5770<\theta^*$"),
                             (0.5800, RED, "-", r"$\theta=0.5800>\theta^*$")]:
        t, E = energy_hist(80, th)
        # dashed green sits on top of solid blue: below the threshold the two are
        # equal to 5e-15, so without distinct styles only one curve is visible.
        ax.plot(t, np.maximum.accumulate(np.abs(E) / abs(E[0])),
                color=col, lw=1.4, ls=ls, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel(L["energy_y"], fontsize=8)
    ax.set_title(L["energy_title"], fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_energy")
    plt.close(fig)


def fig_stability(L, save):
    th = np.linspace(0.40, 1.25, 120)
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    # Past the threshold the amplitude reaches 1e148, so it has to be capped for
    # display. Capping at the old 1e20 put the cap at the top of the axis and the
    # curve read as a line stuck to the frame; 1e12 with headroom above reads as
    # what it is, a curve leaving the plot.
    ceil = 1e12
    for lump, col, ls, lab in [(False, BLUE, "-", L["full"]), (True, GREEN, "--", L["lumped"])]:
        g = [min(growth(160, t, T=1.0, lumped=lump), ceil) for t in th]
        ax.semilogy(th, g, color=col, lw=1.7, ls=ls, label=lab)
    ax.axvline(1 / np.sqrt(3), color=BLUE, ls=":", lw=1.2)
    ax.axvline(1.0, color=GREEN, ls=":", lw=1.2)
    ax.text(1 / np.sqrt(3) - 0.015, 1e6, r"$1/\sqrt{3}$", color=BLUE, ha="right", fontsize=8)
    ax.text(1.015, 1e6, r"$1$", color=GREEN, fontsize=8)
    ax.set_xlabel(r"$\theta=\Delta t/h$")
    ax.set_ylabel(r"$\max|U^M|$")
    ax.set_title(L["stab_title"], fontsize=9)
    ax.set_ylim(1e-1, ceil * 40)
    ax.legend(frameon=False, fontsize=8, loc="center")
    fig.tight_layout()
    save(fig, "fig_stability")
    plt.close(fig)


def main():
    web = "--web" in sys.argv
    if web:
        outdir = os.path.join(HERE, "docs", "figures")
        labels, ext = EN, "svg"
    else:
        outdir = os.path.join(HERE, "figures")
        labels, ext = VI, "pdf"
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    def save(fig, name):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))

    L2, H1 = fig_convergence(labels, save)
    fig_solution(labels, save)
    fig_energy(labels, save)
    fig_stability(labels, save)
    if web:  # a PDF cannot animate; the report keeps the four static snapshots
        fig_solution_animation(labels, outdir)

    print(f"Wrote 4 figures to {outdir}")
    print("L2:", ["%.4e" % v for v in L2])
    print("H1:", ["%.4e" % v for v in H1])


# ----------------------------------------------------------------------
# Animated solution (web only — a PDF cannot move)
# ----------------------------------------------------------------------
def frames_in_windows(N, theta, windows, T, c=1.0):
    """Step the scheme once and keep the states whose time lands in a window.

    Frames are real computed states, one per time step, not interpolated: the
    animation shows what the scheme did, at the resolution it did it.
    """
    (Td, Tl), (Sd, Sl), h = assemble(N, False)
    dt = theta * h
    M = int(round(T / dt))
    x = np.linspace(0, 1, N + 1)
    U0 = np.sin(2 * np.pi * x[1:-1])
    A0 = thomas(Td, Tl, -(c ** 2) * tri_mul(Sd, Sl, U0))
    Um, Uc = U0, U0 + 0.5 * dt ** 2 * A0

    keep = {w: [] for w in windows}

    def record(t, Uint):
        for w in windows:
            if w[0] - 1e-9 <= t <= w[1] + 1e-9:
                keep[w].append((t, np.concatenate(([0.0], Uint, [0.0]))))

    # U_n is the state at time n*dt, so record before stepping, not after: the
    # first version recorded U_1 and labelled it t=0, shifting every frame by a
    # step and making the computed curve look wrong wherever the exact one
    # crosses zero.
    record(0.0, U0)
    record(dt, Uc)
    for n in range(2, M + 1):
        Un = 2 * Uc - Um + dt ** 2 * thomas(Td, Tl, -(c ** 2) * tri_mul(Sd, Sl, Uc))
        Um, Uc = Uc, Un
        record(n * dt, Un)
    return x, keep


def fig_solution_animation(L, outdir, N=20, theta=0.5, c=1.0):
    """u(x,t) against the exact solution, early and after twenty periods.

    Two windows of the same run play side by side. On the left the computed
    solution sits on the exact one; on the right, twenty periods later, the
    amplitude is unchanged and the curves have drifted apart in phase. That is
    the claim the report makes about this scheme, animated rather than asserted.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    early, late = (0.0, 2.0), (18.0, 20.0)
    x, keep = frames_in_windows(N, theta, [early, late], T=20.0, c=c)
    # Every second step: still real computed states, never interpolated, but half
    # the frames and roughly half the bytes. Twenty frames per period is smooth.
    fa, fb = keep[early][::2], keep[late][::2]
    n = min(len(fa), len(fb))
    xf = np.linspace(0, 1, 400)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    lines = []
    for ax, ttl in zip(axes, (L["anim_early"], L["anim_late"])):
        # Without these, the right panel reads as amplitude decay, which is the
        # opposite of what happens: the lagged curve is simply away from its own
        # peak at any given instant. The guides show it still reaches +-1.
        for lvl in (-1.0, 1.0):
            ax.axhline(lvl, color="#999999", ls=":", lw=0.9, zorder=0)
        ex, = ax.plot(xf, np.sin(2 * np.pi * xf), "--", color=RED, lw=1.5, label=L["exact"])
        fe, = ax.plot(x, np.concatenate(([0], np.sin(2 * np.pi * x[1:-1]), [0])),
                      "-o", color=BLUE, ms=3, lw=1.3, label="FEM")
        ax.set_ylim(-1.35, 1.35)
        ax.set_xlabel("$x$")
        ax.set_title(ttl, fontsize=9)
        ax.grid(True, alpha=0.25)
        lines.append((ex, fe))
    axes[0].set_ylabel("$u(x,t)$")
    axes[0].legend(frameon=False, fontsize=8, loc="lower center", ncol=2)
    axes[1].text(0.015, 1.06, L["anim_env"], fontsize=7.5, color="#777777")
    clock = fig.text(0.5, 0.01, "", ha="center", fontsize=8, color="#555555",
                     family="DejaVu Sans Mono")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    def draw(k):
        out = []
        for (ex, fe), frames in zip(lines, (fa, fb)):
            t, U = frames[k]
            ex.set_ydata(np.cos(2 * np.pi * c * t) * np.sin(2 * np.pi * xf))
            fe.set_ydata(U)
            out += [ex, fe]
        clock.set_text(f"t = {fa[k][0]:5.2f}          t = {fb[k][0]:5.2f}")
        return out + [clock]

    anim = FuncAnimation(fig, draw, frames=n, interval=55, blit=False)
    path = os.path.join(outdir, "fig_solution.gif")
    # 96 dpi puts the natural width near the ~670px the page renders it at, so
    # it is not upscaled on a desktop, while staying inside half a megabyte.
    anim.save(path, writer=PillowWriter(fps=18), dpi=96)
    plt.close(fig)
    print(f"  fig_solution.gif  {n} frames, {os.path.getsize(path) // 1024} KB")

if __name__ == "__main__":
    main()
