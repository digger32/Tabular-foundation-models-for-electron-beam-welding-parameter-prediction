#!/usr/bin/env python3
"""fig01_framework: leakage-safe evaluation flow of the EBW-TFM study.

Strictly orthogonal layout; arrow tips and tails sit exactly on the rounded
box borders. Typography matches make_figures.py (DejaVu Sans, base 10 pt).

Machine checks replace eyeballing in a headless environment; ANY failure
aborts the build instead of shipping a defective figure:
  * fit_check     - every label must fit its box AT THE DECLARED SIZE
                    (no auto-shrink, so typography stays uniform);
  * route_check   - no arrow segment may cross any box other than the two
                    it connects;
  * border_check  - the render must show all four borders of every box;
  * contact_check - along each arrow axis the tail-to-border and tip-to-border
                    junctions must be pixel-continuous (max white gap <= 2 px):
                    catches both "arrow falls short" and "arrow penetrates".

Usage:
    python make_fig01_framework.py --out outputs/figures
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

TFM_C = "#f3d9c4"
CLS_C = "#cfe0ee"
NEU_C = "#eeeeee"
DEC_C = "#d9ead3"
GATE_C = "#f4f4f4"
PAD = 0.008
MARGIN = 0.03

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

BOXES = []      # ((x, y, w, h), [text artists])
ARROWS = []     # dict(p0, p1) after end offsets


def box(ax, x, y, w, h, text, fc=NEU_C, lw=0.9, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD}",
                                fc=fc, ec="black", lw=lw, clip_on=False))
    lines = text.split("\n")
    arts = []
    if bold_first and len(lines) > 1:
        arts.append(ax.text(x + w / 2, y + h - 0.024, lines[0], ha="center",
                            va="top", fontsize=10, fontweight="bold"))
        arts.append(ax.text(x + w / 2, y + h / 2 - 0.026, "\n".join(lines[1:]),
                            ha="center", va="center", fontsize=9))
    else:
        arts.append(ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                            fontsize=9))
    BOXES.append(((x, y, w, h), arts))


def arrow(ax, x0, y0, x1, y1):
    """Orthogonal arrow whose tail and tip sit ON the padded borders."""
    assert x0 == x1 or y0 == y1, "orthogonal arrows only"
    if x0 == x1:
        s = 1 if y1 > y0 else -1
        p0, p1 = (x0, y0 + s * PAD), (x1, y1 - s * PAD)
    else:
        s = 1 if x1 > x0 else -1
        p0, p1 = (x0 + s * PAD, y0), (x1 - s * PAD, y1)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color="black", shrinkA=0, shrinkB=0,
                                 clip_on=False))
    ARROWS.append((p0, p1))


# ----------------------------- checks ---------------------------------------

def fit_check(fig, ax):
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for (x, y, w, h), arts in BOXES:
        for t in arts:
            bb = t.get_window_extent(fig.canvas.get_renderer())
            (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            if not (x - 0.004 <= x0 and x1 <= x + w + 0.004 and
                    y - 0.004 <= y0 and y1 <= y + h + 0.004):
                bad.append((t.get_text().split("\n")[0][:45],
                            round(x1 - (x + w), 3), round(y - y0, 3)))
    if bad:
        raise SystemExit(f"fig01: labels overflow at declared size: {bad}")
    print("[check] all labels fit at the declared font sizes")


def _touching(p, rect, tol):
    (bx, by, bw, bh) = rect
    return (bx - tol <= p[0] <= bx + bw + tol and
            by - tol <= p[1] <= by + bh + tol)


def route_check():
    eps = PAD + 0.002
    for p0, p1 in ARROWS:
        for (rect, _) in BOXES:
            if _touching(p0, rect, 2 * PAD) or _touching(p1, rect, 2 * PAD):
                continue                      # the two boxes this arrow connects
            bx, by, bw, bh = rect
            X0, X1, Y0, Y1 = bx - eps, bx + bw + eps, by - eps, by + bh + eps
            (x0, y0), (x1, y1) = p0, p1
            if x0 == x1 and X0 < x0 < X1:
                lo, hi = sorted((y0, y1))
                if max(lo, Y0) < min(hi, Y1):
                    raise SystemExit(f"fig01: arrow {p0}->{p1} crosses box {rect}")
            if y0 == y1 and Y0 < y0 < Y1:
                lo, hi = sorted((x0, x1))
                if max(lo, X0) < min(hi, X1):
                    raise SystemExit(f"fig01: arrow {p0}->{p1} crosses box {rect}")
    print("[check] no arrow crosses a third box")


def _gray(png_path):
    img = plt.imread(png_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img.mean(axis=2)


def _mapper(ax, fig, H):
    trans = ax.transData

    def data_to_png(x, y):
        dx, dy = trans.transform((x, y))
        return int(round(H - dy)), int(round(dx))
    return data_to_png


def border_check(gray, d2p):
    H, W = gray.shape

    def dark_near(r, c, rad=4):
        r0, r1 = max(r - rad, 0), min(r + rad + 1, H)
        c0, c1 = max(c - rad, 0), min(c + rad + 1, W)
        return r0 < r1 and c0 < c1 and gray[r0:r1, c0:c1].min() < 0.45

    for (x, y, w, h), _ in BOXES:
        sides = {"left":   [(x - PAD, y + h * f) for f in (0.3, 0.7)],
                 "right":  [(x + w + PAD, y + h * f) for f in (0.3, 0.7)],
                 "bottom": [(x + w * f, y - PAD) for f in (0.3, 0.7)],
                 "top":    [(x + w * f, y + h + PAD) for f in (0.3, 0.7)]}
        for side, pts in sides.items():
            if not any(dark_near(*d2p(px, py)) for px, py in pts):
                raise SystemExit(f"fig01: {side} border of box ({x:.2f},{y:.2f}) "
                                 f"missing in render")
    print("[check] borders of all boxes present in the render")


def contact_check(gray, d2p):
    """Pixel continuity across every arrow junction (tail and tip).

    Walk along the arrow axis from inside the arrow toward the box; the white
    run between the last dark shaft/head pixel and the first dark border pixel
    must be <= 2 px. The light box fill beyond the border is NOT counted.
    """
    H, W = gray.shape

    def dark_at(rr, cc, vert):
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if vert:
            seg = gray[rr, max(cc - 2, 0):cc + 3]
        else:
            seg = gray[max(rr - 2, 0):rr + 3, cc]
        return seg.min() < 0.55

    for p0, p1 in ARROWS:
        vert = (p0[0] == p1[0])
        for end, other in ((p0, p1), (p1, p0)):
            r, c = d2p(*end)
            # unit step from the box INTO the arrow
            if vert:
                step = 1 if d2p(*other)[0] > r else -1
                samples = [(r + k * step, c) for k in range(-3, 9)]
            else:
                step = 1 if d2p(*other)[1] > c else -1
                samples = [(r, c + k * step) for k in range(-3, 9)]
            ks = list(range(-3, 9))
            flags = [dark_at(rr, cc, vert) for rr, cc in samples]
            dark_ks = [k for k, f in zip(ks, flags) if f]
            if not dark_ks:
                raise SystemExit(f"fig01: no ink at all near junction {end}")
            if not any(abs(k) <= 2 for k in dark_ks):
                raise SystemExit(f"fig01: junction at {end}: nearest ink is "
                                 f"{min(abs(k) for k in dark_ks)} px away "
                                 f"(arrow falls short of the border)")
            span = flags[ks.index(min(dark_ks)):ks.index(max(dark_ks)) + 1]
            gap = best = 0
            for f in span:
                gap = 0 if f else gap + 1
                best = max(best, gap)
            if best > 2:
                raise SystemExit(f"fig01: junction at {end} has a {best}-px white "
                                 f"gap inside the joint")
    print("[check] every arrow junction is pixel-continuous (gap <= 2 px)")


# ------------------------------ layout --------------------------------------

def main(outdir):
    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    ax.set_xlim(-MARGIN, 1 + MARGIN)
    ax.set_ylim(-MARGIN, 1 + MARGIN)
    ax.axis("off")

    box(ax, 0.02, 0.875, 0.44, 0.115,
        "EBW campaign\n72 cross-sections; 18 coupons, 15 schedules;\nfour sections of a coupon share one input")
    box(ax, 0.54, 0.875, 0.44, 0.115,
        "GMAW-1 / GMAW-2\n1006 / 1164 rows; time series\ngrouped by welding run")

    box(ax, 0.02, 0.685, 0.44, 0.115,
        "Leave-one-regime-out\nhold out a whole schedule (15 folds);\nseeds averaged per fold; leak check = 0")
    box(ax, 0.54, 0.685, 0.44, 0.115,
        "Group split by welding run\n80/20, seed-indexed (10 seeds);\nleak check = 0")
    arrow(ax, 0.24, 0.875, 0.24, 0.80)
    arrow(ax, 0.76, 0.875, 0.76, 0.80)

    box(ax, 0.02, 0.525, 0.96, 0.09,
        "Regimes\nfull;  few-shot 50% / 25% (whole schedules or runs removed);  augmentation (classical only)")
    arrow(ax, 0.24, 0.685, 0.24, 0.615)
    arrow(ax, 0.76, 0.685, 0.76, 0.615)

    box(ax, 0.02, 0.33, 0.46, 0.125,
        "Tabular foundation models (GPU)\nTabPFN v2 / v2.5 / v3; in-context, one\nforward pass, no tuning; quantile grid", fc=TFM_C)
    box(ax, 0.52, 0.33, 0.46, 0.125,
        "Tuned classical controls (CPU)\nCatBoost, XGBoost, NGBoost, MLP;\nnested CV (8 x 3); optimism gap logged", fc=CLS_C)
    arrow(ax, 0.24, 0.525, 0.24, 0.455)
    arrow(ax, 0.76, 0.525, 0.76, 0.455)

    box(ax, 0.02, 0.10, 0.58, 0.155,
        "Pooled out-of-fold evaluation\naccuracy + statistics (Friedman, Nemenyi,\nWilcoxon--Holm; 35 blocks); calibration\naudit; conformal CV+ intervals")
    box(ax, 0.665, 0.10, 0.315, 0.155,
        "Acceptance decision\ninterval vs tolerance (one-sided,\nH/B tier); accept / reject / abstain;\nFA and FR separated", fc=DEC_C)
    arrow(ax, 0.24, 0.33, 0.24, 0.255)
    arrow(ax, 0.56, 0.33, 0.56, 0.255)
    arrow(ax, 0.60, 0.178, 0.665, 0.178)

    box(ax, 0.02, 0.0, 0.96, 0.058,
        "Review-proofing gate:  final run without resume  |  leak check = 0 per unit  |  one device config",
        fc=GATE_C, bold_first=False)

    fit_check(fig, ax)
    route_check()

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    chk = outdir / "fig01_framework_check.png"
    fig.savefig(chk, dpi=int(fig.dpi))
    gray = _gray(chk)
    d2p = _mapper(ax, fig, gray.shape[0])
    border_check(gray, d2p)
    contact_check(gray, d2p)
    chk.unlink()

    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig01_framework.{ext}", bbox_inches="tight", dpi=200)
    print(f"[fig] fig01_framework.pdf/.png -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures")
    a = ap.parse_args()
    main(a.out)
