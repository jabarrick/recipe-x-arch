#!/usr/bin/env python3
"""Figures 1-4 of Section 4, drawn to the specifications fixed in the Results template
(V0.1, 2026-11-08) before the data were seen. Reads the result files only.

Usage:
  python -m analysis.figures_section4 --res results/confirmatory --ana <analysis dir> --out <dir> [--tag V0.1]
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from .common import ARCHS, RECIPES, HOME, SESOI, load_long, seed_average, to_cube

# Validated categorical palette (light): blue, orange, aqua, yellow, fixed order by architecture.
COL = dict(zip(ARCHS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
MARK = dict(zip(ARCHS, ["o", "s", "^", "D"]))          # secondary encoding
CAT = {"A": "#2a78d6", "B": "#eb6834", "E": "#1baf7a", "I": "#9a9a96"}
CATNAME = {"A": "first better", "B": "second better", "E": "equivalent", "I": "inconclusive"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b67", "#e6e6e3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def ci95(x):
    x = x[~np.isnan(x)]
    return x.mean(), stats.t.ppf(.975, len(x) - 1) * stats.sem(x)


def fig1(df, out):
    avg = seed_average(df)
    flag = df.groupby(["subject", "architecture", "recipe"], observed=True).training_failure.max().rename("flag").reset_index()
    flag["architecture"] = flag["architecture"].astype(str); flag["recipe"] = flag["recipe"].astype(str)
    avg = avg.assign(architecture=avg.architecture.astype(str), recipe=avg.recipe.astype(str)).merge(
        flag, on=["subject", "architecture", "recipe"], how="left")
    fig, axs = plt.subplots(4, 4, figsize=(7.2, 7.2), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for i, a in enumerate(ARCHS):
        for j, r in enumerate(RECIPES):
            ax = axs[i, j]; c = avg[(avg.architecture == a) & (avg.recipe == r)]
            for ds, mk in (("Lee2019_MI", "o"), ("Cho2017", "^")):
                cc = c[c.dataset == ds]
                for fl, fc in ((0, "none"), (1, INK)):
                    v = cc[cc.flag == fl].y.to_numpy()
                    ax.scatter(rng.uniform(-.3, .3, len(v)), v, s=7 if not fl else 9, marker=mk,
                               facecolor=fc if fl else "none", edgecolor=INK if fl else COL[a],
                               linewidth=.6, alpha=.75 if not fl else .9)
            m, h = ci95(c.y.to_numpy())
            ax.errorbar([0.45], [m], yerr=[h], fmt="o", color=INK, ms=4, capsize=2, lw=1.2)
            ax.axhline(50, color=MUTED, lw=.8, ls="--")
            ax.text(.97, .04, f"{m:.1f}", transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=8)
            ax.set_xticks([]); ax.set_ylim(30, 100); ax.grid(axis="y", color=GRID, lw=.5)
            if i == 0: ax.set_title(r + (" (home)" if False else ""), color=INK)
            if j == 0: ax.set_ylabel(a, color=INK)
            if HOME.get(a) == r:
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=INK, lw=1.8,
                                           clip_on=False))
    # legend text is in the caption
    fig.supylabel("Balanced accuracy, float32 (%)", color=INK, fontsize=9)
    fig.tight_layout(rect=(0.02, 0.02, 1, 1)); fig.savefig(out); plt.close(fig)


def fig2(df, out):
    avg = seed_average(df); y, _, _ = to_cube(avg)
    fig, (ax, hx) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    x = np.arange(4)
    for i, a in enumerate(ARCHS):
        m = np.nanmean(y[:, i], 0); h = np.array([ci95(y[:, i, j])[1] for j in range(4)])
        ax.errorbar(x + (i - 1.5) * .06, m, yerr=h, color=COL[a], marker=MARK[a], ms=5, lw=2, capsize=2,
                    mec="white", mew=.8, label=a)
        nudge = {"EEGNet": .9, "MI-BMInet": -.6}.get(a, 0)
        ax.text(3.15, m[-1] + nudge, a, color=INK, fontsize=7.5, va="center")
    ax.set_xticks(x, RECIPES); ax.set_xlim(-.3, 4.3); ax.set_ylabel("Cell mean, float32 (%) ± 95 % CI")
    ax.grid(axis="y", color=GRID, lw=.5); ax.legend(frameon=False, fontsize=7, loc="lower left")
    t = np.nanmean(y, 0); r = t - t.mean(1, keepdims=True) - t.mean(0, keepdims=True) + t.mean()
    lim = np.abs(r).max()
    im = hx.imshow(r, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(4):
        for j in range(4):
            hx.text(j, i, f"{r[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(r[i, j]) > .6 * lim else INK)
            if HOME.get(ARCHS[i]) == RECIPES[j]:
                hx.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.8))
    hx.set_xticks(range(4), RECIPES); hx.set_yticks(range(4), ARCHS)
    for s in hx.spines.values(): s.set_visible(False)
    hx.set_title("Interaction residuals (points); framed = home", fontsize=8, color=INK)
    fig.colorbar(im, ax=hx, fraction=.046, pad=.04)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig3(df, out):
    y, subj, ds = to_cube(seed_average(df))
    e, v = ARCHS.index("EEGNet"), ARCHS.index("CSOANet-v2")
    c = (y[:, e, 1] - y[:, e, 0]) - (y[:, v, 1] - y[:, v, 0])
    fig, ax = plt.subplots(figsize=(4.2, 3.2)); rng = np.random.default_rng(1)
    ax.axhspan(-SESOI, SESOI, color=GRID, zorder=0); ax.axhline(0, color=MUTED, lw=.8)
    for k, (d, col, mk) in enumerate((("Lee2019_MI", COL["EEGNet"], "o"), ("Cho2017", COL["ShallowConvNet"], "^"))):
        cc = c[np.asarray(ds) == d]
        ax.scatter(k + rng.uniform(-.18, .18, len(cc)), cc, s=12, marker=mk, color=col, alpha=.7, ec="white", lw=.4)
        m, h = ci95(cc); ax.errorbar(k + .3, m, yerr=h, fmt="o", color=INK, capsize=3)
        ax.text(k + .36, m, f"{m:+.2f}", fontsize=7.5, va="center", color=INK)
    m, h = ci95(c); ax.errorbar(2, m, yerr=h, fmt="D", color=INK, capsize=3)
    ax.text(2.07, m, f"{m:+.2f}\n[{m - h:.2f}, {m + h:.2f}]", fontsize=7.5, va="center", color=INK)
    ax.set_xticks([0, 1, 2], ["Lee2019-MI", "Cho2017", "Pooled"]); ax.set_xlim(-.5, 2.8)
    ax.set_ylabel("c[s] (points)"); ax.set_title("H2b: (R2−R1) EEGNet − (R2−R1) CSOANet-v2", fontsize=8.5)
    ax.text(2.75, -SESOI + .3, "±2 SESOI", fontsize=7, color=MUTED, ha="right")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig4(iv, out, ivc=None):
    iv = iv.copy(); order = iv.groupby("pair")["mean"].apply(lambda s: s.abs().mean()).sort_values().index
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 3.0), sharey=True, sharex=True)
    XL = (-8, 9)
    for j, r in enumerate(RECIPES):
        ax = axs[j]; d = iv[iv.recipe == r].set_index("pair").loc[order]
        ax.axvspan(-SESOI, SESOI, color=GRID, zorder=0); ax.axvline(0, color=MUTED, lw=.8)
        for k, (p, row) in enumerate(d.iterrows()):
            if ivc is not None:        # clustered interval (post hoc), drawn underneath in a light tone
                rc = ivc[(ivc.pair == p) & (ivc.recipe == r)].iloc[0]
                lo_c, hi_c = max(rc["lo"], XL[0]), min(rc["hi"], XL[1])
                ax.plot([lo_c, hi_c], [k - .22, k - .22], color=CAT[rc.category], alpha=.45, lw=3, solid_capstyle="butt")
            if row["hi"] > XL[1]:      # off-scale: arrow, value and category
                ax.annotate("", xy=(XL[1] - .2, k), xytext=(XL[1] - 2.2, k),
                            arrowprops=dict(arrowstyle="->", color=CAT[row.category], lw=1.6))
                ax.text(XL[1] - 2.4, k + .28, f"{row['mean']:+.1f} {row.category}", fontsize=6.5, ha="right", color=INK)
                continue
            ax.errorbar(row["mean"], k, xerr=row["half"], fmt="o", color=CAT[row.category], ms=4, capsize=2, lw=1.6)
            ax.text(min(row["hi"] + .4, XL[1] - .6), k + (.3 if row["hi"] + .4 > XL[1] - .6 else 0), row.category,
                    fontsize=7, va="center", color=INK)
        ax.set_xlim(*XL)
        ax.set_title(r, color=INK); ax.set_xlabel("difference (points)", fontsize=8)
        ax.grid(axis="x", color=GRID, lw=.5)
    axs[0].set_yticks(range(len(order)), [p.replace(" - ", " − ") for p in order], fontsize=7.5)
    hs = [plt.Line2D([], [], color=CAT[k], marker="o", lw=1.6, label=f"{k} = {CATNAME[k]}") for k in "ABEI"]
    if ivc is not None:
        hs.append(plt.Line2D([], [], color="#9a9a96", alpha=.45, lw=3, label="clustered (post hoc)"))
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/confirmatory"); ap.add_argument("--ana", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default="V0.1")
    ap.add_argument("--clustered", default=None, help="post hoc clustered intervals CSV (revision_v11)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.concat([load_long(f"{a.res}/F1a_long.csv"), load_long(f"{a.res}/F1b_long.csv")])
    fig1(df, f"{a.out}/fig1_cells_{a.tag}.png")
    fig2(df, f"{a.out}/fig2_interaction_{a.tag}.png")
    fig3(df, f"{a.out}/fig3_h2b_{a.tag}.png")
    ivc = pd.read_csv(a.clustered) if a.clustered else None
    fig4(pd.read_csv(f"{a.ana}/F1_h3_intervals_f32.csv"), f"{a.out}/fig4_h3_intervals_{a.tag}.png", ivc)


if __name__ == "__main__":
    main()
