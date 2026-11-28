#!/usr/bin/env python3
"""Figures 2 and 3 of V1.15 (corrected design). Reads result files and rev19/revision_v19.json.
Usage: python -m analysis.figures_v115 --res data --nc nc --rev6 rev6 --json rev19/revision_v19.json --out figs
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .figures_section4 import COL, MARK, INK, MUTED, GRID, ci95
from .revision_v19 import load, design, DESIGNS, A, HOME
from .common import SESOI

LAB = {"registered": "As registered (R1 with crop; ShallowConvNet without max-norm)",
       "corrected": "Corrected (R1 without crop; ShallowConvNet with max-norm)"}


def cube(g, name):
    spec = DESIGNS[name]; d = design(g, **spec); R = ["R1" if r == "R1nc" else r for r in spec["recipes"]]
    y = d.pivot_table(index="subject", columns=["architecture", "recipe"], values="y")
    ds = d.drop_duplicates("subject").set_index("subject").dataset.reindex(y.index).to_numpy()
    return np.array([[y[(a, r)].to_numpy() for r in R] for a in A]).transpose(2, 0, 1), R, ds


def fig2(g, out):
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    lim = 7.0
    for row, name in enumerate(("registered", "corrected")):
        y, R, _ = cube(g, name); ax, hx = axs[row]; x = np.arange(4)
        for i, a in enumerate(A):
            m = np.nanmean(y[:, i], 0); h = np.array([ci95(y[:, i, j])[1] for j in range(4)])
            ax.errorbar(x + (i - 1.5) * .16, m, yerr=h, color=COL[a], marker=MARK[a], ms=5, lw=0, elinewidth=1.4,
                        capsize=2, mec="white", mew=.8, label=a)
        ax.set_xticks(x, R); ax.set_xlim(-.5, 3.5); ax.set_ylim(55, 82); ax.set_ylabel("Cell mean (%) ± 95 % CI")
        ax.grid(axis="y", color=GRID, lw=.5); ax.set_title(LAB[name], fontsize=7.5, color=INK, loc="left")
        if row == 0: ax.legend(frameon=False, fontsize=7, loc="lower left")
        t = np.nanmean(y, 0); r = t - t.mean(1, keepdims=True) - t.mean(0, keepdims=True) + t.mean()
        im = hx.imshow(r, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(4):
            for j in range(4):
                hx.text(j, i, f"{r[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(r[i, j]) > .6 * lim else INK)
                if HOME.get(A[i]) == R[j]:
                    hx.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.8))
        hx.set_xticks(range(4), R); hx.set_yticks(range(4), A)
        for s in hx.spines.values(): s.set_visible(False)
        hx.set_title("Interaction residuals (points)", fontsize=7.5, color=INK)
        fig.colorbar(im, ax=hx, fraction=.046, pad=.04)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig3(g, J, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.3)); rng = np.random.default_rng(1)
    ax.axhspan(-SESOI, SESOI, color=GRID, zorder=0); ax.axhline(0, color=MUTED, lw=.8)
    for k, name in enumerate(("registered", "corrected")):
        y, R, ds = cube(g, name); e, v = A.index("EEGNet"), A.index("CSOANet-v2")
        c = (y[:, e, 1] - y[:, e, 0]) - (y[:, v, 1] - y[:, v, 0])
        for d, mk in (("Lee2019_MI", "o"), ("Cho2017", "^")):
            cc = c[ds == d]
            ax.scatter(k + rng.uniform(-.2, .2, len(cc)), cc, s=11, marker=mk, color=COL["EEGNet"], alpha=.55, ec="white", lw=.4)
        w = J[name]["H2b"]; m = w["mean"]
        ax.errorbar(k + .32, m, yerr=[[m - w["lo"]], [w["hi"] - m]], fmt="D", color=INK, capsize=3)
        ax.text(k + .38, m, f"{m:+.2f}\n({w['lo']:.2f}, {w['hi']:.2f})", fontsize=7.2, va="center", color=INK)
    ax.set_xticks([0, 1], ["R1 with crop\n(as registered)", "R1 without crop\n(corrected)"]); ax.set_xlim(-.5, 1.95)
    ax.set_ylabel("c[s] (points)")
    ax.set_title("H2b: (R2 − R1) EEGNet − (R2 − R1) CSOANet-v2", fontsize=8.5)
    ax.text(-.46, SESOI - .9, "±2 SESOI", fontsize=7, color=MUTED, ha="left")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    for k in ("res", "nc", "rev6", "json", "out"): ap.add_argument("--" + k, required=True)
    o = ap.parse_args(); os.makedirs(o.out, exist_ok=True); g = load(o); J = json.load(open(o.json))
    fig2(g, f"{o.out}/fig2_registered_vs_corrected_V0.5.png"); fig3(g, J, f"{o.out}/fig3_h2b_V0.5.png")


if __name__ == "__main__":
    main()
