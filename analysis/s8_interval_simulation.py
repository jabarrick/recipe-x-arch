#!/usr/bin/env python3
"""
S8 — expected interval width and H3 outcome probabilities.

Implements Section 7 of docs/decision_rules.md (frozen before the binary pilot).

Input
-----
A CSV with one row per subject × architecture × recipe from the binary pilot
(seed-averaged, or single seed), columns:

    subject, architecture, recipe, bacc, n_trials

  bacc      balanced accuracy in PERCENT (0–100)
  n_trials  number of scored test trials for that subject in that cell

Steps
-----
1. Per pair (a,b) × recipe r: subject-level differences d[s]; decompose
   Var_obs = Var_between + Var_binom  (binomial component = sum of the two
   cells' p(1-p)/n, averaged over subjects; assumes uncorrelated trial-level
   errors — conservative); Var_between = max(Var_obs - Var_binom, 0);
   SD_conf = sqrt(Var_between + Var_binom at n = N_CONF).
2. Full 16-cell subject-level covariance from the pilot, binomial diagonal
   removed, projected to PSD, binomial diagonal at N_CONF added back. This
   gives correlated differences across pairs/recipes, which the joint
   B1 / B2 / A rules need.
3. Simulate n subjects over the scenario grid; apply the H3 rule and the
   B1 / B2 thresholds; report P(A), P(B1), P(B2), median half-widths.

Scenario conventions (fixed here, stated in S8)
-----------------------------------------------
D  = largest true pairwise architecture difference (points). Architecture
     main effects are D * (0, 1/3, 2/3, 1), so pairwise differences are
     D/3, 2D/3 and D. Grid: D in {1, 2, 3}.
G  = largest |true interaction residual| (points). Pattern: architectures 0
     and 1 swap by ±G between recipes R1 and R2, zero elsewhere (already
     double-centred). Grid: G in {0, 1, 2, 3}. G = 0 gives the family-wise
     false-positive rate of the H3 rule.

Category precedence
-------------------
A CI that excludes 0 AND lies within ±SESOI is both "better" and
"equivalent". Default: "better" wins (--equiv-first flips this). Record the
choice in the amendment.

Usage
-----
  python s8_interval_simulation.py pilot.csv --out s8_out/
  python s8_interval_simulation.py --demo --out s8_demo/      # synthetic check
"""
import argparse, itertools, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

ARCHS = ["EEGNet", "ShallowConvNet", "MI-BMInet", "CSOANet-v2"]
RECIPES = ["R1", "R2", "R3", "R4"]
PAIRS = list(itertools.combinations(range(4), 2))          # 6
RECIPE_PAIRS = list(itertools.combinations(range(4), 2))   # 6 -> 36 opportunities

# ----------------------------------------------------------------------------
# 1. load / pivot
# ----------------------------------------------------------------------------

def load_pilot(path):
    df = pd.read_csv(path)
    need = {"subject", "architecture", "recipe", "bacc", "n_trials"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"missing columns: {missing}")
    df["architecture"] = pd.Categorical(df["architecture"], ARCHS)
    df["recipe"] = pd.Categorical(df["recipe"], RECIPES)
    if df["architecture"].isna().any() or df["recipe"].isna().any():
        sys.exit(f"architecture must be in {ARCHS}, recipe in {RECIPES}")
    Y = df.pivot_table(index="subject", columns=["architecture", "recipe"],
                       values="bacc", observed=False)
    N = df.pivot_table(index="subject", columns=["architecture", "recipe"],
                       values="n_trials", observed=False)
    complete = Y.notna().all(axis=1)
    if (~complete).any():
        print(f"[S8] dropping {(~complete).sum()} subjects with incomplete 4x4 tables")
    Y, N = Y[complete], N[complete]
    y = Y.to_numpy().reshape(len(Y), 4, 4)    # [s, a, r]  percent
    n = N.to_numpy().reshape(len(Y), 4, 4)
    return y, n, list(Y.index)


# ----------------------------------------------------------------------------
# 2. noise decomposition
# ----------------------------------------------------------------------------

def binom_var_cell(p_pct, n):
    """binomial variance of an accuracy estimate, in percent^2."""
    p = np.clip(p_pct / 100.0, 1e-3, 1 - 1e-3)
    return 100.0**2 * p * (1 - p) / n


def decompose_pairs(y, n, n_conf):
    rows = []
    for (a, b) in PAIRS:
        for r in range(4):
            d = y[:, a, r] - y[:, b, r]
            var_obs = d.var(ddof=1)
            vb_pilot = (binom_var_cell(y[:, a, r], n[:, a, r]) +
                        binom_var_cell(y[:, b, r], n[:, b, r])).mean()
            vb_conf = (binom_var_cell(y[:, a, r], n_conf) +
                       binom_var_cell(y[:, b, r], n_conf)).mean()
            var_between = max(var_obs - vb_pilot, 0.0)
            rows.append(dict(
                pair=f"{ARCHS[a]} - {ARCHS[b]}", recipe=RECIPES[r],
                mean_diff_pilot=d.mean(), sd_pilot=np.sqrt(var_obs),
                sd_binom_pilot=np.sqrt(vb_pilot),
                sd_between=np.sqrt(var_between),
                truncated=var_obs < vb_pilot,
                sd_binom_conf=np.sqrt(vb_conf),
                sd_conf=np.sqrt(var_between + vb_conf)))
    return pd.DataFrame(rows)


def subject_covariance(y, n, n_conf, model="full"):
    """16x16 covariance of subject-level deviations at the confirmatory trial
    count: pilot covariance of double-centred tables, minus pilot binomial
    diagonal, projected to PSD, plus binomial diagonal at n_conf."""
    S = y.shape[0]
    # remove each subject's own overall level (u_s cancels in every difference,
    # so it is irrelevant to the intervals; keeping it would just inflate
    # the common component)
    dev = y - y.mean(axis=(1, 2), keepdims=True)
    X = dev.reshape(S, 16)
    C = np.cov(X, rowvar=False)
    vb_pilot = binom_var_cell(y, n).mean(axis=0).reshape(16)
    vb_conf = binom_var_cell(y, n_conf).mean(axis=0).reshape(16)
    C_between = C - np.diag(vb_pilot)
    if model == "full":
        w, V = np.linalg.eigh(C_between)
        n_clipped = int((w < 0).sum())
        w = np.clip(w, 0, None)
        C_between = (V * w) @ V.T
    elif model == "arch":
        # simplified: between-subject variance from a subject x architecture
        # component only (shared across the 4 recipes of an architecture)
        # plus an independent cell residual. Use when the full 16x16 estimate
        # is poorly conditioned (many clipped eigenvalues).
        blocks = C_between.reshape(4, 4, 4, 4)           # [a, r, a', r']
        same_arch = np.array([[blocks[a, :, a, :][~np.eye(4, dtype=bool)].mean()
                               for a in range(4)]]).ravel()   # off-diag within arch
        resid = np.array([np.diag(blocks[a, :, a, :]).mean() for a in range(4)]) - same_arch
        same_arch, resid = np.clip(same_arch, 0, None), np.clip(resid, 0, None)
        C_between = np.zeros((16, 16))
        for a in range(4):
            idx = slice(4 * a, 4 * a + 4)
            C_between[idx, idx] = same_arch[a] + np.diag(np.full(4, resid[a]))
        n_clipped = -1
    else:
        raise ValueError(model)
    C_conf = C_between + np.diag(vb_conf)
    return C_conf, C_between, n_clipped


# ----------------------------------------------------------------------------
# 3. simulation
# ----------------------------------------------------------------------------

def true_means(D, G, base=70.0, recipe_effects=(0.0, 1.0, 0.5, -0.5)):
    alpha = D * np.array([0, 1 / 3, 2 / 3, 1.0])
    rho = np.array(recipe_effects)
    inter = np.zeros((4, 4))
    if G > 0:
        inter[0, 0], inter[0, 1] = +G, -G     # arch 0: R1 up, R2 down
        inter[1, 0], inter[1, 1] = -G, +G     # arch 1: mirror
    return base + alpha[:, None] + rho[None, :] + inter


def classify(mean, half, sesoi, equiv_first):
    lo, hi = mean - half, mean + half
    better = "A" if lo > 0 else ("B" if hi < 0 else None)
    equiv = (lo > -sesoi) and (hi < sesoi)
    if equiv_first:
        if equiv: return "E"
        if better: return better
    else:
        if better: return better
        if equiv: return "E"
    return "I"


def evaluate_table(y_sim, tcrit, sesoi, equiv_first):
    """y_sim: [n, 4, 4]. Returns categories [6 pairs, 4 recipes], halfwidths,
    H3 flag, B1/B2 flags, change counts."""
    n = y_sim.shape[0]
    cats = np.empty((6, 4), dtype=object)
    halves = np.zeros((6, 4))
    for i, (a, b) in enumerate(PAIRS):
        d = y_sim[:, a, :] - y_sim[:, b, :]              # [n, 4]
        m, sd = d.mean(0), d.std(0, ddof=1)
        half = tcrit * sd / np.sqrt(n)
        halves[i] = half
        for r in range(4):
            cats[i, r] = classify(m[r], half[r], sesoi, equiv_first)
    definitive = np.isin(cats, ["A", "B", "E"])
    n_def = int(definitive.sum())
    def_def, inc_def, sign_rev = 0, 0, 0
    for i in range(6):
        for (r1, r2) in RECIPE_PAIRS:
            c1, c2 = cats[i, r1], cats[i, r2]
            if c1 == c2: continue
            if c1 != "I" and c2 != "I":
                def_def += 1
                if {c1, c2} == {"A", "B"}: sign_rev += 1
            else:
                inc_def += 1
    A = def_def > 0                       # registered H3 rule
    A_sign = sign_rev > 0                 # sign reversals only (reported alongside)
    A_equiv_only = A and not A_sign       # support comes only from equivalent <-> better
    B1 = (not A) and n_def >= 16 and def_def == 0
    B2 = (not A) and n_def < 8
    return dict(A=A, A_sign=A_sign, A_equiv_only=A_equiv_only, B1=B1, B2=B2,
                n_definitive=n_def, changes_def_def=def_def,
                changes_sign_rev=sign_rev, changes_inc_def=inc_def), halves


def simulate(C_conf, n, D, G, n_sims, rng, sesoi, equiv_first, alpha_fw=0.05):
    tcrit = stats.t.ppf(1 - alpha_fw / 24 / 2, df=n - 1)   # 99.79 %
    mu = true_means(D, G).reshape(16)
    L = np.linalg.cholesky(C_conf + 1e-9 * np.eye(16))
    res, halves = [], []
    for _ in range(n_sims):
        z = rng.standard_normal((n, 16))
        y_sim = (mu + z @ L.T).reshape(n, 4, 4)
        r, h = evaluate_table(y_sim, tcrit, sesoi, equiv_first)
        res.append(r); halves.append(h)
    res = pd.DataFrame(res)
    halves = np.array(halves)                     # [sims, 6, 4]
    return dict(D=D, G=G, n=n,
                P_A=res.A.mean(), P_A_sign=res.A_sign.mean(),
                P_A_equiv_only=res.A_equiv_only.mean(),
                P_B1=res.B1.mean(), P_B2=res.B2.mean(),
                mean_n_definitive=res.n_definitive.mean(),
                mean_changes_def_def=res.changes_def_def.mean(),
                mean_changes_sign_rev=res.changes_sign_rev.mean(),
                mean_changes_inc_def=res.changes_inc_def.mean(),
                median_halfwidth=float(np.median(halves)),
                ), np.median(halves, axis=0)


# ----------------------------------------------------------------------------
# demo data (synthetic pilot) — for testing the pipeline only
# ----------------------------------------------------------------------------

def make_demo(path, n_subj=105, n_trials=45, seed=0):
    rng = np.random.default_rng(seed)
    mu = true_means(2.0, 1.0, base=65.0)
    u = rng.normal(0, 8, n_subj)                        # subject level
    sa = rng.normal(0, 3, (n_subj, 4))                  # subject x arch
    rows = []
    for s in range(n_subj):
        for a in range(4):
            for r in range(4):
                p = np.clip((mu[a, r] + u[s] + sa[s, a]) / 100, 0.05, 0.95)
                k = rng.binomial(n_trials, p)
                rows.append(dict(subject=f"S{s:03d}", architecture=ARCHS[a],
                                 recipe=RECIPES[r], bacc=100 * k / n_trials,
                                 n_trials=n_trials))
    pd.DataFrame(rows).to_csv(path, index=False)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_csv", nargs="?")
    ap.add_argument("--out", default="s8_out")
    ap.add_argument("--n-conf", type=int, default=200,
                    help="scored trials per subject per cell in confirmatory data")
    ap.add_argument("--n-subjects", type=int, nargs="+", default=[106],
                    help="sample sizes to simulate, e.g. 106 193")
    ap.add_argument("--D", type=float, nargs="+", default=[1, 2, 3])
    ap.add_argument("--G", type=float, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--sesoi", type=float, default=2.0)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--equiv-first", action="store_true")
    ap.add_argument("--cov", choices=["full", "arch"], default="full",
                    help="between-subject covariance model (see subject_covariance)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.demo:
        args.pilot_csv = os.path.join(args.out, "demo_pilot.csv")
        make_demo(args.pilot_csv)
    if not args.pilot_csv:
        ap.error("pilot_csv required (or --demo)")

    y, n, subjects = load_pilot(args.pilot_csv)
    print(f"[S8] {len(subjects)} subjects, pilot trials/cell median = {np.median(n):.0f}")

    # step 1
    dec = decompose_pairs(y, n, args.n_conf)
    dec.to_csv(os.path.join(args.out, "S8_pair_sd_decomposition.csv"), index=False)
    print(f"[S8] truncations (Var_obs < Var_binom): {int(dec.truncated.sum())} / {len(dec)}")

    # step 2
    C_conf, C_between, n_clipped = subject_covariance(y, n, args.n_conf, args.cov)
    if args.cov == "full":
        print(f"[S8] covariance eigenvalues clipped to 0: {n_clipped} / 16"
              + ("  -> consider --cov arch" if n_clipped >= 8 else ""))
    np.savetxt(os.path.join(args.out, "S8_cov_conf.csv"), C_conf, delimiter=",")

    # step 3
    rng = np.random.default_rng(args.seed)
    rows, hw_tables = [], {}
    for nsub in args.n_subjects:
        for D in args.D:
            for G in args.G:
                r, hw = simulate(C_conf, nsub, D, G, args.n_sims, rng,
                                 args.sesoi, args.equiv_first)
                rows.append(r); hw_tables[(nsub, D, G)] = hw
                print(f"  n={nsub} D={D} G={G}: P(A)={r['P_A']:.3f} "
                      f"P(A_sign)={r['P_A_sign']:.3f} "
                      f"P(B1)={r['P_B1']:.3f} P(B2)={r['P_B2']:.3f} "
                      f"half={r['median_halfwidth']:.2f}")
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(args.out, "S8_scenario_grid.csv"), index=False)

    # median half-width per pair/recipe at first n, D=max, G=0 (width barely depends on D,G)
    hw = hw_tables[(args.n_subjects[0], args.D[-1], 0.0 if 0.0 in args.G else args.G[0])]
    pd.DataFrame(hw, index=[f"{ARCHS[a]} - {ARCHS[b]}" for a, b in PAIRS],
                 columns=RECIPES).to_csv(os.path.join(args.out, "S8_halfwidth_by_pair.csv"))

    # decision quantity (decision_rules.md §4)
    sel = grid[(grid.n == args.n_subjects[0]) & (grid.G == 0) & (grid.D.isin([1, 2, 3]))]
    pb2 = sel.P_B2.mean()
    decision = {"P_B2_zero_interaction_D1to3_n%d" % args.n_subjects[0]: pb2,
                "add_third_dataset": bool(pb2 > 0.5),
                "P_A_zero_interaction_(false_positive_rate)": sel.P_A.mean(),
                "P_A_sign_zero_interaction": sel.P_A_sign.mean(),
                "P_A_equiv_only_zero_interaction": sel.P_A_equiv_only.mean(),
                "cov_model": args.cov,
                "sesoi": args.sesoi, "n_conf": args.n_conf,
                "equiv_first": args.equiv_first, "n_sims": args.n_sims}
    with open(os.path.join(args.out, "S8_decision.json"), "w") as f:
        json.dump(decision, f, indent=2)

    # markdown summary
    md = [f"# S8 — interval-width simulation\n",
          f"Pilot: {len(subjects)} subjects, median {np.median(n):.0f} trials/cell; "
          f"confirmatory n_trials = {args.n_conf}; SESOI = ±{args.sesoi}; "
          f"{args.n_sims} simulations per scenario; category precedence: "
          f"{'equivalent first' if args.equiv_first else 'better first'}.\n",
          "## Per-pair SD decomposition (points)\n",
          dec.round(2).to_markdown(index=False), "\n",
          "## Scenario grid\n", grid.round(3).to_markdown(index=False), "\n",
          "## Decision quantity (decision_rules.md §4)\n",
          f"P(B2 | G = 0, D ∈ {{1,2,3}}, n = {args.n_subjects[0]}) = **{pb2:.3f}** → "
          f"{'ADD third dataset' if pb2 > 0.5 else 'do NOT add'}\n",
          f"P(A | G = 0) = {sel.P_A.mean():.3f} (family-wise false-positive rate of the registered H3 rule)\n",
          f"P(A_sign | G = 0) = {sel.P_A_sign.mean():.3f} (sign reversals only)\n",
          f"P(A from equivalent ↔ better only | G = 0) = {sel.P_A_equiv_only.mean():.3f}\n",
          "\nReading rule (Amendment 1 §3.2): if H3 support comes only from "
          "equivalent ↔ better changes with no sign reversal, it is reported as "
          "'difference below SESOI with stable direction', not as a change in ranking.\n"]
    with open(os.path.join(args.out, "S8_summary.md"), "w") as f:
        f.write("\n".join(md))

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
        g0 = grid[grid.n == args.n_subjects[0]]
        for ax, key, ttl in zip(axes, ["P_A", "P_A_sign", "P_B1", "P_B2"],
                                ["P(A): H3 supported (registered rule)",
                                 "P(A_sign): sign reversals only",
                                 "P(B1): stable", "P(B2): inconclusive"]):
            for D in args.D:
                sub = g0[g0.D == D]
                ax.plot(sub.G, sub[key], marker="o", label=f"D = {D:g}")
            ax.set_title(ttl); ax.set_xlabel("true interaction G (points)")
            ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3)
        axes[0].set_ylabel("probability"); axes[0].legend(title="max pairwise diff")
        fig.suptitle(f"n = {args.n_subjects[0]}, median half-width ≈ "
                     f"{g0.median_halfwidth.median():.2f} points")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "S8_outcome_probabilities.png"), dpi=150)
    except Exception as e:  # matplotlib optional
        print(f"[S8] figure skipped: {e}")

    print(f"[S8] done → {args.out}/")


if __name__ == "__main__":
    main()
