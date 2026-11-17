#!/usr/bin/env python3
"""Confirmatory training driver (registration OSF 8st47; Amendment 1).

Evaluation units
  F1a  Lee2019-MI  cross-subject, 5 subject folds        (primary family F1, with F1b)
  F1b  Cho2017     cross-subject, 5 subject folds
  F2   Lee2019-MI  within-subject, session 1 -> session 2 (secondary family)

Stages (registered run order; --stage selects one)
  recipes   R1-R4 on the chosen unit(s)            -> <out>/<unit>_long.csv     (input to analysis.run_confirmatory)
  T         24-configuration search per architecture on the unit's tuning split, then the
            best configuration on all folds / subjects  -> <out>/<unit>_T_search.csv, <unit>_T_long.csv
  Tprime    the configurations in docs/Tprime/tprime_selected.json on all folds / subjects
                                                    -> <out>/<unit>_Tprime_long.csv

Fixed choices (registration, Amendment 1, decision_rules.md)
  preprocessing  MOABB 1.7.2 LeftRightImagery: band-pass 1-40 Hz, resample 250 Hz, dataset interval
                 (Lee 0-4 s, Cho 0-3 s after cue), all EEG channels; common-average reference
  normalisation  one gain per channel: cross-subject from the fold's training subjects; within-subject
                 from the subject's session-1 training trials
  splits         cross-subject: subjects shuffled with seed 0 into 5 folds, 10 % of training subjects
                 validate; within-subject: 20 % of session 1 validates, session 2 tests
  seeds          0, 1, 2 (training seed = seed*1000 + fold or subject; retry once with +100 if the loss
                 becomes non-finite)
  protocol       cross-subject 100 epochs, batch 64, patience 15; within-subject 500 epochs, batch 32,
                 patience 80; R3 keeps batch 16; T / T' configurations keep their own batch size
  dropout        EEGNet 0.25 cross / 0.5 within; MI-BMInet 0.2 cross / 0.5 within
  failure flag   non-finite loss after retry, or un-augmented training balanced accuracy <= 55 % at the
                 selected checkpoint (training-side only)
  int8           ONNX Runtime static QDQ, per-channel symmetric int8 weights, MinMax; calibration on 256
                 un-augmented training trials drawn without replacement, seed = fold index (cross) or
                 subject index (within), the same set for every model in that fold / subject; indices saved
  exclusions     a subject MOABB cannot load, or with < 80 % of 200 nominal trials, is excluded before
                 training and listed in <out>/<unit>_exclusions.csv

Resumable: every finished (architecture, recipe, fold-or-subject, seed) is skipped on rerun.

    python -m train.confirmatory --csoanet-root ..\\csoanet2026_v2_deploy --unit F1a --stage recipes
    python -m train.confirmatory --csoanet-root ... --unit F1a --stage recipes --pilot-dryrun --epochs 2 --max-subjects 10
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from train.pilot_binary import ARCHS, load_deploy, constrain
from train import configurable as C

RECIPES = ["R1", "R2", "R3", "R4"]
UNITS = {"F1a": ("Lee2019_MI", "cross-subject"), "F1b": ("Cho2017", "cross-subject"),
         "F2": ("Lee2019_MI", "within-subject")}
FS = 250.0
NOMINAL_TRIALS = 200
FAIL_BACC = 55.0
N_CALIB = 256


# ------------------------------------------------------------------ data
MOABB_VERSION = "1.7.2"                                          # registered


def _check_moabb():
    import moabb
    if moabb.__version__ != MOABB_VERSION:
        raise RuntimeError(f"MOABB {MOABB_VERSION} is registered; found {moabb.__version__}. "
                           f"pip install moabb=={MOABB_VERSION}")
    return moabb.__version__


def load_unit(dataset_name, cache_dir, max_subjects=None):
    """Returns X (n, C, T) float32 CAR, y (0 left, 1 right), subject ids, session index (0/1), exclusions."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) / f"cache_{dataset_name}_1-40Hz_250Hz.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if "moabb" not in d.files or str(d["moabb"]) != MOABB_VERSION:
            raise RuntimeError(f"{cache} was not built with MOABB {MOABB_VERSION}; delete it and rebuild")
        X, y, g, sess, excl = d["X"], d["y"], d["g"], d["sess"], list(d["excl"])
    else:
        version = _check_moabb()
        import mne
        from moabb.paradigms import LeftRightImagery
        import moabb.datasets as MD
        mne.set_log_level("ERROR")
        ds = getattr(MD, dataset_name)()
        paradigm = LeftRightImagery(fmin=1.0, fmax=40.0, resample=FS)
        Xs, ys, gs, ss, excl = [], [], [], [], []
        for s in ds.subject_list[:max_subjects]:
            try:
                Xi, labels, meta = paradigm.get_data(dataset=ds, subjects=[s])
            except Exception as e:                              # cannot be loaded
                excl.append(dict(subject=int(s), reason=f"load error: {type(e).__name__}: {e}"[:200]))
                print(f"  {dataset_name} S{s:02d}: EXCLUDED, {excl[-1]['reason']}")
                continue
            if len(labels) < 0.8 * NOMINAL_TRIALS:
                excl.append(dict(subject=int(s), reason=f"{len(labels)} trials < 80 % of {NOMINAL_TRIALS} "
                                                        f"(sessions {sorted(meta['session'].unique())})"))
                print(f"  {dataset_name} S{s:02d}: EXCLUDED, {excl[-1]['reason']}")
                continue
            Xi = Xi.astype(np.float32)
            Xi -= Xi.mean(axis=1, keepdims=True)                # common-average reference
            names = sorted(meta["session"].unique())
            Xs.append(Xi); ys.append((np.asarray(labels) == "right_hand").astype(np.int64))
            gs.append(np.full(len(labels), int(s))); ss.append(meta["session"].map({n: i for i, n in enumerate(names)}).to_numpy())
            print(f"  {dataset_name} S{s:02d}: {len(labels)} trials, {Xi.shape[1]} ch x {Xi.shape[2]}, "
                  f"sessions {dict(meta['session'].value_counts().sort_index())}")
        if not Xs:
            raise RuntimeError(f"all {len(excl)} subjects of {dataset_name} were excluded; first reasons: {excl[:3]}")
        T = min(x.shape[2] for x in Xs)                          # guard against off-by-one sample counts
        X = np.concatenate([x[:, :, :T] for x in Xs]); y = np.concatenate(ys)
        g = np.concatenate(gs); sess = np.concatenate(ss)
        np.savez(cache, X=X, y=y, g=g, sess=sess, excl=np.array(excl, dtype=object), moabb=np.array(version))
    return X, y, g, sess, excl


def load_pilot_dryrun(cache, max_subjects, within):
    """PhysioNet left/right pilot cache, used only to dry-run this driver before any confirmatory download.
    For the within-subject path a fake two-session split (first / second half of each subject) is made."""
    d = np.load(cache)
    X, y, g = d["X"], d["y"], d["g"]
    keep = np.isin(g, np.unique(g)[:max_subjects]) if max_subjects else np.ones(len(y), bool)
    X, y, g = X[keep], y[keep], g[keep]
    sess = np.zeros(len(y), np.int64)
    if within:
        for s in np.unique(g):
            idx = np.flatnonzero(g == s); sess[idx[len(idx) // 2:]] = 1
    return X, y, g, sess, []


# ------------------------------------------------------------------ folds / normalisation
def folds(g, sess, protocol, val_frac, split_seed=0):
    """Yield (key, train_idx, val_idx, test_idx, test_subjects)."""
    if protocol == "cross-subject":
        subs = np.unique(g); rng = np.random.default_rng(split_seed); rng.shuffle(subs)
        for k, test in enumerate(np.array_split(subs, 5)):
            rest = np.setdiff1d(subs, test); rest = rest[rng.permutation(len(rest))]
            n_val = max(1, int(round(len(rest) * val_frac)))
            tr_s, va_s = rest[n_val:], rest[:n_val]
            yield k, *(np.flatnonzero(np.isin(g, x)) for x in (tr_s, va_s, test)), test
    else:
        rng = np.random.default_rng(split_seed)
        for s in np.unique(g):
            first = np.flatnonzero((g == s) & (sess == 0)); first = first[rng.permutation(len(first))]
            n_val = max(1, int(round(len(first) * val_frac)))
            yield int(s), first[n_val:], first[:n_val], np.flatnonzero((g == s) & (sess == 1)), np.array([s])


def gain_normalise(X, train_idx):
    gain = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-8
    return (X / gain).astype(np.float32)


# ------------------------------------------------------------------ models
def build(arch, ch, t, within, deps):
    tp, bl, mb, v2 = deps
    if arch == "EEGNet":
        return bl.make_eegnet(ch, t, 2, fs=FS, drop=0.5 if within else 0.25)
    if arch == "ShallowConvNet":
        from train.shallow import ShallowConvNet
        return ShallowConvNet(ch, t, 2)
    if arch == "MI-BMInet":
        return mb.MIBMInet(ch, t, 2, fs=FS, nk=32, nf=64, p_dropout=0.5 if within else 0.2)
    if arch == "CSOANet-v2":
        return v2.CSOANetV2(ch, t, 2, kt=tp._long(FS))
    raise ValueError(arch)


def recipe_cfg(name, within):
    cfg = dict(C.REGISTERED[name])
    if within and name != "R3":
        cfg["batch"] = 32
    return cfg


# ------------------------------------------------------------------ int8
def op_coverage(int8_path):
    """Share of compute nodes wrapped in Q/DQ (same rule as deploy.quantize.float_ops), plus float op types."""
    import onnx
    g = onnx.load(str(int8_path)).graph
    init = {i.name for i in g.initializer}
    produced_by = {o: n.op_type for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n.op_type)
    skip = {"QuantizeLinear", "DequantizeLinear", "Constant", "Shape", "Gather", "Unsqueeze", "Concat",
            "Reshape", "Flatten", "Squeeze"}
    graph_out = {o.name for o in g.output}
    total, quant, floats = 0, 0, set()
    for n in g.node:
        if n.op_type in skip:
            continue
        total += 1
        ins = [produced_by[i] for i in n.input if i in produced_by and i not in init]
        ok_in = all(t == "DequantizeLinear" for t in ins)
        ok_out = all(o in graph_out or all(c == "QuantizeLinear" for c in consumers.get(o, [])) for o in n.output)
        if ok_in and ok_out:
            quant += 1
        else:
            floats.add(n.op_type)
    return quant, total, sorted(floats)


# ------------------------------------------------------------------ one run
def run_one(arch, cfg, Xn, Y, y, g, tr, va, te, te_s, within, epochs, patience, dev, seed, deps, calib_idx,
            onnx_dir, name, amp):
    ch, t = Xn.shape[2], Xn.shape[3]
    retried = False
    model = build(arch, ch, t, within, deps)
    model, info = C.train_cfg(model, lambda m: constrain(arch, m, deps), cfg, Xn[tr], Y[tr], Xn[va], Y[va],
                              2, epochs, dev, seed=seed, amp=amp, patience=patience)
    if info["nonfinite"]:
        model = build(arch, ch, t, within, deps)
        model, info = C.train_cfg(model, lambda m: constrain(arch, m, deps), cfg, Xn[tr], Y[tr], Xn[va], Y[va],
                                  2, epochs, dev, seed=seed + 100, amp=amp, patience=patience)
        retried = True
    failure = int(info["nonfinite"] or (not math.isnan(info["train_bacc"]) and info["train_bacc"] <= FAIL_BACC))
    tp = deps[0]
    export = tp.deploy_form(model.cpu().eval())
    from deploy.quantize import int8_report
    fp, q, (_, qpath) = int8_report(export, Xn[calib_idx].numpy(), Xn[te].numpy(), onnx_dir, name, "minmax")
    pf, pq = fp.argmax(1), q.argmax(1)
    rows = []
    for s in te_s:
        k = g[te] == s
        rows.append(dict(subject=int(s), bacc_f32=C.bacc(y[te][k], pf[k], 2), bacc_int8=C.bacc(y[te][k], pq[k], 2),
                         int8_agreement=100.0 * float((pf[k] == pq[k]).mean()), n_trials=int(k.sum()),
                         train_acc=info["train_bacc"], val_acc=info["val_bacc"], best_epoch=info["best_epoch"],
                         epochs=info["epochs_run"], training_failure=failure, retried=int(retried)))
    return rows, qpath


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csoanet-root", required=True)
    ap.add_argument("--unit", choices=list(UNITS), required=True)
    ap.add_argument("--stage", choices=["recipes", "T", "Tprime"], default="recipes")
    ap.add_argument("--out", default="results/confirmatory")
    ap.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    ap.add_argument("--recipes", nargs="+", default=RECIPES, choices=RECIPES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--t-scoring-seeds", nargs="+", type=int, default=list(C.SCORING_SEEDS),
                    help="registered: 0 1; reduce to 0 only under the pre-specified compute cut")
    ap.add_argument("--tprime", default="docs/Tprime/tprime_selected.json")
    ap.add_argument("--epochs", type=int, default=None, help="override, dry runs only")
    ap.add_argument("--max-subjects", type=int, default=None, help="dry runs only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--pilot-dryrun", default=None, const="results/pilot_binary/cache_lr_1-40Hz_0-4s.npz", nargs="?",
                    help="run on the PhysioNet pilot cache instead of the confirmatory data (dry run)")
    a = ap.parse_args()

    deps = load_deploy(a.csoanet_root)
    dataset_name, protocol = UNITS[a.unit]
    within = protocol == "within-subject"
    epochs = a.epochs or (500 if within else 100)
    patience = 80 if within else 15
    val_frac = 0.2 if within else 0.1
    dev = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    tag = a.unit + ("_DRYRUN" if a.pilot_dryrun else "")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    onnx_dir = out / "onnx"; onnx_dir.mkdir(exist_ok=True)

    if a.pilot_dryrun:
        X, y, g, sess, excl = load_pilot_dryrun(a.pilot_dryrun, a.max_subjects, within)
    else:
        X, y, g, sess, excl = load_unit(dataset_name, out, a.max_subjects)
    pd.DataFrame(excl, columns=["subject", "reason"]).to_csv(out / f"{tag}_exclusions.csv", index=False)
    print(f"{tag}: {len(y)} trials, {len(np.unique(g))} subjects, {X.shape[1]} ch x {X.shape[2]} | "
          f"{protocol}, {epochs} epochs | stage {a.stage} | {dev} | excluded {len(excl)}")
    Y = torch.from_numpy(y)
    fold_list = list(folds(g, sess, protocol, val_frac))

    def configs_for_stage():
        if a.stage == "recipes":
            return {(arch, r): recipe_cfg(r, within) for arch in a.archs for r in a.recipes}
        if a.stage == "Tprime":
            sel = json.loads(Path(a.tprime).read_text())
            return {(arch, "Tprime"): sel[arch]["config"] for arch in a.archs}
        return {(arch, "T"): t_search(arch) for arch in a.archs}

    def t_search(arch):
        """Registered T search on the unit's tuning split; returns the selected configuration."""
        log = out / f"{tag}_T_search.csv"
        rows = pd.read_csv(log).to_dict("records") if log.exists() else []
        done = {(r["architecture"], r["config_id"], r["seed"], r["unit_key"]) for r in rows}
        if within:   # session 1 of the 10 lowest-ID subjects
            tune = [f for f in fold_list if f[0] in sorted(np.unique(g))[:10]]
        else:        # fold 1's training and validation subjects
            tune = fold_list[:1]
        for cfg in C.sample_configs():
            for seed in a.t_scoring_seeds:
                for key, tr, va, te, te_s in tune:
                    if (arch, cfg["id"], seed, key) in done:
                        continue
                    Xn = torch.from_numpy(gain_normalise(X, tr)).unsqueeze(1)
                    model = build(arch, X.shape[1], X.shape[2], within, deps)
                    _, info = C.train_cfg(model, lambda m: constrain(arch, m, deps), cfg, Xn[tr], Y[tr], Xn[va], Y[va],
                                          2, epochs, dev, seed=seed * 1000, amp=not a.no_amp, patience=patience)
                    rows.append(dict(architecture=arch, config_id=cfg["id"], seed=seed, unit_key=key,
                                     val_bacc=info["val_bacc"], nonfinite=info["nonfinite"]))
                    pd.DataFrame(rows).to_csv(log, index=False)
        df = pd.DataFrame(rows); df = df[df.architecture == arch]
        sc = df.groupby("config_id").val_bacc.mean().fillna(-math.inf).reset_index()
        best = int(sc.sort_values(["val_bacc", "config_id"], ascending=[False, True]).iloc[0].config_id)
        print(f"  T search {arch}: selected configuration {best} (score {sc.val_bacc.max():.2f})")
        return next(c for c in C.sample_configs() if c["id"] == best)

    cfgs = configs_for_stage()
    long_csv = out / (f"{tag}_long.csv" if a.stage == "recipes" else f"{tag}_{a.stage}_long.csv")
    rows = pd.read_csv(long_csv).to_dict("records") if long_csv.exists() else []
    done = {(r["architecture"], r["recipe"], r["fold"], r["seed"]) for r in rows}
    cov_csv = out / f"{tag}_int8_coverage.csv"
    cov = pd.read_csv(cov_csv).to_dict("records") if cov_csv.exists() else []
    calib_log = out / f"{tag}_calibration_indices.json"
    calib_all = json.loads(calib_log.read_text()) if calib_log.exists() else {}

    for key, tr, va, te, te_s in fold_list:
        if all((arch, rec, key, s) in done for (arch, rec) in cfgs for s in a.seeds):
            continue
        Xn = torch.from_numpy(gain_normalise(X, tr)).unsqueeze(1)
        rng = np.random.default_rng(int(key))                     # seed = fold index / subject index
        calib_idx = np.sort(rng.choice(tr, min(N_CALIB, len(tr)), replace=False))
        calib_all[str(key)] = calib_idx.tolist(); calib_log.write_text(json.dumps(calib_all))
        for (arch, rec), cfg in cfgs.items():
            for seed in a.seeds:
                if (arch, rec, key, seed) in done:
                    continue
                t0 = time.time()
                name = f"{tag}_{arch}_{rec}_{key}_s{seed}"
                rr, qpath = run_one(arch, cfg, Xn, Y, y, g, tr, va, te, te_s, within, epochs, patience, dev,
                                    seed * 1000 + int(key), deps, calib_idx, onnx_dir, name, not a.no_amp)
                ds_label = "PhysioNet-MI (dry run)" if a.pilot_dryrun else dataset_name
                for r in rr:
                    rows.append(dict(subject=f"{a.unit}-S{r.pop('subject'):03d}", dataset=ds_label, fold=key,
                                     architecture=arch, recipe=rec, seed=seed, **r))
                q, tot, fl = op_coverage(qpath)
                cov.append(dict(architecture=arch, recipe=rec, fold=key, seed=seed, quantized_ops=q, compute_ops=tot,
                                share=q / max(tot, 1), float_op_types=" ".join(fl)))
                for p in onnx_dir.glob(f"{name}_*.onnx"):                # keep disk use bounded
                    try: p.unlink()
                    except OSError: pass
                pd.DataFrame(rows).to_csv(long_csv, index=False); pd.DataFrame(cov).to_csv(cov_csv, index=False)
                print(f"{a.unit} {key} {arch:<14} {rec:<6} seed {seed}: test bacc "
                      f"{np.mean([r['bacc_f32'] for r in rr]):.1f} (int8 {np.mean([r['bacc_int8'] for r in rr]):.1f}) "
                      f"| {rr[0]['epochs']} ep{' FAIL' if rr[0]['training_failure'] else ''}"
                      f"{' retried' if rr[0]['retried'] else ''} | {time.time() - t0:.0f}s")
    print(f"done -> {long_csv}")


if __name__ == "__main__":
    main()
