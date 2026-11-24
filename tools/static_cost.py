#!/usr/bin/env python3
"""Table 9: static cost of the four architectures per dataset input shape.

Theoretical quantities only, read off the exported ONNX graph of the deployed form
(BN folded, EEGNet spatial-first) by the CSOANet2026 repository's own profiler
(deploy/profile_model.py), so that the definitions match the deployment paper:
  params      weights stored in the graph after BN folding
  flash_kB    the same weights as int8 (biases as int32)
  MMACs       multiply-accumulates of Conv / Gemm plus element-wise tensor products
  peak_kB     largest set of activations alive at once, int8, liveness over the graph
  stream_kB   the same without the raw input window
No runtime, latency or energy is measured. Models are untrained; costs do not depend
on the weights or on the recipe.

Usage (repository root, same environment as the confirmatory runs):
  python -m tools.static_cost --csoanet-root ..\\csoanet2026_v2_deploy --out docs\\static_cost_V1.0.csv
"""
from __future__ import annotations
import argparse
import pandas as pd
import torch
from train.pilot_binary import ARCHS, load_deploy
from train.confirmatory import build

SHAPES = {"Lee2019_MI (F1a, F2)": (62, 1000), "Cho2017 (F1b)": (64, 750)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csoanet-root", required=True)
    ap.add_argument("--out", default="docs/static_cost_V1.0.csv")
    a = ap.parse_args()
    deps = load_deploy(a.csoanet_root)
    tp = deps[0]
    from deploy.profile_model import profile_model      # importable once load_deploy set sys.path
    rows = []
    for ds, (ch, t) in SHAPES.items():
        for arch in ARCHS:
            torch.manual_seed(0)
            m = build(arch, ch, t, False, deps).cpu().eval()
            p_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
            prof = profile_model(tp.deploy_form(m), torch.zeros(1, 1, ch, t))
            rows.append(dict(dataset=ds, architecture=arch, channels=ch, samples=t,
                             params_train=p_train, params_deployed=int(prof["params"]),
                             flash_kB=round(prof["flash_bytes"] / 1024, 1),
                             MMACs=round(prof["macs"] / 1e6, 3),
                             peak_kB=round(prof["peak_ram_bytes"] / 1024, 1),
                             stream_kB=round(prof["peak_ram_stream_bytes"] / 1024, 1),
                             peak_at=prof["peak_at"]))
            print({k: v for k, v in rows[-1].items() if k != "peak_at"})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("written", a.out)


if __name__ == "__main__":
    main()
