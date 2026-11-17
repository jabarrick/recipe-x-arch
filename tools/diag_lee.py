"""Diagnostic only (not part of the confirmatory code): how MOABB returns Lee2019-MI sessions for subject 1."""
import moabb, moabb.datasets as MD, numpy as np, mne
from moabb.paradigms import LeftRightImagery
mne.set_log_level("ERROR")
print("moabb", moabb.__version__, "| MNE_DATA", mne.get_config("MNE_DATA"))
raw = MD.Lee2019_MI().get_data(subjects=[1])
print("dataset.get_data sessions/runs:", {s: list(r.keys()) for s, r in raw[1].items()})
p = LeftRightImagery(fmin=1.0, fmax=40.0, resample=250.0)
for sess in [(1,), (2,), (1, 2)]:
    X, l, m = p.get_data(dataset=MD.Lee2019_MI(sessions=sess), subjects=[1])
    print(f"sessions={sess}: X{X.shape} sessions={m.session.unique().tolist()} runs={m.run.unique().tolist()} "
          f"fingerprint={float(X[0, 0, :50].sum()):.6e}")
