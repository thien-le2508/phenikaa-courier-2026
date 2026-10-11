"""Honest evaluation of the existing candidate ranker (HGB 'A' on v1 bank features, and v2 features).

Protocol: 5-fold scene-grouped OOF on train (KFold over scenes, seed 2026) + train-only model on validation.
Writes work/ml2/baseline_<featset>.json and per-scene OOF probabilities for later ensembling.
    python scripts/ml2_baseline.py v1|v2
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ml2"


def xy(feats, labels, robot, idx):
    x, y = [], []
    for i in idx:
        for d, row in feats[i][robot].items():
            x.append(row)
            y.append(d == labels[i, robot])
    return np.asarray(x, np.float32), np.asarray(y, np.int8)


def scene_probs(model, feats, robot, idx):
    """[len(idx), 4] probability (-1 for illegal)."""
    out = np.full((len(idx), 4), -1.0)
    rows, own = [], []
    for j, i in enumerate(idx):
        for d, row in feats[i][robot].items():
            rows.append(row)
            own.append((j, d))
    if rows:
        p = model.predict_proba(np.asarray(rows, np.float32))[:, 1]
        for (j, d), pp in zip(own, p):
            out[j, d] = pp
    return out


def fit_a(x, y):
    return HistGradientBoostingClassifier(max_iter=800, learning_rate=.08, max_leaf_nodes=31,
                                          l2_regularization=1., early_stopping=False, random_state=0).fit(x, y)


def main():
    fs = sys.argv[1] if len(sys.argv) > 1 else "v1"
    robots = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(1, 10))
    path = ROOT / "work" / ("policy_features_v1.pkl" if fs == "v1" else "policy_features_v2.pkl")
    data = pickle.loads(path.read_bytes())
    ft, yt = data["train"]
    fv, yv = data["validation"]
    yt, yv = np.asarray(yt), np.asarray(yv)
    n = len(ft)
    kf = list(KFold(5, shuffle=True, random_state=2026).split(np.arange(n)))
    res = {}
    probs = {}
    for r in robots:
        oof = np.full((n, 4), -1.0)
        for tr, va in kf:
            x, y = xy(ft, yt, r, tr)
            m = fit_a(x, y)
            oof[va] = scene_probs(m, ft, r, va)
        x, y = xy(ft, yt, r, range(n))
        m = fit_a(x, y)
        pv = scene_probs(m, fv, r, range(len(fv)))
        oacc = float(np.mean(oof.argmax(1) == yt[:, r]))
        vacc = float(np.mean(pv.argmax(1) == yv[:, r]))
        res[r] = dict(oof=oacc, val=vacc)
        probs[r] = dict(oof=oof, val=pv)
        print(f"{fs} R{r} OOF={oacc:.4f} val={vacc:.4f}", flush=True)
        (OUT / f"baseline_{fs}.json").write_text(json.dumps(res, indent=1))
        (OUT / f"baseline_{fs}_probs.pkl").write_bytes(pickle.dumps(probs))


if __name__ == "__main__":
    main()
