"""Candidate rankers on ML2 features (optionally + v1 bank features). Honest protocol:
5-fold scene-grouped OOF on train (KFold seed 2026, same as ml2_baseline) + train-only model on validation.

    python scripts/ml2_rank.py <featset: ml2|ml2v1|ml2v2> <robots> [model: hgb|hgb2]
Writes work/ml2/rank_<featset>_<model>.json and _probs.pkl ({robot: {oof: [n,4], val: [m,4]}}).
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ml2"


def load_feats(featset):
    tr = pickle.loads((OUT / "feats_train.pkl").read_bytes())
    va = pickle.loads((OUT / "feats_validation.pkl").read_bytes())
    extra = None
    if featset in ("ml2v1", "ml2v2"):
        name = "policy_features_v1.pkl" if featset == "ml2v1" else "policy_features_v2.pkl"
        extra = pickle.loads((ROOT / "work" / name).read_bytes())
    return tr, va, extra


def scene_matrix(d, legged, ext=None, robot=None):
    """X [B,4,f], legal [B,4]."""
    X = d["res"][legged]["X"]
    legal = d["res"][legged]["legal"].copy()
    if ext is not None:
        feats = ext[0]
        k = len(next(iter(feats[0][robot].values())))
        Z = np.zeros((X.shape[0], 4, k), np.float32)
        for i, sf in enumerate(feats):
            for dd, row in sf[robot].items():
                Z[i, dd] = row
            # v1 legal set is authoritative for consistency with the old ranker
            for dd in range(4):
                if dd not in sf[robot]:
                    legal[i, dd] = False
        X = np.concatenate([X, Z], -1)
    return X, legal


def fit(model, x, y):
    if model == "hgb":
        m = HistGradientBoostingClassifier(max_iter=800, learning_rate=.08, max_leaf_nodes=31,
                                           l2_regularization=1., early_stopping=False, random_state=0)
    else:
        m = HistGradientBoostingClassifier(max_iter=500, learning_rate=.06, max_leaf_nodes=15,
                                           min_samples_leaf=30, l2_regularization=2., max_features=.5,
                                           early_stopping=False, random_state=0)
    return m.fit(x, y)


def rows(X, legal, Y, idx):
    ii, dd = np.nonzero(legal[idx])
    x = X[idx][ii, dd]
    y = (Y[idx][ii] == dd).astype(np.int8)
    return x, y


def predict(m, X, legal, idx):
    P = np.full((len(idx), 4), -1.0)
    ii, dd = np.nonzero(legal[idx])
    if len(ii):
        P[ii, dd] = m.predict_proba(X[idx][ii, dd])[:, 1]
    return P


def main():
    featset = sys.argv[1]
    robots = [int(x) for x in sys.argv[2].split(",")]
    model = sys.argv[3] if len(sys.argv) > 3 else "hgb"
    tr, va, ext = load_feats(featset)
    tag = f"{featset}_{model}"
    rp, pp = OUT / f"rank_{tag}.json", OUT / f"rank_{tag}_probs.pkl"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    probs = pickle.loads(pp.read_bytes()) if pp.exists() else {}
    n = len(tr["Y"])
    kf = list(KFold(5, shuffle=True, random_state=2026).split(np.arange(n)))
    for r in robots:
        t0 = time.time()
        legged = r == 4
        Xt, Lt = scene_matrix(tr, legged, ext and ext["train"], r)
        Xv, Lv = scene_matrix(va, legged, ext and ext["validation"], r)
        yt, yv = tr["Y"][:, r], va["Y"][:, r]
        oof = np.full((n, 4), -1.0)
        for a, b in kf:
            m = fit(model, *rows(Xt, Lt, yt, a))
            oof[b] = predict(m, Xt, Lt, b)
        m = fit(model, *rows(Xt, Lt, yt, np.arange(n)))
        pv = predict(m, Xv, Lv, np.arange(len(yv)))
        res[str(r)] = dict(oof=float(np.mean(oof.argmax(1) == yt)), val=float(np.mean(pv.argmax(1) == yv)))
        probs[r] = dict(oof=oof, val=pv)
        print(f"{tag} R{r} OOF={res[str(r)]['oof']:.4f} val={res[str(r)]['val']:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        rp.write_text(json.dumps(res, indent=1))
        pp.write_bytes(pickle.dumps(probs))


if __name__ == "__main__":
    main()
