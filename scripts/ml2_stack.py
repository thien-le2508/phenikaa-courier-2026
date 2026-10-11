"""Level-2 stacker over base-model probabilities (OOF on train, train-only on val).

Per move features: normalised prob of each base model, its rank within the scene, is-argmax, plus
context (relative turn, abs dir, rain/urgent/fragile/via/night).
    python scripts/ml2_stack.py out_name base1 base2 ...
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml2_core as c  # noqa: E402
from ml2_ensemble import labels, norm  # noqa: E402

OUT = c.ROOT / "work" / "ml2"
REL = np.array([[c.relative_turn(h, d) for d in range(4)] for h in range(4)])


def ctx(d):
    B = len(d["scenes"])
    X = np.zeros((B, 4, 13), np.float32)
    for i, o in enumerate(d["scenes"]):
        h = o["heading"]
        sc = list(d["conds"][i].astype(float)) + [float(bool(o["vias"])), float(d["style"][i] == "night")]
        for dd in range(4):
            X[i, dd] = sc + [float(REL[h, dd] == k) for k in range(4)] + [float(dd == k) for k in range(4)]
    return X


def base_feats(ps):
    out = []
    for p in ps:
        q = norm(p)
        rank = (-q).argsort(1).argsort(1).astype(np.float32)
        out += [q[..., None], rank[..., None], (rank == 0)[..., None].astype(np.float32)]
    return np.concatenate(out, -1)


def fit(x, y):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1., early_stopping=False, random_state=0).fit(x, y)


def main():
    name, bases = sys.argv[1], sys.argv[2:]
    probs = {n: pickle.loads((OUT / f"{n}_probs.pkl").read_bytes()) for n in bases}
    tr, va = c.cache("train"), c.cache("validation")
    Ct, Cv = ctx(tr), ctx(va)
    yt, yv = labels()
    n = len(yt)
    kf = list(KFold(5, shuffle=True, random_state=7).split(np.arange(n)))
    res, out = {}, {}
    for r in range(1, 10):
        if not all(r in probs[b] for b in bases):
            continue
        Xt = np.concatenate([base_feats([probs[b][r]["oof"] for b in bases]), Ct], -1)
        Xv = np.concatenate([base_feats([probs[b][r]["val"] for b in bases]), Cv], -1)
        legal_t = np.max([probs[b][r]["oof"] for b in bases], 0) > 0
        legal_v = np.max([probs[b][r]["val"] for b in bases], 0) > 0
        oof = np.full((n, 4), -1.0)
        for a, b in kf:
            ii, dd = np.nonzero(legal_t[a])
            m = fit(Xt[a][ii, dd], (yt[a, r][ii] == dd).astype(int))
            ii, dd = np.nonzero(legal_t[b])
            oof[b[ii], dd] = m.predict_proba(Xt[b][ii, dd])[:, 1]
        ii, dd = np.nonzero(legal_t)
        m = fit(Xt[ii, dd], (yt[ii, r] == dd).astype(int))
        pv = np.full((len(yv), 4), -1.0)
        ii, dd = np.nonzero(legal_v)
        pv[ii, dd] = m.predict_proba(Xv[ii, dd])[:, 1]
        res[r] = dict(oof=float(np.mean(oof.argmax(1) == yt[:, r])), val=float(np.mean(pv.argmax(1) == yv[:, r])))
        out[r] = dict(oof=oof, val=pv)
        print(f"{name} R{r} OOF={res[r]['oof']:.4f} val={res[r]['val']:.4f}", flush=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"{name}_probs.pkl").write_bytes(pickle.dumps(out))


if __name__ == "__main__":
    main()
