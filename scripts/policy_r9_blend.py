"""Blend stable base-B and local-lookahead v2 A/B for R9."""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def xy(feats, labels, keep):
    x, y = [], []
    for sf, label in zip(feats, labels):
        for d, row in sf[9].items():
            x.append(np.asarray(row)[keep]); y.append(d == label[9])
    return np.asarray(x, np.float32), np.asarray(y, np.int8)


def fit(kind, x, y):
    kw = dict(early_stopping=False, random_state=0)
    if kind == "A":
        return HistGradientBoostingClassifier(max_iter=800, learning_rate=.08, max_leaf_nodes=31,
                                              l2_regularization=1., **kw).fit(x, y)
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=.10, max_leaf_nodes=63,
                                          min_samples_leaf=5, l2_regularization=.5, **kw).fit(x, y)


def main():
    with (ROOT / "work" / "policy_features_v2.pkl").open("rb") as f: d = pickle.load(f)
    ft, yt = d["train"]; fv, yv = d["validation"]
    kb = list(range(106)); kv = kb + list(range(290, 319))
    xb, yb = xy(ft, yt, kb); xv, yx = xy(ft, yt, kv)
    mb = fit("B", xb, yb); ma = fit("A", xv, yx); mv = fit("B", xv, yx)
    pred = {w: [] for w in (0., .25, .5, .75, 1.)}
    for sf in fv:
        rows = sf[9]; dirs = list(rows)
        pb = mb.predict_proba(np.asarray([np.asarray(rows[d])[kb] for d in dirs], np.float32))[:, 1]
        pa = ma.predict_proba(np.asarray([np.asarray(rows[d])[kv] for d in dirs], np.float32))[:, 1]
        pv = mv.predict_proba(np.asarray([np.asarray(rows[d])[kv] for d in dirs], np.float32))[:, 1]
        p2 = (pa + pv) / 2
        for w in pred: pred[w].append(dirs[int(np.argmax((1-w)*pb + w*p2))])
    for w, p in pred.items(): print(f"v2_weight={w:.2f} validation={np.mean(np.asarray(p)==yv[:,9]):.4f}")


if __name__ == "__main__": main()
