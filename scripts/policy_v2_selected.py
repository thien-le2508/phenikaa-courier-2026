"""Confirm selected v2 features for R7/R9 with A/B and scene-level OOF."""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def xy(feats, labels, robot, keep, indices=None):
    indices = range(len(feats)) if indices is None else indices
    x, y = [], []
    for i in indices:
        for direction, row in feats[i][robot].items():
            x.append(np.asarray(row)[keep])
            y.append(direction == labels[i, robot])
    return np.asarray(x, np.float32), np.asarray(y, np.int8)


def probs(model, feats, robot, keep):
    result = []
    for sf in feats:
        rows = sf[robot]
        dirs = list(rows)
        x = np.asarray([np.asarray(rows[d])[keep] for d in dirs], np.float32)
        result.append((dirs, model.predict_proba(x)[:, 1]))
    return result


def decisions(ps):
    return np.asarray([dirs[int(np.argmax(p))] for dirs, p in ps])


def fit_a(x, y):
    return HistGradientBoostingClassifier(max_iter=800, learning_rate=.08, max_leaf_nodes=31,
                                          l2_regularization=1., early_stopping=False,
                                          random_state=0).fit(x, y)


def fit_b(x, y):
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=.10, max_leaf_nodes=63,
                                          min_samples_leaf=5, l2_regularization=.5,
                                          early_stopping=False, random_state=0).fit(x, y)


def oof_score(feats, labels, robot, keep):
    pred = np.zeros(len(feats), np.int8)
    kf = KFold(5, shuffle=True, random_state=2026)
    for tr, va in kf.split(np.arange(len(feats))):
        x, y = xy(feats, labels, robot, keep, tr)
        model = fit_a(x, y)
        pred[va] = decisions(probs(model, [feats[i] for i in va], robot, keep))
    return float(np.mean(pred == labels[:, robot]))


def main():
    with (ROOT / "work" / "policy_features_v2.pkl").open("rb") as f:
        data = pickle.load(f)
    ft, yt = data["train"]
    fv, yv = data["validation"]
    selected = {7: list(range(319)), 9: list(range(106)) + list(range(290, 319))}
    for robot, keep in selected.items():
        x, y = xy(ft, yt, robot, keep)
        a, b = fit_a(x, y), fit_b(x, y)
        pa, pb = probs(a, fv, robot, keep), probs(b, fv, robot, keep)
        da, db = decisions(pa), decisions(pb)
        pab = [(d1, (p1 + p2) / 2) for (d1, p1), (_, p2) in zip(pa, pb)]
        dab = decisions(pab)
        baseline_oof = oof_score(ft, yt, robot, list(range(106)))
        selected_oof = oof_score(ft, yt, robot, keep)
        print(f"R{robot}: val A={np.mean(da == yv[:, robot]):.4f} "
              f"B={np.mean(db == yv[:, robot]):.4f} AB={np.mean(dab == yv[:, robot]):.4f} "
              f"OOF base={baseline_oof:.4f} selected={selected_oof:.4f}", flush=True)


if __name__ == "__main__":
    main()
