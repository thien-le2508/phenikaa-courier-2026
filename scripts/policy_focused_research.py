"""Focused policy experiments after broad model-family screening.

Uses the cached route-bank features produced by policy_research.py.  Tests an
average of two strong rankers and condition-specific experts suggested by the
public robot hints.  Research only: it does not alter the inference pipeline.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from courier.data import load_split  # noqa: E402


def xy(feats, labels, robot, indices=None):
    if indices is None:
        indices = range(len(feats))
    xs, ys = [], []
    for i in indices:
        for direction, row in feats[i][robot].items():
            xs.append(row)
            ys.append(int(direction == labels[i, robot]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8)


def scene_prob(model, scene_feat, robot):
    rows = scene_feat[robot]
    dirs = list(rows)
    prob = model.predict_proba(np.asarray([rows[d] for d in dirs], dtype=np.float32))[:, 1]
    return dirs, prob


def predict(model, feats, robot):
    return np.asarray([lambda_result(model, f, robot) for f in feats], dtype=np.int8)


def lambda_result(model, scene_feat, robot):
    dirs, prob = scene_prob(model, scene_feat, robot)
    return dirs[int(np.argmax(prob))]


def fit_a(x, y):
    return HistGradientBoostingClassifier(max_iter=800, learning_rate=.08, max_leaf_nodes=31,
                                          l2_regularization=1., random_state=0).fit(x, y)


def fit_b(x, y):
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=.10, max_leaf_nodes=63,
                                          min_samples_leaf=5, l2_regularization=.5,
                                          random_state=0).fit(x, y)


def main():
    with (ROOT / "work" / "policy_features_v1.pkl").open("rb") as f:
        data = pickle.load(f)
    ft, yt = data["train"]
    fv, yv = data["validation"]
    _, _, train_scenes = load_split("train")
    _, _, val_scenes = load_split("validation")

    split_for_robot = {3: "rain", 4: "rain", 6: "urgent", 7: "fragile"}
    scores = []
    for robot in range(1, 10):
        x, y = xy(ft, yt, robot)
        a, b = fit_a(x, y), fit_b(x, y)
        pa, pb, pe = [], [], []
        for scene_feat in fv:
            dirs, qa = scene_prob(a, scene_feat, robot)
            _, qb = scene_prob(b, scene_feat, robot)
            pa.append(dirs[int(np.argmax(qa))])
            pb.append(dirs[int(np.argmax(qb))])
            pe.append(dirs[int(np.argmax((qa + qb) / 2))])
        pa, pb, pe = map(np.asarray, (pa, pb, pe))
        line = {
            "A": np.mean(pa == yv[:, robot]),
            "B": np.mean(pb == yv[:, robot]),
            "AB": np.mean(pe == yv[:, robot]),
        }

        if robot in split_for_robot:
            field = split_for_robot[robot]
            if field == "rain":
                tr_key = np.asarray([s["weather"] == "rain" for s in train_scenes])
                va_key = np.asarray([s["weather"] == "rain" for s in val_scenes])
            else:
                tr_key = np.asarray([bool(s["mission"][field]) for s in train_scenes])
                va_key = np.asarray([bool(s["mission"][field]) for s in val_scenes])
            experts = {}
            for key in (False, True):
                xx, yy = xy(ft, yt, robot, np.flatnonzero(tr_key == key))
                experts[key] = fit_a(xx, yy)
            pp = np.asarray([lambda_result(experts[bool(key)], feat, robot)
                             for feat, key in zip(fv, va_key)])
            line[field + "_experts"] = np.mean(pp == yv[:, robot])
            line[field + "_false"] = np.mean(pa[~va_key] == yv[~va_key, robot])
            line[field + "_true"] = np.mean(pa[va_key] == yv[va_key, robot])
        scores.append(line)
        print(f"R{robot}: " + " ".join(f"{k}={v:.4f}" for k, v in line.items()), flush=True)

    for key in ("A", "B", "AB"):
        print(f"mean {key}: {np.mean([s[key] for s in scores]):.4f}")


if __name__ == "__main__":
    main()
