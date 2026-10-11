"""Ablate nuisance features from the learned policy ranker (research only)."""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def dataset(feats, labels, robot, keep):
    x, y, groups = [], [], []
    for i, (scene_feat, label) in enumerate(zip(feats, labels)):
        for direction, row in scene_feat[robot].items():
            x.append(np.asarray(row)[keep])
            y.append(direction == label[robot])
            groups.append((i, direction))
    return np.asarray(x, np.float32), np.asarray(y, np.int8), groups


def fit_predict(ft, yt, fv, robot, keep):
    x, y, _ = dataset(ft, yt, robot, keep)
    model = HistGradientBoostingClassifier(max_iter=800, learning_rate=.08, max_leaf_nodes=31,
                                           l2_regularization=1., random_state=0).fit(x, y)
    xv, _, groups = dataset(fv, np.zeros((len(fv), 10), dtype=np.int8), robot, keep)
    prob = model.predict_proba(xv)[:, 1]
    pred = np.zeros(len(fv), dtype=np.int8)
    pos = 0
    for i, scene_feat in enumerate(fv):
        dirs = list(scene_feat[robot])
        n = len(dirs)
        pred[i] = dirs[int(np.argmax(prob[pos:pos + n]))]
        pos += n
    return pred


def main():
    with (ROOT / "work" / "policy_features_v1.pkl").open("rb") as f:
        data = pickle.load(f)
    ft, yt = data["train"]
    fv, yv = data["validation"]
    n = len(next(iter(ft[0][1].values())))
    all_features = np.arange(n)
    variants = {
        # style one-hot occupies [87, 91)
        "no_style": np.delete(all_features, np.arange(87, 91)),
        # Also remove absolute move [61,65) and absolute heading [102,106).
        "rotation_invariant": np.delete(all_features, np.r_[61:65, 87:91, 102:106]),
        # Goal type can legitimately affect only a specialized robot; test pure route behavior.
        "route_context": np.delete(all_features, np.r_[61:65, 87:91, 91:102, 102:106]),
    }
    scores = {name: [] for name in variants}
    for robot in range(1, 10):
        line = []
        for name, keep in variants.items():
            pred = fit_predict(ft, yt, fv, robot, keep)
            score = float(np.mean(pred == yv[:, robot]))
            scores[name].append(score)
            line.append(f"{name}={score:.4f}")
        print(f"R{robot}: " + " ".join(line), flush=True)
    for name, values in scores.items():
        print(f"mean {name}: {np.mean(values):.4f}")


if __name__ == "__main__":
    main()
