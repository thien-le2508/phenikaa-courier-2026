"""Targeted v2 feature ablation for the weak robots R3, R7 and R9."""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def xy(feats, labels, robot, keep):
    x, y = [], []
    for sf, label in zip(feats, labels):
        for direction, row in sf[robot].items():
            x.append(np.asarray(row)[keep])
            y.append(direction == label[robot])
    return np.asarray(x, np.float32), np.asarray(y, np.int8)


def predict(model, feats, robot, keep):
    out = []
    for sf in feats:
        rows = sf[robot]
        dirs = list(rows)
        x = np.asarray([np.asarray(rows[d])[keep] for d in dirs], np.float32)
        out.append(dirs[int(np.argmax(model.predict_proba(x)[:, 1]))])
    return np.asarray(out)


def main():
    with (ROOT / "work" / "policy_features_v2.pkl").open("rb") as f:
        data = pickle.load(f)
    ft, yt = data["train"]
    fv, yv = data["validation"]
    base = list(range(106))
    local_shared = list(range(290, 319))
    summary_core = []
    for k in range(8):
        # Per bank: 3 cost/margin features + 10 aggregate path counters.
        summary_core.extend(range(106 + 23 * k, 106 + 23 * k + 13))
    variants = {
        "base": base,
        "base_local": base + local_shared,
        "base_summary_core": base + summary_core + local_shared,
        "v2_all": list(range(319)),
    }
    for robot in (3, 7, 9):
        line = []
        for name, keep in variants.items():
            x, y = xy(ft, yt, robot, keep)
            m = HistGradientBoostingClassifier(max_iter=800, learning_rate=.08,
                                               max_leaf_nodes=31, l2_regularization=1.,
                                               early_stopping=False, random_state=0).fit(x, y)
            p = predict(m, fv, robot, keep)
            line.append(f"{name}={np.mean(p == yv[:, robot]):.4f}")
        print(f"R{robot}: " + " ".join(line), flush=True)


if __name__ == "__main__":
    main()
