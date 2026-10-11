"""Prototype: learned per-robot move ranker on top of a bank of cost-model path costs.

    COURIER_DATA=delivery_public python scripts/policy_ml.py [--save]
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.data import load_split  # noqa: E402
from courier import policy as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def featurize_scene(s):
    g, mi = P.graph_from_scene(s), P.mission_from_scene(s)
    g["style"] = s["style"]
    return P.move_features(g, mi)


import json, os
HP = json.loads(os.environ.get("HP", '{"max_iter": 200, "learning_rate": 0.05, "max_depth": 4, "min_samples_leaf": 20, "l2_regularization": 1.0, "random_state": 0}'))


def main():
    save = "--save" in sys.argv
    data = {}
    with Pool(15) as pool:
        for split in ("train", "validation"):
            rows, labels, scenes = load_split(split)
            feats = pool.map(featurize_scene, scenes, chunksize=8)
            data[split] = (feats, np.array(labels).reshape(-1, 10))
    from sklearn.ensemble import HistGradientBoostingClassifier
    models = {}
    for r in range(10):
        if r == 0:
            continue
        X, y = [], []
        for F, Y in zip(*data["train"]):
            fr = F[r]
            for d, x in fr.items():
                X.append(x)
                y.append(int(d == Y[r]))
        clf = HistGradientBoostingClassifier(**HP)
        clf.fit(np.array(X), np.array(y))
        models[r] = clf
        for split in ("train", "validation"):
            ok = n = 0
            for F, Y in zip(*data[split]):
                fr = F[r]
                if not fr:
                    continue
                ds = list(fr)
                p = clf.predict_proba(np.array([fr[d] for d in ds]))[:, 1]
                ok += ds[int(np.argmax(p))] == Y[r]
                n += 1
            print(f"R{r} {split}: {ok / n:.4f}", flush=True)
    if save:
        with open(ROOT / "work" / "policy_ml.pkl", "wb") as f:
            pickle.dump(models, f)


if __name__ == "__main__":
    main()
