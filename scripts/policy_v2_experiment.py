"""Build v2 structured features and compare them with the current ranker."""
import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from courier.data import load_split  # noqa: E402
from courier.policy import graph_from_scene, mission_from_scene  # noqa: E402
from courier.policy_features_v2 import move_features_v2  # noqa: E402


def featurize(scene):
    graph = graph_from_scene(scene)
    graph["style"] = scene["style"]
    return move_features_v2(graph, mission_from_scene(scene))


def load_data(workers):
    cache = ROOT / "work" / "policy_features_v2.pkl"
    if cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    data = {}
    for split in ("train", "validation"):
        _, labels, scenes = load_split(split)
        with Pool(workers) as pool:
            feats = pool.map(featurize, scenes, chunksize=4)
        data[split] = (feats, np.asarray(labels, np.int8).reshape(-1, 10))
        print(f"features {split}: {len(feats)} scenes, dim={len(next(iter(feats[0][1].values())))}", flush=True)
    with cache.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def xy(feats, labels, robot, scene_indices=None):
    scene_indices = range(len(feats)) if scene_indices is None else scene_indices
    x, y, groups = [], [], []
    for i in scene_indices:
        for direction, row in feats[i][robot].items():
            x.append(row)
            y.append(direction == labels[i, robot])
            groups.append(i)
    return np.asarray(x, np.float32), np.asarray(y, np.int8), np.asarray(groups)


def model():
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=.06, max_leaf_nodes=31,
                                          min_samples_leaf=12, l2_regularization=2.,
                                          early_stopping=False, random_state=0)


def predict(m, feats, robot):
    out = []
    for scene in feats:
        rows = scene[robot]
        dirs = list(rows)
        p = m.predict_proba(np.asarray([rows[d] for d in dirs], np.float32))[:, 1]
        out.append(dirs[int(np.argmax(p))])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    data = load_data(args.workers)
    ft, yt = data["train"]
    fv, yv = data["validation"]
    val_scores, oof_scores = [], []
    splitter = GroupKFold(args.folds)
    scene_groups = np.arange(len(ft))
    for robot in range(1, 10):
        x, y, groups = xy(ft, yt, robot)
        oof = np.zeros(len(ft), np.int8)
        for tr_scene, va_scene in splitter.split(scene_groups, groups=scene_groups):
            train_mask = np.isin(groups, tr_scene)
            m = model().fit(x[train_mask], y[train_mask])
            fold_feats = [ft[i] for i in va_scene]
            oof[va_scene] = predict(m, fold_feats, robot)
        oof_score = np.mean(oof == yt[:, robot])
        final = model().fit(x, y)
        vp = predict(final, fv, robot)
        val_score = np.mean(vp == yv[:, robot])
        oof_scores.append(oof_score)
        val_scores.append(val_score)
        print(f"R{robot}: OOF={oof_score:.4f} validation={val_score:.4f}", flush=True)
    print(f"mean R1..R9: OOF={np.mean(oof_scores):.4f} validation={np.mean(val_scores):.4f}")
    print(f"macro with exact R0: OOF={(1 + sum(oof_scores))/10:.4f} validation={(1 + sum(val_scores))/10:.4f}")


if __name__ == "__main__":
    main()
