"""CPU-only policy model experiments on ground-truth graphs and missions.

The expensive route-bank features are cached once.  This script deliberately
does not touch CV/NLP checkpoints or the production prediction path.
"""
import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from courier.data import load_split  # noqa: E402
from courier.policy import graph_from_scene, mission_from_scene, move_features  # noqa: E402


def featurize(scene):
    graph = graph_from_scene(scene)
    graph["style"] = scene["style"]
    return move_features(graph, mission_from_scene(scene))


def load_features(workers):
    cache = ROOT / "work" / "policy_features_v1.pkl"
    if cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    data = {}
    for split in ("train", "validation"):
        _, labels, scenes = load_split(split)
        with Pool(workers) as pool:
            feats = pool.map(featurize, scenes, chunksize=8)
        data[split] = (feats, np.asarray(labels, dtype=np.int8).reshape(-1, 10))
        print(f"cached features in memory: {split}={len(scenes)}", flush=True)
    with cache.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def candidate_xy(feats, labels, robot, add_robot=False):
    xs, ys = [], []
    robot_1h = [float(robot == r) for r in range(1, 10)]
    for scene_feats, scene_labels in zip(feats, labels):
        for direction, x in scene_feats[robot].items():
            xs.append(x + robot_1h if add_robot else x)
            ys.append(int(direction == scene_labels[robot]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8)


def predict_candidate(model, feats, robot, add_robot=False):
    out = []
    robot_1h = [float(robot == r) for r in range(1, 10)]
    for scene_feats in feats:
        rows = scene_feats[robot]
        dirs = list(rows)
        x = [rows[d] + robot_1h if add_robot else rows[d] for d in dirs]
        p = model.predict_proba(np.asarray(x, dtype=np.float32))[:, 1]
        out.append(dirs[int(np.argmax(p))])
    return np.asarray(out)


def scene_xy(feats, labels, robot):
    """One row per scene; preserve the four-way choice structure."""
    nfeat = len(next(iter(feats[0][robot].values())))
    xs = []
    for scene_feats in feats:
        rows = scene_feats[robot]
        row = []
        for direction in range(4):
            if direction in rows:
                row.extend(rows[direction])
                row.append(1.0)
            else:
                row.extend([0.0] * nfeat)
                row.append(0.0)
        xs.append(row)
    return np.asarray(xs, dtype=np.float32), labels[:, robot]


def pairwise_xy(feats, labels, robot):
    """Learn preference from feature differences, not independent labels."""
    xs, ys = [], []
    for scene_feats, scene_labels in zip(feats, labels):
        rows = scene_feats[robot]
        good = int(scene_labels[robot])
        if good not in rows:
            continue
        a = np.asarray(rows[good], dtype=np.float32)
        for direction, x in rows.items():
            if direction == good:
                continue
            delta = a - np.asarray(x, dtype=np.float32)
            xs.extend((delta, -delta))
            ys.extend((1, 0))
    return np.asarray(xs), np.asarray(ys, dtype=np.int8)


def predict_pairwise(model, feats, robot):
    out = []
    for scene_feats in feats:
        rows = scene_feats[robot]
        dirs = list(rows)
        score = np.zeros(len(dirs), dtype=np.float64)
        for i, di in enumerate(dirs):
            for j, dj in enumerate(dirs):
                if i == j:
                    continue
                delta = np.asarray(rows[di], dtype=np.float32) - np.asarray(rows[dj], dtype=np.float32)
                score[i] += model.predict_proba(delta[None])[:, 1][0]
        out.append(dirs[int(np.argmax(score))])
    return np.asarray(out)


def accuracy(pred, labels, robot):
    return float(np.mean(pred == labels[:, robot]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    data = load_features(args.workers)
    ft, yt = data["train"]
    fv, yv = data["validation"]

    results = {name: {} for name in ("candidate_hgb", "candidate_extra", "scene_hgb", "pairwise_hgb")}
    predictions = {name: {} for name in results}
    for robot in range(1, 10):
        x, y = candidate_xy(ft, yt, robot)
        model = HistGradientBoostingClassifier(max_iter=800, learning_rate=.08, max_leaf_nodes=31,
                                               l2_regularization=1., random_state=0).fit(x, y)
        pred = predict_candidate(model, fv, robot)
        results["candidate_hgb"][robot] = accuracy(pred, yv, robot)
        predictions["candidate_hgb"][robot] = pred

        model = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features=.75,
                                     class_weight="balanced", n_jobs=args.workers,
                                     random_state=0).fit(x, y)
        pred = predict_candidate(model, fv, robot)
        results["candidate_extra"][robot] = accuracy(pred, yv, robot)
        predictions["candidate_extra"][robot] = pred

        x, y = scene_xy(ft, yt, robot)
        model = HistGradientBoostingClassifier(max_iter=500, learning_rate=.06, max_leaf_nodes=31,
                                               l2_regularization=2., random_state=0).fit(x, y)
        xv, _ = scene_xy(fv, yv, robot)
        pred = model.predict(xv)
        results["scene_hgb"][robot] = accuracy(pred, yv, robot)
        predictions["scene_hgb"][robot] = pred

        x, y = pairwise_xy(ft, yt, robot)
        model = HistGradientBoostingClassifier(max_iter=500, learning_rate=.06, max_leaf_nodes=31,
                                               l2_regularization=2., random_state=0).fit(x, y)
        pred = predict_pairwise(model, fv, robot)
        results["pairwise_hgb"][robot] = accuracy(pred, yv, robot)
        predictions["pairwise_hgb"][robot] = pred
        print("R%d " % robot + " ".join(f"{k}={results[k][robot]:.4f}" for k in results), flush=True)

    print("\nmodel means (R1..R9)")
    for name, scores in results.items():
        print(f"{name}: {np.mean(list(scores.values())):.4f}")
    oracle = []
    majority = []
    for robot in range(1, 10):
        stack = np.stack([predictions[name][robot] for name in results])
        oracle.append(np.mean(np.any(stack == yv[:, robot][None, :], axis=0)))
        vote = np.asarray([np.bincount(stack[:, i], minlength=4).argmax() for i in range(stack.shape[1])])
        majority.append(accuracy(vote, yv, robot))
    print(f"model oracle mean: {np.mean(oracle):.4f}")
    print(f"majority-vote mean: {np.mean(majority):.4f}")


if __name__ == "__main__":
    main()
