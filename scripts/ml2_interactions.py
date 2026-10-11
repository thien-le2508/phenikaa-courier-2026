"""Train-only study of context-dependent robot policies.

The experiment treats each labelled move as a set of pairwise preferences:
the chosen legal move must outrank every other legal move in the same scene.
It compares a shared linear utility with progressively richer, regularised
context interactions.  Evaluation is 5-fold scene-level OOF on train only.

By default no validation or test files are read.  Passing ``validation`` as
the third argument evaluates the already specified experiment on the public
validation split; test files are never read.

    python scripts/ml2_interactions.py [robots] [variants]
    python scripts/ml2_interactions.py 1,2,3,4,5,6,7,8,9 global,main,pair,style
    python scripts/ml2_interactions.py 9 style validation
"""
import json
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ml2"

# courier.policy.move_features appends a 23-column scene-only suffix.  Scene-only
# columns cancel in within-scene action differences; context enters explicitly
# below by modulating the action-dependent prefix.
SCENE_SUFFIX = 23
FLAGS = ("rain", "urgent", "fragile", "via")
STYLES = ("classic", "night", "print", "sketch")


def load_train():
    feature_file = ROOT / "work" / "policy_features_v1.pkl"
    state_file = OUT / "states_train.pkl"
    data = pickle.loads(feature_file.read_bytes())
    feats, labels = data["train"]
    states = pickle.loads(state_file.read_bytes())
    labels = np.asarray(labels, dtype=np.int64)
    if len(feats) != len(labels) or len(feats) != len(states["conds"]):
        raise ValueError("train feature/state lengths do not match")
    ctx = np.column_stack([states["conds"].astype(float), states["has_via"].astype(float)])
    styles = np.asarray([[s == name for name in STYLES] for s in states["style"]], dtype=float)
    return data, feats, labels, ctx, styles


def load_validation(data):
    """Load the public validation split only when explicitly requested."""
    feats, labels = data["validation"]
    states = pickle.loads((OUT / "states_validation.pkl").read_bytes())
    labels = np.asarray(labels, dtype=np.int64)
    ctx = np.column_stack([states["conds"].astype(float), states["has_via"].astype(float)])
    styles = np.asarray([[s == name for name in STYLES] for s in states["style"]], dtype=float)
    return feats, labels, ctx, styles


def raw_actions(feats, robot):
    """Return action-varying features [scene, direction, feature] and legal mask."""
    sample = next(iter(feats[0][robot].values()))
    width = len(sample) - SCENE_SUFFIX
    if width <= 0:
        raise ValueError("unexpected policy feature width")
    X = np.zeros((len(feats), 4, width), dtype=np.float64)
    legal = np.zeros((len(feats), 4), dtype=bool)
    for i, scene in enumerate(feats):
        for direction, row in scene[robot].items():
            X[i, direction] = np.asarray(row[:-SCENE_SUFFIX], dtype=np.float64)
            legal[i, direction] = True
    return X, legal


def design(X, ctx, styles, variant):
    """Hierarchical utility features; shared block shrinks condition blocks."""
    blocks = [("shared", X)]
    if variant in ("main", "pair", "style"):
        for j, name in enumerate(FLAGS):
            blocks.append((name, X * ctx[:, None, j, None]))
    if variant == "pair":
        for a, b in combinations(range(len(FLAGS)), 2):
            blocks.append((f"{FLAGS[a]}*{FLAGS[b]}",
                           X * (ctx[:, a] * ctx[:, b])[:, None, None]))
    if variant == "style":
        # Omit classic as the reference level.
        for j, name in enumerate(STYLES[1:], start=1):
            blocks.append((f"style={name}", X * styles[:, None, j, None]))
    if variant not in ("global", "main", "pair", "style"):
        raise ValueError(f"unknown variant: {variant}")
    return np.concatenate([b for _, b in blocks], axis=2), [name for name, _ in blocks]


def pair_rows(X, legal, y, indices):
    diffs = []
    for i in indices:
        chosen = int(y[i])
        if not legal[i, chosen]:
            continue
        for other in np.flatnonzero(legal[i]):
            if other != chosen:
                diffs.append(X[i, chosen] - X[i, other])
    if not diffs:
        raise ValueError("no pairwise training rows")
    pos = np.asarray(diffs, dtype=np.float64)
    return np.concatenate([pos, -pos]), np.concatenate([np.ones(len(pos)), np.zeros(len(pos))])


def fit_model(X, legal, y, train_idx, c_value=0.1):
    x, yy = pair_rows(X, legal, y, train_idx)
    scaler = StandardScaler(with_mean=False).fit(x)
    model = LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", fit_intercept=False,
                               max_iter=600, tol=1e-5).fit(scaler.transform(x), yy)
    return model, scaler


def predict_model(model, scaler, X, legal, test_idx):
    pred = np.full(len(test_idx), -1, dtype=np.int64)
    for j, i in enumerate(test_idx):
        directions = np.flatnonzero(legal[i])
        if len(directions):
            score = model.decision_function(scaler.transform(X[i, directions]))
            pred[j] = int(directions[int(np.argmax(np.atleast_1d(score)))])
    return pred


def fit_predict(X, legal, y, train_idx, test_idx, c_value=0.1):
    model, scaler = fit_model(X, legal, y, train_idx, c_value)
    return predict_model(model, scaler, X, legal, test_idx), model, scaler


def breakdown(pred, y, ctx):
    out = {}
    hit = pred == y
    for j, name in enumerate(FLAGS):
        out[name] = {}
        for value in (0, 1):
            mask = ctx[:, j] == value
            out[name][str(value)] = {"n": int(mask.sum()), "acc": round(float(hit[mask].mean()), 5)}
    group = (ctx.astype(np.int64) * np.array([8, 4, 2, 1])).sum(1)
    out["groups"] = {}
    for g in range(16):
        mask = group == g
        if mask.any():
            out["groups"][format(g, "04b")] = {
                "n": int(mask.sum()), "acc": round(float(hit[mask].mean()), 5)
            }
    return out


def main():
    robots = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3,4,5,6,7,8,9").split(",")]
    variants = (sys.argv[2] if len(sys.argv) > 2 else "global,main,pair,style").split(",")
    validate = len(sys.argv) > 3 and sys.argv[3] == "validation"
    data, feats, labels, ctx, styles = load_train()
    validation = load_validation(data) if validate else None
    folds = list(KFold(5, shuffle=True, random_state=2026).split(np.arange(len(feats))))
    result_path = OUT / "interaction_study.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        "protocol": "5-fold scene OOF on train only; no validation/test read",
        "flags": list(FLAGS), "robots": {}
    }
    for robot in robots:
        base, legal = raw_actions(feats, robot)
        result["robots"].setdefault(str(robot), {})
        for variant in variants:
            t0 = time.time()
            X, block_names = design(base, ctx, styles, variant)
            oof = np.full(len(feats), -1, dtype=np.int64)
            fold_acc = []
            for train_idx, test_idx in folds:
                pred, _, _ = fit_predict(X, legal, labels[:, robot], train_idx, test_idx)
                oof[test_idx] = pred
                fold_acc.append(float(np.mean(pred == labels[test_idx, robot])))
            acc = float(np.mean(oof == labels[:, robot]))
            record = {
                "oof": round(acc, 6),
                "folds": [round(x, 6) for x in fold_acc],
                "features": int(X.shape[2]),
                "blocks": block_names,
                "breakdown": breakdown(oof, labels[:, robot], ctx),
            }
            if validation is not None:
                vf, vy, vc, vs = validation
                vbase, vlegal = raw_actions(vf, robot)
                VX, _ = design(vbase, vc, vs, variant)
                model, scaler = fit_model(X, legal, labels[:, robot], np.arange(len(feats)))
                vp = predict_model(model, scaler, VX, vlegal, np.arange(len(vf)))
                record["validation"] = round(float(np.mean(vp == vy[:, robot])), 6)
                record["validation_breakdown"] = breakdown(vp, vy[:, robot], vc)
            result["robots"][str(robot)][variant] = record
            result_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
            print(f"interaction R{robot} {variant}: OOF={acc:.4f} "
                  f"folds={[round(x, 4) for x in fold_acc]} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
