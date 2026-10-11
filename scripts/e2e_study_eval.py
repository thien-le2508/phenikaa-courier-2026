"""Honest public-validation decomposition of policy vs CV/NLP input errors.

This research script intentionally reads only ``train`` and ``validation``.
It builds a train-only copy of the v1/v2 policy bundle, reads validation maps
with the two-crop ensemble, and evaluates every combination of oracle/predicted
graph, mission and drawing style.  It never changes the submission pipeline.

    Z:/anaconda3/python.exe scripts/e2e_study_eval.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from courier.cv.reader import MapReader  # noqa: E402
from courier.data import DATA, load_split  # noqa: E402
from courier.nlp.labels import nlp_target  # noqa: E402
from courier.nlp.metrics import FIELDS, field_eq, score as nlp_score  # noqa: E402
from courier.policy import graph_from_scene, mission_from_scene, predict_all_ml  # noqa: E402
from courier.style import style_features  # noqa: E402
from scripts.cv_eval import compare as compare_graph  # noqa: E402
from scripts.cv_eval import gold_graph  # noqa: E402

WORK = ROOT / "work"
GRAPH_CACHE = WORK / "e2e_study_cv_graphs.pkl"
POLICY_CACHE = WORK / "e2e_study_policy_trainonly.pkl"
OUT = WORK / "e2e_study_results.json"


def candidate_xy(feats, labels, robot, keep=None):
    x, y = [], []
    for sf, yy in zip(feats, labels):
        for direction, row in sf[robot].items():
            a = np.asarray(row, np.float32)
            x.append(a if keep is None else a[keep])
            y.append(direction == yy[robot])
    return np.asarray(x, np.float32), np.asarray(y, np.int8)


def fit_a(x, y):
    return HistGradientBoostingClassifier(
        max_iter=800, learning_rate=.08, max_leaf_nodes=31,
        l2_regularization=1., random_state=0).fit(x, y)


def fit_b(x, y):
    return HistGradientBoostingClassifier(
        max_iter=600, learning_rate=.10, max_leaf_nodes=63,
        min_samples_leaf=5, l2_regularization=.5,
        random_state=0).fit(x, y)


def train_only_bundle():
    """Recreate the selected policy without fitting on validation."""
    if POLICY_CACHE.exists():
        with POLICY_CACHE.open("rb") as f:
            return pickle.load(f)
    with (WORK / "policy_features_v1.pkl").open("rb") as f:
        v1 = pickle.load(f)
    with (WORK / "policy_features_v2.pkl").open("rb") as f:
        v2 = pickle.load(f)
    feats, labels = v1["train"]
    ma, mb = {}, {}
    for robot in range(1, 10):
        x, y = candidate_xy(feats, labels, robot)
        ma[robot], mb[robot] = fit_a(x, y), fit_b(x, y)
        print(f"trained honest policy R{robot}", flush=True)

    feats2, labels2 = v2["train"]
    keep7 = list(range(319))
    x7, y7 = candidate_xy(feats2, labels2, 7, keep7)
    keep9 = list(range(106)) + list(range(290, 319))
    x9, y9 = candidate_xy(feats2, labels2, 9, keep9)
    vm = {
        7: {"mode": "b", "keep": keep7, "b": fit_b(x7, y7)},
        9: {"mode": "ab", "keep": keep9,
            "a": fit_a(x9, y9), "b": fit_b(x9, y9)},
    }

    # Train-only style classifier: the production bundle was refit on train+val.
    _, _, train_scenes = load_split("train")
    xs = np.asarray([style_features(DATA / "train" / s["image"])
                     for s in train_scenes], np.float32)
    ys = np.asarray([s["style"] for s in train_scenes])
    sm = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1,
                              max_features=.75, n_jobs=4,
                              random_state=0).fit(xs, ys)
    bundle = {
        "version": "e2e-study-train-only",
        "models_a": ma, "models_b": mb,
        "choice": {1: "a", 2: "ab", 3: "b", 4: "a", 5: "a",
                   6: "a", 7: "a", 8: "ab", 9: "b"},
        "style_model": sm, "v2_models": vm,
        "styles": ["classic", "night", "print", "sketch"],
    }
    with POLICY_CACHE.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return bundle


def read_cv_graphs(scenes):
    if GRAPH_CACHE.exists():
        with GRAPH_CACHE.open("rb") as f:
            return pickle.load(f)
    reader = MapReader(
        WORK / "heat_v3.pt",
        [WORK / "crops_v3.pt", WORK / "crops_v3b.pt"],
        device="cpu")
    graphs, auxs = [], []
    t0 = time.time()
    for i, s in enumerate(scenes):
        g, aux = reader.read(str(DATA / "validation" / s["image"]))
        graphs.append(g)
        # Only fields needed for graph comparison; avoids caching large tensors.
        auxs.append({k: aux.get(k) for k in
                     ("look_to_status", "legend_kinds")})
        if (i + 1) % 25 == 0:
            print(f"read CV {i + 1}/{len(scenes)} ({time.time() - t0:.0f}s)",
                  flush=True)
    with GRAPH_CACHE.open("wb") as f:
        pickle.dump((graphs, auxs), f, protocol=pickle.HIGHEST_PROTOCOL)
    return graphs, auxs


def acc(pred, y):
    pred = np.asarray(pred)
    return {
        "macro": float(np.mean([np.mean(pred[:, r] == y[:, r])
                                for r in range(10)])),
        "per_robot": {str(r): float(np.mean(pred[:, r] == y[:, r]))
                      for r in range(10)},
    }


def condition_breakdown(pred, y, scenes):
    pred = np.asarray(pred)
    attrs = {
        "rain": [s["weather"] == "rain" for s in scenes],
        "urgent": [s["mission"]["urgent"] for s in scenes],
        "fragile": [s["mission"]["fragile"] for s in scenes],
        "via": [s["mission"]["via"] is not None for s in scenes],
    }
    out = {}
    for name, values in attrs.items():
        out[name] = {}
        for val in (False, True):
            ix = np.flatnonzero(np.asarray(values) == val)
            out[name][str(val).lower()] = {
                "n_scenes": int(len(ix)),
                "macro": float(np.mean(pred[ix] == y[ix])),
            }
    for style in sorted({s["style"] for s in scenes}):
        ix = [i for i, s in enumerate(scenes) if s["style"] == style]
        out.setdefault("style", {})[style] = {
            "n_scenes": len(ix), "macro": float(np.mean(pred[ix] == y[ix]))}
    return out


def main():
    rows, labels_flat, scenes = load_split("validation")
    assert scenes is not None and labels_flat is not None and len(scenes) == 300
    labels = np.asarray(labels_flat, np.int64).reshape(len(scenes), 10)
    bundle = train_only_bundle()
    cv_graphs, auxs = read_cv_graphs(scenes)

    gt_graphs = [graph_from_scene(s) for s in scenes]
    gt_missions = [mission_from_scene(s, drop_map_goal_type=True) for s in scenes]
    pred_missions = json.loads(
        (WORK / "nlp_v3_holdout.valpred.json").read_text(encoding="utf-8"))

    # Honest style prediction from train-only classifier.
    sx = np.asarray([style_features(DATA / "validation" / s["image"])
                     for s in scenes], np.float32)
    pred_styles = bundle["style_model"].predict(sx).tolist()
    gt_styles = [s["style"] for s in scenes]

    graph_checks, graph_errs = [], []
    for g, a, s in zip(cv_graphs, auxs, scenes):
        chk, errs = compare_graph(g, gold_graph(s), s, a)
        graph_checks.append(chk)
        graph_errs.append(errs)

    gold_nlp = [nlp_target(s["mission"]) for s in scenes]
    nlp_acc, nlp_exact = nlp_score(pred_missions, gold_nlp, scenes)
    field_errors = {f: int(sum(not field_eq(f, p, g)
                               for p, g in zip(pred_missions, gold_nlp)))
                    for f in FIELDS}

    variants = {
        "oracle": (gt_graphs, gt_missions, gt_styles),
        "cv_only": (cv_graphs, gt_missions, gt_styles),
        "nlp_only": (gt_graphs, pred_missions, gt_styles),
        "style_only": (gt_graphs, gt_missions, pred_styles),
        "cv_nlp": (cv_graphs, pred_missions, gt_styles),
        "cv_style": (cv_graphs, gt_missions, pred_styles),
        "nlp_style": (gt_graphs, pred_missions, pred_styles),
        "end_to_end": (cv_graphs, pred_missions, pred_styles),
    }
    predictions = {}
    for name, (graphs, missions, styles) in variants.items():
        pp = []
        for graph, mission, style in zip(graphs, missions, styles):
            g = dict(graph, style=style)
            pp.append(predict_all_ml(g, mission, bundle))
        predictions[name] = np.asarray(pp, np.int64)
        print(name, acc(predictions[name], labels), flush=True)

    oracle = predictions["oracle"]
    impacts = {}
    for name, pred in predictions.items():
        changed = pred != oracle
        impacts[name] = {
            "changed_decisions": int(changed.sum()),
            "changed_scenes": int(changed.any(1).sum()),
            "hurt": int(((oracle == labels) & (pred != labels)).sum()),
            "helped": int(((oracle != labels) & (pred == labels)).sum()),
            "accuracy": acc(pred, labels),
        }

    # Which observed CV component errors actually alter policy output?
    cv_changed = (predictions["cv_only"] != oracle).any(1)
    err_type_count, err_type_changed = Counter(), Counter()
    for changed, errs in zip(cv_changed, graph_errs):
        for typ in {e.split()[0] for e in errs}:
            err_type_count[typ] += 1
            err_type_changed[typ] += int(changed)

    result = {
        "scope": "train + public validation only; no test_private access",
        "n_scenes": len(scenes),
        "policy": "v1/v2 selected bundle refit on train only",
        "cv": {
            "model": "heat_v3 + mean(crops_v3,crops_v3b), CPU",
            "perfect_graph_scenes": int(sum(c["graph"] for c in graph_checks)),
            "component_correct": {k: int(sum(c.get(k, False) for c in graph_checks))
                                  for k in sorted(graph_checks[0])},
            "error_types": dict(Counter(e.split()[0] for es in graph_errs for e in es)),
            "error_type_policy_changed": dict(err_type_changed),
            "bad_scenes": [{"i": i, "style": scenes[i]["style"], "errors": es}
                           for i, es in enumerate(graph_errs) if es],
        },
        "nlp": {
            "checkpoint": "nlp_v3_holdout (never trained on validation)",
            "metrics": {k: float(v) for k, v in nlp_acc.items()},
            "field_error_counts": field_errors,
            "exact_errors": int(sum(not x for x in nlp_exact)),
        },
        "style": {
            "train_only_accuracy": float(np.mean(np.asarray(pred_styles) == gt_styles)),
            "errors": [{"i": i, "gold": g, "pred": p}
                       for i, (g, p) in enumerate(zip(gt_styles, pred_styles)) if g != p],
        },
        "variants": impacts,
        "end_to_end_breakdown": condition_breakdown(
            predictions["end_to_end"], labels, scenes),
        "oracle_breakdown": condition_breakdown(oracle, labels, scenes),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
