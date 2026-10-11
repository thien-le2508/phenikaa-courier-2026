"""Structured conditional-utility study with honest per-action OOF scores.

Only public train is used for model/configuration/ensemble selection.  Public
validation is evaluated once after selection.  This script never discovers or
reads test_private.

Outputs (all deliberately separate from the final pipeline):
  work/ml2/conditional_results.json
  work/ml2/conditional_probs.pkl

Usage: python scripts/conditional_models.py [comma-separated robots]
"""
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ml2"
SCENE_SUFFIX = 23
FLAGS = ("rain", "urgent", "fragile", "via")
STYLES = ("classic", "night", "print", "sketch")
FOLDS = list(KFold(5, shuffle=True, random_state=2026).split(np.arange(2000)))


def load_data():
    data = pickle.loads((ROOT / "work" / "policy_features_v1.pkl").read_bytes())
    st = pickle.loads((OUT / "states_train.pkl").read_bytes())
    sv = pickle.loads((OUT / "states_validation.pkl").read_bytes())
    ft, yt = data["train"]
    fv, yv = data["validation"]
    def context(s):
        c = np.column_stack([s["conds"].astype(float), s["has_via"].astype(float)])
        z = np.asarray([[x == name for name in STYLES] for x in s["style"]], float)
        return c, z
    ct, zt = context(st)
    cv, zv = context(sv)
    return ft, np.asarray(yt), ct, zt, fv, np.asarray(yv), cv, zv


def action_tensor(feats, robot):
    width = len(next(iter(feats[0][robot].values()))) - SCENE_SUFFIX
    x = np.zeros((len(feats), 4, width), dtype=np.float64)
    legal = np.zeros((len(feats), 4), dtype=bool)
    for i, scene in enumerate(feats):
        for d, row in scene[robot].items():
            x[i, d] = row[:-SCENE_SUFFIX]
            legal[i, d] = True
    return x, legal


def design(x, ctx, style, variant):
    blocks = [("shared", x)]
    if variant in ("main", "main_style"):
        blocks += [(name, x * ctx[:, None, j, None]) for j, name in enumerate(FLAGS)]
    if variant in ("style", "main_style"):
        blocks += [("style=" + name, x * style[:, None, j, None])
                   for j, name in enumerate(STYLES[1:], 1)]
    return np.concatenate([b for _, b in blocks], 2), [n for n, _ in blocks]


def pair_data(x, legal, y, idx):
    rows = []
    for i in idx:
        chosen = int(y[i])
        for d in np.flatnonzero(legal[i]):
            if d != chosen:
                rows.append(x[i, chosen] - x[i, d])
    pos = np.asarray(rows)
    return np.r_[pos, -pos], np.r_[np.ones(len(pos)), np.zeros(len(pos))]


def fit_ranker(x, legal, y, idx, c_value, shrink, base_width):
    xx, yy = pair_data(x, legal, y, idx)
    scaler = StandardScaler(with_mean=False).fit(xx)
    xx = scaler.transform(xx)
    # A smaller multiplier means the interaction needs a larger coefficient,
    # hence receives a stronger effective L2 penalty than the shared block.
    mult = np.ones(x.shape[2])
    mult[base_width:] = shrink
    model = LogisticRegression(C=c_value, solver="lbfgs", penalty="l2",
                               fit_intercept=False, max_iter=500, tol=2e-5)
    model.fit(xx * mult, yy)
    return model, scaler, mult


def utilities(model, scaler, mult, x, legal, idx):
    out = np.full((len(idx), 4), -np.inf)
    for j, i in enumerate(idx):
        ds = np.flatnonzero(legal[i])
        if len(ds):
            out[j, ds] = model.decision_function(scaler.transform(x[i, ds]) * mult)
    return out


def softmax(scores, legal, temp=1.0):
    q = np.where(legal, scores / temp, -np.inf)
    m = np.max(q, axis=1, keepdims=True)
    e = np.where(legal, np.exp(q - m), 0.0)
    return e / e.sum(1, keepdims=True)


def best_temp(scores, legal, y, idx):
    def loss(logt):
        p = softmax(scores[idx], legal[idx], np.exp(logt))
        return -np.log(np.maximum(p[np.arange(len(idx)), y[idx]], 1e-12)).mean()
    return float(np.exp(minimize_scalar(loss, bounds=(-3.5, 3.5), method="bounded").x))


def cross_calibrate(scores, legal, y):
    p = np.zeros_like(scores)
    ts = []
    all_idx = np.arange(len(y))
    for _, hold in FOLDS:
        fit = np.setdiff1d(all_idx, hold, assume_unique=True)
        t = best_temp(scores, legal, y, fit)
        p[hold] = softmax(scores[hold], legal[hold], t)
        ts.append(t)
    return p, ts, best_temp(scores, legal, y, all_idx)


def normalize_saved(p):
    legal = p >= 0
    q = np.where(legal, np.maximum(p, 1e-7), 0.0)
    return q / np.maximum(q.sum(1, keepdims=True), 1e-12), legal


def calibrate_saved(oof, val, y):
    po, legal = normalize_saved(oof)
    pv, legalv = normalize_saved(val)
    scores = np.where(legal, np.log(np.maximum(po, 1e-7)), -np.inf)
    pc, ts, tfull = cross_calibrate(scores, legal, y)
    sv = np.where(legalv, np.log(np.maximum(pv, 1e-7)), -np.inf)
    return pc, softmax(sv, legalv, tfull), ts, tfull


def configs(robot):
    variants = ["main_style", "style"] if robot in (1, 5, 9) else ["main", "main_style"]
    # Broad enough to measure shrinkage while keeping this parallel study light.
    return [(v, c, s) for v in variants for c in (.01, .03, .1)
            for s in (.2, .5, 1.0)]


def fold_acc(pred, y):
    return [float(np.mean(pred[b] == y[b])) for _, b in FOLDS]


def weight_grid(k, total=10):
    if k == 1:
        yield (total,)
        return
    for cuts in itertools.combinations(range(total + k - 1), k - 1):
        pts = (-1,) + cuts + (total + k - 1,)
        yield tuple(pts[i + 1] - pts[i] - 1 for i in range(k))


def main():
    robots = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,5,6,7,8,9").split(",")]
    ft, yt, ct, zt, fv, yv, cv, zv = load_data()
    if len(ft) != 2000:
        raise ValueError("fold definition expects the 2000-scene public train split")
    saved = {}
    for name in ("baseline_v1", "baseline_v2", "nip"):
        saved[name] = pickle.loads((OUT / f"{name}_probs.pkl").read_bytes())
    results, outputs = {}, {}
    for robot in robots:
        started = time.time()
        x, legal = action_tensor(ft, robot)
        xv, legalv = action_tensor(fv, robot)
        base_width = x.shape[2]
        records = []
        candidate_scores = {}
        candidate_meta = {}
        for variant, c_value, shrink in configs(robot):
            xd, blocks = design(x, ct, zt, variant)
            raw = np.full((len(ft), 4), -np.inf)
            for tr, hold in FOLDS:
                model, scaler, mult = fit_ranker(xd, legal, yt[:, robot], tr,
                                                  c_value, shrink, base_width)
                raw[hold] = utilities(model, scaler, mult, xd, legal, hold)
            pred = raw.argmax(1)
            rec = dict(variant=variant, C=c_value, shrink=shrink,
                       oof=float(np.mean(pred == yt[:, robot])),
                       folds=fold_acc(pred, yt[:, robot]), blocks=blocks)
            records.append(rec)
            key = (rec["oof"], -np.std(rec["folds"]), -len(blocks), -c_value, -shrink)
            candidate_scores[key] = raw
            candidate_meta[key] = rec
        best_key = max(candidate_scores)
        best = candidate_meta[best_key]
        raw = candidate_scores[best_key]
        cp, temps, temp_full = cross_calibrate(raw, legal, yt[:, robot])

        variant, c_value, shrink = best["variant"], best["C"], best["shrink"]
        xd, _ = design(x, ct, zt, variant)
        xdv, _ = design(xv, cv, zv, variant)
        model, scaler, mult = fit_ranker(xd, legal, yt[:, robot], np.arange(len(ft)),
                                          c_value, shrink, base_width)
        rawv = utilities(model, scaler, mult, xdv, legalv, np.arange(len(fv)))
        cpv = softmax(rawv, legalv, temp_full)

        sources = {"conditional": (cp, cpv)}
        source_cal = {"conditional": {"fold_temps": temps, "full_temp": temp_full}}
        for name in ("baseline_v1", "baseline_v2", "nip"):
            po, pv, ts, tf = calibrate_saved(saved[name][robot]["oof"],
                                               saved[name][robot]["val"], yt[:, robot])
            sources[name] = (po, pv)
            source_cal[name] = {"fold_temps": ts, "full_temp": tf}

        names = list(sources)
        ensemble_rows = []
        for ints in weight_grid(len(names)):
            if not any(ints):
                continue
            w = np.asarray(ints, float) / sum(ints)
            po = sum(w[j] * sources[n][0] for j, n in enumerate(names))
            pred = po.argmax(1)
            acc = float(np.mean(pred == yt[:, robot]))
            folds = fold_acc(pred, yt[:, robot])
            # Accuracy is primary; fold stability and sparsity only break ties.
            key = (acc, -np.std(folds), -np.count_nonzero(w))
            ensemble_rows.append((key, w, folds))
        _, bestw, efolds = max(ensemble_rows, key=lambda row: row[0])
        epo = sum(bestw[j] * sources[n][0] for j, n in enumerate(names))
        epv = sum(bestw[j] * sources[n][1] for j, n in enumerate(names))
        result = {
            "selection": "train OOF accuracy; validation confirmation only",
            "candidates": sorted(records, key=lambda q: -q["oof"]),
            "selected_structured": best,
            "structured_calibration": source_cal["conditional"],
            "structured_oof": float(np.mean(cp.argmax(1) == yt[:, robot])),
            "structured_validation": float(np.mean(cpv.argmax(1) == yv[:, robot])),
            "ensemble_weights": {n: float(bestw[j]) for j, n in enumerate(names)},
            "ensemble_oof": float(np.mean(epo.argmax(1) == yt[:, robot])),
            "ensemble_folds": efolds,
            "ensemble_validation": float(np.mean(epv.argmax(1) == yv[:, robot])),
            "source_calibration": source_cal,
            "source_oof": {n: float(np.mean(p[0].argmax(1) == yt[:, robot])) for n, p in sources.items()},
            "source_validation": {n: float(np.mean(p[1].argmax(1) == yv[:, robot])) for n, p in sources.items()},
            "seconds": time.time() - started,
        }
        results[str(robot)] = result
        outputs[robot] = {"structured_oof": cp, "structured_val": cpv,
                          "ensemble_oof": epo, "ensemble_val": epv}
        (OUT / "conditional_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        (OUT / "conditional_probs.pkl").write_bytes(pickle.dumps(outputs))
        print(f"R{robot}: structured {result['structured_oof']:.4f}/{result['structured_validation']:.4f}; "
              f"ensemble {result['ensemble_oof']:.4f}/{result['ensemble_validation']:.4f}; "
              f"weights={result['ensemble_weights']} ({result['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
