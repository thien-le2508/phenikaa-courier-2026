"""Robust train-OOF comparison of saved policy candidates.

Reads only train/validation labels, train context metadata, and already saved
OOF/validation probabilities.  Model ranking is based on train OOF.  The public
validation score is reported but never used to rank candidates.  Private test
data is never read.

    python scripts/ml2_robust_selection.py baseline_v1 baseline_v2 nip
"""
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ml2"


def normalise(p):
    p = np.where(np.asarray(p) < 0, 0.0, np.asarray(p, dtype=float))
    total = p.sum(1, keepdims=True)
    return p / np.where(total > 0, total, 1.0)


def paired_ci(hit, base_hit, seed, draws=10000):
    delta = hit.astype(float) - base_hit.astype(float)
    rng = np.random.default_rng(seed)
    # Chunk bootstrap indices so peak memory remains small.
    means = []
    for _ in range(0, draws, 500):
        k = min(500, draws - len(means) * 500)
        idx = rng.integers(0, len(delta), size=(k, len(delta)))
        means.append(delta[idx].mean(1))
    boot = np.concatenate(means)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return [round(float(lo), 6), round(float(hi), 6)]


def main():
    names = sys.argv[1:] or ["baseline_v1", "baseline_v2", "nip"]
    bundles = {name: pickle.loads((OUT / f"{name}_probs.pkl").read_bytes()) for name in names}
    yt = np.asarray(json.loads((ROOT / "delivery_public/train/labels.json").read_text())).reshape(-1, 10)
    yv = np.asarray(json.loads((ROOT / "delivery_public/validation/labels.json").read_text())).reshape(-1, 10)
    states = pickle.loads((OUT / "states_train.pkl").read_bytes())
    c = np.column_stack([states["conds"], states["has_via"]]).astype(np.int64)
    group = (c * np.array([8, 4, 2, 1])).sum(1)
    folds = list(KFold(5, shuffle=True, random_state=2026).split(np.arange(len(yt))))
    report = {
        "protocol": "rank by 5-fold scene OOF on train; validation report-only; no private test",
        "sources": names,
        "robots": {},
    }
    for robot in range(1, 10):
        avail = [name for name in names if robot in bundles[name]]
        if not avail:
            continue
        candidates = {}
        for size in range(1, len(avail) + 1):
            for combo in itertools.combinations(avail, size):
                po = sum(normalise(bundles[name][robot]["oof"]) for name in combo)
                pv = sum(normalise(bundles[name][robot]["val"]) for name in combo)
                candidates["+".join(combo)] = (po.argmax(1), pv.argmax(1))
        base_name = "baseline_v1" if "baseline_v1" in candidates else next(iter(candidates))
        base_hit = candidates[base_name][0] == yt[:, robot]
        rows = []
        for rank_name, (op, vp) in candidates.items():
            hit = op == yt[:, robot]
            fold_acc = [float(hit[idx].mean()) for _, idx in folds]
            group_acc = {format(g, "04b"): float(hit[group == g].mean())
                         for g in range(16) if np.any(group == g)}
            rows.append({
                "name": rank_name,
                "oof": round(float(hit.mean()), 6),
                "validation": round(float(np.mean(vp == yv[:, robot])), 6),
                "folds": [round(x, 6) for x in fold_acc],
                "fold_std": round(float(np.std(fold_acc)), 6),
                "fold_wins_vs_base": int(sum(a > b for a, b in zip(
                    fold_acc, [float(base_hit[idx].mean()) for _, idx in folds]))),
                "delta_vs_base": round(float((hit.astype(float) - base_hit).mean()), 6),
                "delta95_vs_base": paired_ci(hit, base_hit, 2026 + robot * 100 + len(rows)),
                "group_macro": round(float(np.mean(list(group_acc.values()))), 6),
                "group_min": round(float(np.min(list(group_acc.values()))), 6),
                "groups": {k: round(v, 6) for k, v in group_acc.items()},
            })
        rows.sort(key=lambda row: (-row["oof"], row["fold_std"], -row["group_macro"]))
        report["robots"][str(robot)] = rows
        top = rows[0]
        print(f"R{robot} {top['name']}: OOF={top['oof']:.4f} val={top['validation']:.4f} "
              f"delta95={top['delta95_vs_base']} folds={top['folds']}")
    (OUT / "robust_selection.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
