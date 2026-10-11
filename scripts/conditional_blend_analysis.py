"""Nested OOF audit of the conditional-model blends; public train/validation only."""
import json
import pickle
from pathlib import Path

import numpy as np

from conditional_models import OUT, ROOT, FOLDS, calibrate_saved, weight_grid


def choose(sources, y, idx):
    names = list(sources)
    rows = []
    for ints in weight_grid(len(names)):
        w = np.asarray(ints, float) / sum(ints)
        p = sum(w[j] * sources[n][0] for j, n in enumerate(names))
        acc = float(np.mean(p[idx].argmax(1) == y[idx]))
        rows.append(((acc, -np.count_nonzero(w)), w))
    return max(rows, key=lambda z: z[0])[1]


def metrics(p, y):
    chosen = np.maximum(p[np.arange(len(y)), y], 1e-12)
    onehot = np.eye(4)[y]
    conf = p.max(1)
    hit = p.argmax(1) == y
    ece = 0.0
    for lo in np.linspace(0, .9, 10):
        m = (conf >= lo) & (conf < lo + .1)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return {"accuracy": float(hit.mean()), "nll": float(-np.log(chosen).mean()),
            "brier": float(np.square(p - onehot).sum(1).mean()), "ece10": float(ece)}


def breakdown(pred, y, states):
    hit = pred == y
    out = {}
    for style in ("classic", "night", "print", "sketch"):
        m = np.asarray(states["style"]) == style
        out["style=" + style] = {"n": int(m.sum()), "acc": float(hit[m].mean())}
    names = ("rain", "urgent", "fragile")
    for j, name in enumerate(names):
        for value in (0, 1):
            m = states["conds"][:, j] == value
            out[f"{name}={value}"] = {"n": int(m.sum()), "acc": float(hit[m].mean())}
    for value in (0, 1):
        m = states["has_via"].astype(int) == value
        out[f"via={value}"] = {"n": int(m.sum()), "acc": float(hit[m].mean())}
    return out


def main():
    probs = pickle.loads((OUT / "conditional_probs.pkl").read_bytes())
    base = {n: pickle.loads((OUT / f"{n}_probs.pkl").read_bytes())
            for n in ("baseline_v1", "baseline_v2", "nip")}
    labels = pickle.loads((ROOT / "work" / "policy_features_v1.pkl").read_bytes())
    yt, yv = np.asarray(labels["train"][1]), np.asarray(labels["validation"][1])
    st = pickle.loads((OUT / "states_train.pkl").read_bytes())
    sv = pickle.loads((OUT / "states_validation.pkl").read_bytes())
    results = {}
    for r in sorted(probs):
        sources = {"conditional": (probs[r]["structured_oof"], probs[r]["structured_val"])}
        for name in base:
            po, pv, _, _ = calibrate_saved(base[name][r]["oof"], base[name][r]["val"], yt[:, r])
            sources[name] = (po, pv)
        nested = np.zeros_like(sources["conditional"][0])
        fold_weights = []
        all_idx = np.arange(len(yt))
        for _, hold in FOLDS:
            fit = np.setdiff1d(all_idx, hold, assume_unique=True)
            w = choose(sources, yt[:, r], fit)
            nested[hold] = sum(w[j] * sources[n][0][hold] for j, n in enumerate(sources))
            fold_weights.append({n: float(w[j]) for j, n in enumerate(sources)})
        fullw = choose(sources, yt[:, r], all_idx)
        val = sum(fullw[j] * sources[n][1] for j, n in enumerate(sources))
        conditional = sources["conditional"]
        results[str(r)] = {
            "nested_blend_oof": metrics(nested, yt[:, r]),
            "validation_after_full_oof_selection": metrics(val, yv[:, r]),
            "full_oof_weights": {n: float(fullw[j]) for j, n in enumerate(sources)},
            "fold_weights": fold_weights,
            "conditional_train_metrics": metrics(conditional[0], yt[:, r]),
            "conditional_validation_metrics": metrics(conditional[1], yv[:, r]),
            "conditional_train_breakdown": breakdown(conditional[0].argmax(1), yt[:, r], st),
            "conditional_validation_breakdown": breakdown(conditional[1].argmax(1), yv[:, r], sv),
        }
        print(f"R{r}: nested={results[str(r)]['nested_blend_oof']['accuracy']:.4f} "
              f"val={results[str(r)]['validation_after_full_oof_selection']['accuracy']:.4f}")
    (OUT / "conditional_nested.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
