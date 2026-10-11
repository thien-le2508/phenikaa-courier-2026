"""Evaluate per-robot ensembles of saved OOF/val probabilities and pick per-robot best by OOF.

    python scripts/ml2_ensemble.py name1 name2 ...   (names: files work/ml2/<name>_probs.pkl)
"""
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ml2"


def norm(p):
    p = np.where(p < 0, 0.0, p)
    s = p.sum(1, keepdims=True)
    return p / np.where(s > 0, s, 1)


def labels():
    tr = json.loads((ROOT / "delivery_public/train/labels.json").read_text())
    va = json.loads((ROOT / "delivery_public/validation/labels.json").read_text())
    return np.asarray(tr).reshape(-1, 10), np.asarray(va).reshape(-1, 10)


def main():
    names = sys.argv[1:]
    probs = {n: pickle.loads((OUT / f"{n}_probs.pkl").read_bytes()) for n in names}
    yt, yv = labels()
    table = {}
    for r in range(1, 10):
        avail = [n for n in names if r in probs[n]]
        rows = []
        for k in range(1, len(avail) + 1):
            for combo in itertools.combinations(avail, k):
                po = sum(norm(probs[n][r]["oof"]) for n in combo)
                pv = sum(norm(probs[n][r]["val"]) for n in combo)
                rows.append(("+".join(combo), float(np.mean(po.argmax(1) == yt[:, r])),
                             float(np.mean(pv.argmax(1) == yv[:, r]))))
        rows.sort(key=lambda t: -t[1])
        table[r] = rows
        print(f"R{r}: " + " | ".join(f"{n} {o:.4f}/{v:.4f}" for n, o, v in rows[:6]))
    (OUT / "ensemble_table.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
