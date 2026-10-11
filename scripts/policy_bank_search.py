"""Search interpretable cost-bank + tie-break policies (research only)."""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from courier.data import load_split
from courier.policy import HEADING, relative_turn

NBANK = 19
ORDERS = list(itertools.permutations(range(4)))  # relative: straight/right/u-turn/left


def bank_predict(scene_feat, robot, bank, order, heading):
    tied = []
    for direction, row in scene_feat[robot].items():
        if row[bank * 3 + 1] > .5:
            tied.append(direction)
    if not tied:
        return next(iter(scene_feat[robot]))
    rank = {rel: i for i, rel in enumerate(order)}
    return min(tied, key=lambda d: rank[relative_turn(heading, d)])


def keys(scenes, mode):
    if mode == "none":
        return np.zeros(len(scenes), dtype=np.int8)
    if mode == "rain":
        return np.asarray([s["weather"] == "rain" for s in scenes], dtype=np.int8)
    if mode in ("urgent", "fragile"):
        return np.asarray([bool(s["mission"][mode]) for s in scenes], dtype=np.int8)
    if mode == "all":
        return np.asarray([(s["weather"] == "rain") * 4 + bool(s["mission"]["urgent"]) * 2
                           + bool(s["mission"]["fragile"]) for s in scenes], dtype=np.int8)
    raise ValueError(mode)


def main():
    with (ROOT / "work" / "policy_features_v1.pkl").open("rb") as f:
        data = pickle.load(f)
    ft, yt = data["train"]
    fv, yv = data["validation"]
    _, _, st = load_split("train")
    _, _, sv = load_split("validation")
    ht = np.asarray([HEADING[s["robot"]["heading"]] for s in st])
    hv = np.asarray([HEADING[s["robot"]["heading"]] for s in sv])

    hinted = {1: "none", 2: "none", 3: "rain", 4: "none", 5: "none",
              6: "urgent", 7: "fragile", 8: "none"}
    for robot in range(1, 9):
        modes = list(dict.fromkeys(("none", hinted[robot], "all")))
        for mode in modes:
            kt, kv = keys(st, mode), keys(sv, mode)
            pred = np.zeros(len(fv), dtype=np.int8)
            choices = []
            for key in np.unique(kt):
                tri = np.flatnonzero(kt == key)
                vai = np.flatnonzero(kv == key)
                best = (-1., None, None)
                for bank in range(NBANK):
                    for order in ORDERS:
                        p = np.asarray([bank_predict(ft[i], robot, bank, order, ht[i]) for i in tri])
                        score = np.mean(p == yt[tri, robot])
                        if score > best[0]:
                            best = (score, bank, order)
                for i in vai:
                    pred[i] = bank_predict(fv[i], robot, best[1], best[2], hv[i])
                choices.append((int(key), round(best[0], 4), best[1], best[2], len(tri)))
            score = np.mean(pred == yv[:, robot])
            print(f"R{robot} mode={mode} val={score:.4f} choices={choices}")


if __name__ == "__main__":
    main()
