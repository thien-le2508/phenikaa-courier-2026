"""Vectorized weight search over Pareto fronts (features: edges, crowded, noncov, lm, right, left, uturn).
Usage: python exactB_wsearch.py robot cond_expr split   (cond uses U,F,R,N(night),V(via) booleans, S=style)"""
import pickle, sys, itertools, os
import numpy as np
from exactB_common import *

SL = int(os.environ.get("SL", 6))
FN = ["E", "cr", "nc", "lm", "tr", "tl", "tu"]


def prep(split, mask=None):
    D = load(split)
    F = pickle.load(open(ROOT / f"work/exactB_pareto_{split}_s{SL}.pkl", "rb"))
    idx = [i for i in range(len(D)) if (mask is None or mask[i]) and F[i]]
    rows, segmove, segscene, scene_start = [], [], [], []
    for j, i in enumerate(idx):
        scene_start.append(len(segmove))
        for d in sorted(F[i]):
            rows.append(F[i][d])
            segmove.append(d)
            segscene.append(j)
    starts = np.cumsum([0] + [len(x) for x in rows[:-1]])
    R = np.concatenate(rows).astype(np.float64)
    return dict(D=D, F=F, idx=idx, R=R, starts=starts, segmove=np.array(segmove), segscene=np.array(segscene),
                scene_start=np.array(scene_start))


def evaluate(P, W, robot):
    """W: (7, nW). returns ok (nscene x nW) label in argmin, nties."""
    C = P["R"] @ W
    M = np.minimum.reduceat(C, P["starts"], axis=0)
    best = np.minimum.reduceat(M, P["scene_start"], axis=0)
    eq = (M <= best[P["segscene"]] + 1e-9)
    nties = np.add.reduceat(eq.astype(np.int32), P["scene_start"], axis=0)
    D, idx = P["D"], P["idx"]
    lab = np.array([D[idx[j]]["y"][robot] for j in P["segscene"]])
    islab = P["segmove"] == lab
    ok = np.zeros(best.shape, bool)
    ok[P["segscene"][islab]] = eq[islab]
    return ok, nties


def cond_mask(D, cond):
    return [bool(eval(cond, {}, env_of(sc))) for sc in D]


if __name__ == "__main__":
    robot = int(sys.argv[1])
    cond = sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else "train"
    D0 = load(split)
    P = prep(split, cond_mask(D0, cond))
    print("scenes", len(P["idx"]))
    grid = dict(cr=[0, 1, 3, 5, 10, 20, 50, 100, 1000], nc=[0, 1, 3, 5, 10, 30, 1000], lm=[0, 1, 5, 10, 20, 50, 100, 1000],
                tr=[0, 1, 5, 10, 30, 100], tl=[0, 1, 5, 10, 30, 100], tu=[0, 1, 10, 50, 100])
    Ws = np.array([[100, *c] for c in itertools.product(*grid.values())], dtype=np.float64).T
    acc, uni = [], []
    for k in range(0, Ws.shape[1], 500):
        ok, nt = evaluate(P, Ws[:, k:k + 500], robot)
        acc.append(ok.mean(0)); uni.append((ok & (nt == 1)).mean(0))
    acc = np.concatenate(acc); uni = np.concatenate(uni)
    for j in np.argsort(-acc, kind="stable")[:20]:
        print(dict(zip(FN, Ws[:, j].astype(int))), f"inargmin={acc[j]:.4f} unique_ok={uni[j]:.4f}")
