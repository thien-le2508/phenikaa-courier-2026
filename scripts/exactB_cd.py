"""Coordinate descent over weights (E fixed 100): cr nc lm tr tl tu fr fl fu. objective: inargmin + lam*unique."""
import sys
import numpy as np
from exactB_common import *
from exactB_wsearch import prep, cond_mask

robot, cond, split = int(sys.argv[1]), sys.argv[2], sys.argv[3]
lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
init = eval(sys.argv[5]) if len(sys.argv) > 5 else {}
D0 = load(split)
P = prep(split, cond_mask(D0, cond))
D, idx = P["D"], P["idx"]
segrel = np.array([relative_turn(D[idx[j]]["m"].heading, d) for j, d in zip(P["segscene"], P["segmove"])])
lab = np.array([D[idx[j]]["y"][robot] for j in P["segscene"]])
islab = P["segmove"] == lab
keys = ["cr", "nc", "lm", "tr", "tl", "tu", "fr", "fl", "fu"]


def score(cb):
    cb = np.atleast_2d(np.array(cb, float))
    W = np.concatenate([np.full((len(cb), 1), 100.0), cb[:, :6]], 1).T
    M = np.minimum.reduceat(P["R"] @ W, P["starts"], axis=0)
    fm = np.stack([np.zeros(len(cb)), cb[:, 6], cb[:, 8], cb[:, 7]])
    M = M + fm[segrel]
    best = np.minimum.reduceat(M, P["scene_start"], axis=0)
    eq = M <= best[P["segscene"]] + 1e-9
    nt = np.add.reduceat(eq.astype(np.int32), P["scene_start"], axis=0)
    ok = np.zeros(best.shape, bool)
    ok[P["segscene"][islab]] = eq[islab]
    return ok.mean(0), (ok & (nt == 1)).mean(0)


cur = np.array([init.get(k, 0) for k in keys], float)
vals = sorted(set(list(range(0, 410, 10)) + [1, 2, 3, 5, 15, 25, 450, 500, 600, 700, 800, 1000, 5000]))
a, u = score(cur)
best_obj = a[0] + lam * u[0]
print("start", dict(zip(keys, cur)), a[0], u[0])
for it in range(6):
    improved = False
    for k in range(len(keys)):
        cands = np.repeat(cur[None], len(vals), 0)
        cands[:, k] = vals
        a, u = score(cands)
        obj = a + lam * u
        j = int(np.argmax(obj))
        if obj[j] > best_obj + 1e-12:
            best_obj = obj[j]; cur = cands[j]; improved = True
    a, u = score(cur)
    print("iter", it, dict(zip(keys, cur.astype(int))), f"in={a[0]:.4f} uniq={u[0]:.4f}", flush=True)
    if not improved:
        break
