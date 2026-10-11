"""Random search over linear weights on Pareto fronts. objective = in + uni (both fractions)."""
import sys, os
import numpy as np
from exactB_common import *
from exactB_wsearch import prep, cond_mask
robot, cond, split, nsamp = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
D0 = load(split)
P = prep(split, cond_mask(D0, cond))
D, idx = P["D"], P["idx"]
segrel = np.array([relative_turn(D[idx[j]]["m"].heading, d) for j, d in zip(P["segscene"], P["segmove"])])
lab = np.array([D[idx[j]]["y"][robot] for j in P["segscene"]])
islab = P["segmove"] == lab
keys = ["cr", "nc", "lm", "tr", "tl", "tu", "fr", "fl", "fu"]
hi = np.array([float(x) for x in os.environ.get("HI", "400,150,400,150,150,1000,150,150,1000").split(",")])
rng = np.random.default_rng(seed)
def score(cb):
    W = np.concatenate([np.full((len(cb), 1), 100.0), cb[:, :6]], 1).T
    M = np.minimum.reduceat(P["R"] @ W, P["starts"], axis=0)
    fm = np.stack([np.zeros(len(cb)), cb[:, 6], cb[:, 8], cb[:, 7]])
    M = M + fm[segrel]
    best = np.minimum.reduceat(M, P["scene_start"], axis=0)
    eq = M <= best[P["segscene"]] + 1e-9
    nt = np.add.reduceat(eq.astype(np.int32), P["scene_start"], axis=0)
    ok = np.zeros(best.shape, bool); ok[P["segscene"][islab]] = eq[islab]
    return ok.mean(0), (ok & (nt == 1)).mean(0)
allres = []
for b in range(nsamp // 500):
    cb = rng.uniform(0, 1, (500, 9)) ** 2 * hi
    cb = np.round(cb) * (rng.uniform(0, 1, (500, 9)) > 0.35)
    a, u = score(cb)
    for j in range(500): allres.append((a[j] + u[j], a[j], u[j], cb[j]))
allres.sort(key=lambda x: -x[0])
for o, a, u, c in allres[:10]:
    print(dict(zip(keys, c.astype(int))), f"in={a:.4f} uni={u:.4f}")
