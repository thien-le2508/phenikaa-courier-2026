"""Weight search with separate first-move turn costs. grid given as python dict literal in argv[4]."""
import sys, itertools
import numpy as np
from exactB_common import *
from exactB_wsearch import prep, cond_mask, FN

robot, cond, split, grid = int(sys.argv[1]), sys.argv[2], sys.argv[3], eval(sys.argv[4])
D0 = load(split)
P = prep(split, cond_mask(D0, cond))
D, idx = P["D"], P["idx"]
segrel = np.array([relative_turn(D[idx[j]]["m"].heading, d) for j, d in zip(P["segscene"], P["segmove"])])
lab = np.array([D[idx[j]]["y"][robot] for j in P["segscene"]])
islab = P["segmove"] == lab
keys = ["cr", "nc", "lm", "tr", "tl", "tu", "fr", "fl", "fu"]
combos = list(itertools.product(*[grid.get(k, [0]) for k in keys]))
print("scenes", len(idx), "combos", len(combos))
res = []
for k0 in range(0, len(combos), 400):
    cb = np.array(combos[k0:k0 + 400], dtype=np.float64)
    W = np.concatenate([np.full((len(cb), 1), 100.0), cb[:, :6]], 1).T
    C = P["R"] @ W
    M = np.minimum.reduceat(C, P["starts"], axis=0)
    fm = np.stack([np.zeros(len(cb)), cb[:, 6], cb[:, 8], cb[:, 7]])  # rel 0 S,1 R,2 U,3 L
    M = M + fm[segrel]
    best = np.minimum.reduceat(M, P["scene_start"], axis=0)
    eq = M <= best[P["segscene"]] + 1e-9
    nt = np.add.reduceat(eq.astype(np.int32), P["scene_start"], axis=0)
    ok = np.zeros(best.shape, bool)
    ok[P["segscene"][islab]] = eq[islab]
    for a, u, c in zip(ok.mean(0), (ok & (nt == 1)).mean(0), combos[k0:k0 + 400]):
        res.append((a, u, c))
res.sort(key=lambda x: (-x[0], -x[1]))
for a, u, c in res[:int(sys.argv[5]) if len(sys.argv) > 5 else 15]:
    print(dict(zip(keys, c)), f"inargmin={a:.4f} unique_ok={u:.4f}")
