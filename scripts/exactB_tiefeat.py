"""For tied scenes under a given cost, compare label move vs other tied moves on first-step & path features."""
import sys, pickle
import numpy as np
from collections import Counter
from exactB_common import *
robot, cond, split, w = int(sys.argv[1]), sys.argv[2], sys.argv[3], eval(sys.argv[4])
D = load(split)
F = pickle.load(open(ROOT / f"work/exactB_pareto_{split}_s6.pkl", "rb"))
Wv = np.array([w.get(k, 0) for k in ["E", "cr", "nc", "lm", "tr", "tl", "tu"]], float); Wv[0] = w.get("E", 100)
fmc = {0: 0, 1: w.get("fr", 0), 2: w.get("fu", 0), 3: w.get("fl", 0)}
stats = Counter(); n = 0
for i, sc in enumerate(D):
    mi = sc["mi"]; m = sc["m"]
    if not eval(cond, {}, dict(U=mi["urgent"], F=mi["fragile"], R=m.rain, N=sc["s"]["style"] == "night")) or not F[i]:
        continue
    q = {}; best = {}
    for d, fr in F[i].items():
        c = fr @ Wv
        q[d] = c.min() + fmc[relative_turn(m.heading, d)]
        best[d] = fr[c == c.min()]
    am = argmins(q); lab = sc["y"][robot]
    if lab not in am or len(am) < 2:
        continue
    n += 1
    step = {d: (st, v, sti) for d, v, st, sti in m.adj[m.start]}
    def feats(d):
        st, v, _ = step[d]
        b = best[d]
        return dict(crowded1=st == "crowded", covered1=st == "covered", lm1=v in m.lm_cells, deg=len(m.adj[v]),
                    pcr=b[:, 1].min(), pnc=b[:, 2].min(), plm=b[:, 3].min(), pturns=(b[:, 4] + b[:, 5] + b[:, 6]).min(),
                    pR=b[:, 4].min(), pL=b[:, 5].min(), pU=b[:, 6].min(), maxcr=b[:, 1].max())
    fl = feats(lab)
    for d in am - {lab}:
        fo = feats(d)
        for k in fl:
            stats[k, "lab<" if fl[k] < fo[k] else ("lab>" if fl[k] > fo[k] else "=")] += 1
print("tied scenes", n)
for k in ["crowded1", "covered1", "lm1", "deg", "pcr", "pnc", "plm", "pturns", "pR", "pL", "pU", "maxcr"]:
    print(k, stats[k, "lab<"], stats[k, "lab>"], stats[k, "="])
