"""Show failures for a weight config on Pareto fronts. args robot cond split wdict n [offset]"""
import sys, pickle
import numpy as np
from exactB_common import *
robot, cond, split, w, n = int(sys.argv[1]), sys.argv[2], sys.argv[3], eval(sys.argv[4]), int(sys.argv[5])
off = int(sys.argv[6]) if len(sys.argv) > 6 else 0
D = load(split); F = pickle.load(open(ROOT / f"work/exactB_pareto_{split}_s6.pkl", "rb"))
Wv = np.array([100, w.get("cr", 0), w.get("nc", 0), w.get("lm", 0), w.get("tr", 0), w.get("tl", 0), w.get("tu", 0)], float)
fm = {0: 0, 1: w.get("fr", 0), 2: w.get("fu", 0), 3: w.get("fl", 0)}
k = 0; fails = []
for i, sc in enumerate(D):
    mi = sc["mi"]; m = sc["m"]
    if not F[i] or not eval(cond, {}, dict(U=mi["urgent"], F=mi["fragile"], R=m.rain, N=sc["s"]["style"] == "night")): continue
    q = {}; arg = {}
    for d, fr in F[i].items():
        c = fr @ Wv; j = int(np.argmin(c)); q[d] = c[j] + fm[relative_turn(m.heading, d)]; arg[d] = fr[j]
    if sc["y"][robot] in argmins(q) : continue
    fails.append(i)
    if len(fails) <= off or k >= n: continue
    k += 1
    print(render(sc)); print("style", sc["s"]["style"])
    for d in q: print("  ", NAMES[d], "SRUL"[relative_turn(m.heading, d)], q[d], "[E,cr,nc,lm,R,L,U]=", tuple(arg[d]), "<-LABEL" if d == sc["y"][robot] else "")
print("nfail", len(fails), fails[:50])
