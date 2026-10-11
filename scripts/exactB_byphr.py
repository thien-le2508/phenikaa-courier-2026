import sys, pickle
import numpy as np
from collections import Counter
from exactB_common import *
robot, split, w = int(sys.argv[1]), sys.argv[2], eval(sys.argv[3])
D = load(split)
F = pickle.load(open(ROOT / f"work/exactB_pareto_{split}_s6.pkl", "rb"))
Wv = np.array([w.get(k, 0) for k in ["E", "cr", "nc", "lm", "tr", "tl", "tu"]], float); Wv[0] = w.get("E", 100)
fmc = {0: 0, 1: w.get("fr", 0), 2: w.get("fu", 0), 3: w.get("fl", 0)}
c = Counter()
for i, sc in enumerate(D):
    if not F[i]: continue
    q = {d: (fr @ Wv).min() + fmc[relative_turn(sc["m"].heading, d)] for d, fr in F[i].items()}
    p = urg_phrase(sc["s"]["mission"]["text"]) + ("U" if sc["mi"]["urgent"] else "-")
    am = argmins(q)
    c[p, "n"] += 1; c[p, "in"] += sc["y"][robot] in am; c[p, "uniq"] += am == {sc["y"][robot]}
for p in sorted({k for k, _ in c}):
    print(p, c[p, "n"], f"in={c[p,'in']/c[p,'n']:.3f} uniq={c[p,'uniq']/c[p,'n']:.3f}")
