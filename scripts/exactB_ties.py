"""Given weights, evaluate tie orders (relative and absolute) per condition group.
args: robot cond split weights(dict: E cr nc lm tr tl tu fr fl fu) groupexpr"""
import sys, itertools
import numpy as np
from collections import defaultdict, Counter
from exactB_common import *
from exactB_wsearch import prep, cond_mask

robot, cond, split, w = int(sys.argv[1]), sys.argv[2], sys.argv[3], eval(sys.argv[4])
gexpr = sys.argv[5] if len(sys.argv) > 5 else "0"
D0 = load(split)
P = prep(split, cond_mask(D0, cond))
D, idx = P["D"], P["idx"]
W = np.array([w.get(k, 0) for k in ["E", "cr", "nc", "lm", "tr", "tl", "tu"]], float)
W[0] = w.get("E", 100)
C = P["R"] @ W
M = np.minimum.reduceat(C, P["starts"])
f = {0: 0, 1: w.get("fr", 0), 2: w.get("fu", 0), 3: w.get("fl", 0)}
groups = defaultdict(list)
miss = []
for j, i in enumerate(idx):
    sc = D[i]
    a = P["scene_start"][j]; b = P["scene_start"][j + 1] if j + 1 < len(idx) else len(M)
    q = {P["segmove"][k]: M[k] + f[relative_turn(sc["m"].heading, P["segmove"][k])] for k in range(a, b)}
    am = argmins(q)
    lab = sc["y"][robot]
    if lab not in am:
        miss.append(i)
        continue
    g = eval(gexpr, {}, dict(U=sc["mi"]["urgent"], F=sc["mi"]["fragile"], R=sc["m"].rain, N=sc["s"]["style"] == "night", S=sc["s"]["style"], H=sc["m"].heading))
    groups[g].append((sc["m"].heading, am, lab))
print("miss", len(miss), miss[:40])
for g, L in sorted(groups.items(), key=str):
    ties = [x for x in L if len(x[1]) > 1]
    res = []
    for perm in itertools.permutations(range(4)):
        okr = sum(next(d for d in rel_order(h, perm) if d in am) == lab for h, am, lab in ties)
        oka = sum(next(d for d in perm if d in am) == lab for h, am, lab in ties)
        res.append((okr, "rel " + "".join("SRUL"[k] for k in perm)))
        res.append((oka, "abs " + "".join("UDLR"[k] for k in perm)))
    res.sort(reverse=True)
    print("group", g, "n", len(L), "ties", len(ties), res[:4])
