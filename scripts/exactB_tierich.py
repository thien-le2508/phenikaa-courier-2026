import sys, pickle
from collections import Counter
from exactB_common import *
from exactB_paths import *
robot, cond, split = int(sys.argv[1]), sys.argv[2], sys.argv[3]
fu = int(sys.argv[4]) if len(sys.argv) > 4 else 2
D = load(split)
stats = Counter(); n = 0
for sc in D:
    mi, m = sc["mi"], sc["m"]
    if not eval(cond, {}, dict(U=mi["urgent"], F=mi["fragile"], R=m.rain, N=sc["s"]["style"] == "night")): continue
    lb = bfs_lb(m, sc["goals"], sc["vias"])
    vias_s = set(sc["vias"]) if sc["vias"] else set()
    s0 = 1 if (not sc["vias"] or m.start in vias_s) else 0
    q = {}
    for d, v, st, sti in m.adj[m.start]:
        if sti: continue
        s1 = 1 if (s0 or v in vias_s) else 0
        if (v, s1) in lb: q[d] = 1 + lb[(v, s1)] + (fu if relative_turn(m.heading, d) == 2 else 0)
    am = argmins(q); lab = sc["y"][robot]
    if lab not in am or len(am) < 2: continue
    n += 1
    mins = {}
    for d in am:
        P = enum_paths(m, sc["goals"], sc["vias"], d)
        fs = [path_feats(m, p, sc["goals"], sc["vias"], m.heading) for p in P]
        keys = set().union(*fs)
        mins[d] = {k: min(f.get(k, 0) for f in fs) for k in keys}
        mins[d].update({"max_" + k: max(f.get(k, 0) for f in fs) for k in keys})
    allk = set().union(*[set(x) for x in mins.values()])
    for d in am - {lab}:
        for k in allk:
            a, b = mins[lab].get(k, 0), mins[d].get(k, 0)
            stats[k, "<" if a < b else (">" if a > b else "=")] += 1
print("tied", n)
for k in sorted({k for k, _ in stats}, key=str):
    lt, gt = stats[k, "<"], stats[k, ">"]
    if lt + gt >= 10: print(f"{str(k):22s} lab< {lt:4d}  lab> {gt:4d}  = {stats[k, '=']}")
