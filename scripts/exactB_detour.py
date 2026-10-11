"""Scenes where label not on a shortest path: compare label's best (min-E) path rich features to the shortest path's."""
import sys
from collections import Counter
from exactB_common import *
from exactB_paths import enum_paths, path_feats, bfs_lb
robot, split, cond = int(sys.argv[1]), sys.argv[2], sys.argv[3]
D = load(split)
c = Counter(); n = 0; dl = Counter()
for sc in D:
    env = env_of(sc)
    if not eval(cond, {}, env): continue
    m = sc["m"]
    lb = bfs_lb(m, sc["goals"], sc["vias"]); vs = set(sc["vias"] or [])
    s0 = 1 if (not sc["vias"] or m.start in vs) else 0
    L = {}
    for d, v, st, sti in m.adj[m.start]:
        if sti: continue
        s1 = 1 if (s0 or v in vs) else 0
        if (v, s1) in lb: L[d] = 1 + lb[(v, s1)]
    lab = sc["y"][robot]
    if lab not in L or L[lab] == min(L.values()): continue
    n += 1; dl[L[lab] - min(L.values())] += 1
    def mins(d, extra=0):
        P = enum_paths(m, sc["goals"], sc["vias"], d, extra=extra, cap=2000)
        fs = [path_feats(m, p, sc["goals"], sc["vias"], m.heading) for p in P]
        ks = set().union(*fs) if fs else set()
        return {k: min(f.get(k, 0) for f in fs) for k in ks}
    a = mins(lab)
    for d in L:
        if L[d] != min(L.values()): continue
        b = mins(d)
        rel = relative_turn(m.heading, d)
        a2 = dict(a); a2["firstU"] = int(relative_turn(m.heading, lab) == 2); b["firstU"] = int(rel == 2)
        for k in set(a2) | set(b):
            x, y = a2.get(k, 0), b.get(k, 0)
            c[k, "<" if x < y else (">" if x > y else "=")] += 1
print("n", n, "detour lengths", dict(dl))
for k in sorted({k for k, _ in c}, key=str):
    if c[k, "<"] + c[k, ">"] >= 5: print(f"{k:14s} lab< {c[k, '<']:4d} lab> {c[k, '>']:4d} = {c[k, '=']}")
