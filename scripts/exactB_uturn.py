import sys
from collections import Counter
from exactB_common import *
from exactB_paths import bfs_lb
robot = int(sys.argv[1]); split = sys.argv[2] if len(sys.argv) > 2 else "train"
D = load(split)
c = Counter(); ex = {}
for sc in D:
    m = sc["m"]; mi = sc["mi"]
    lb = bfs_lb(m, sc["goals"], sc["vias"])
    vs = set(sc["vias"] or [])
    s0 = 1 if (not sc["vias"] or m.start in vs) else 0
    L = {}
    for d, v, st, sti in m.adj[m.start]:
        if sti: continue
        s1 = 1 if (s0 or v in vs) else 0
        if (v, s1) in lb: L[d] = 1 + lb[(v, s1)]
    U = [d for d in L if relative_turn(m.heading, d) == 2]
    N = [d for d in L if relative_turn(m.heading, d) != 2]
    if not U or not N: continue
    du = L[U[0]] - min(L[d] for d in N)
    lab = sc["y"][robot]
    key = (du, "N" if sc["s"]["style"] == "night" else "-", "U" if mi["urgent"] else "-", "F" if mi["fragile"] else "-", "R" if m.rain else "-")
    c[key, lab == U[0]] += 1
for du in (-3, -2, -1):
    keys = sorted({k for k, _ in c if k[0] == du})
    print("du", du, {k[1:]: f"{c[k, True]}/{c[k, True] + c[k, False]}" for k in keys})
