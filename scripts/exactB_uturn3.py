import sys
from collections import Counter
from exactB_common import *
from exactB_paths import bfs_lb
split = sys.argv[1] if len(sys.argv) > 1 else "train"
D = load(split)
for robot in range(10):
    c = Counter()
    for sc in D:
        m = sc["m"]; mi = sc["mi"]
        lb = bfs_lb(m, sc["goals"], sc["vias"])
        vs = set(sc["vias"] or [])
        s0 = 1 if (not sc["vias"] or m.start in vs) else 0
        L = {}
        for d, v, st, sti in m.adj[m.start]:
            if sti and robot != 4: continue
            s1 = 1 if (s0 or v in vs) else 0
            if (v, s1) in lb: L[d] = 1 + lb[(v, s1)]
        if not L: continue
        bd = min(L.values())
        U = [d for d in L if relative_turn(m.heading, d) == 2]
        if not U or L[U[0]] != bd: continue
        straight = any(relative_turn(m.heading, d) == 0 for d, *_ in m.adj[m.start])
        nonu_short = any(L[d] == bd for d in L if d != U[0])
        key = ("S" if straight else "noS", "alt" if nonu_short else "only", "F" if mi["fragile"] else "-")
        c[key, sc["y"][robot] == U[0]] += 1
    print(robot, {k: f"{c[k, True]}/{c[k, True] + c[k, False]}" for k in sorted({k for k, _ in c})})
