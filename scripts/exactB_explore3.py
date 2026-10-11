from collections import Counter
from exactB_common import *
unit = lambda h, d, st, s, u, v: 1
for split in ("train", "validation"):
    D = load(split)
    for r in (1, 6, 7, 8, 9):
        c = Counter()
        for sc in D:
            mi = sc["mi"]
            q = dijkstra_moves(sc["m"], sc["goals"], sc["vias"], unit)
            key = sc["s"]["style"] + ("U" if mi["urgent"] else "-")
            c[key, sc["y"][r] in argmins(q)] += 1
        keys = sorted({k for k, _ in c})
        print(split, r, {k: f"{c[k, True]}/{c[k, True] + c[k, False]}" for k in keys})
