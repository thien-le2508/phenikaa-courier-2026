import sys
from exactB_common import *
split, r, cond, n = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
unit = lambda h, d, st, s, u, v: 1
D = load(split)
k = 0
for sc in D:
    mi = sc["mi"]
    if not eval(cond):
        continue
    q = dijkstra_moves(sc["m"], sc["goals"], sc["vias"], unit)
    print(render(sc))
    print("unit:", {NAMES[d]: c for d, c in q.items()}, "label", NAMES[sc["y"][r]], "| text:", sc["s"]["mission"]["text"])
    print()
    k += 1
    if k >= n:
        break
