from collections import Counter
from exactB_common import *
unit = lambda h, d, st, s, u, v: 1
for split in ("train", "validation"):
    D = load(split)
    for r in (0, 6, 7, 8, 9):
        c = Counter()
        for sc in D:
            mi = sc["mi"]; m = sc["m"]
            q = dijkstra_moves(m, sc["goals"], sc["vias"], unit)
            lab = sc["y"][r]
            c["U" if mi["urgent"] else "-", "labelUturn", relative_turn(m.heading, lab) == 2] += 1
            qn = {d: v for d, v in q.items() if relative_turn(m.heading, d) != 2}
            c["U" if mi["urgent"] else "-", "inNonU", lab in argmins(qn) if qn else None] += 1
        print(split, r, sorted(c.items(), key=str))
