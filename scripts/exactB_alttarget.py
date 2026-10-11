import sys
from collections import Counter
from exactB_common import *
robot, split, cond = int(sys.argv[1]), sys.argv[2], sys.argv[3]
unit = lambda h, d, st, s, u, v: 1
D = load(split); c = Counter()
for sc in D:
    if not eval(cond, {}, env_of(sc)): continue
    m = sc["m"]; lab = sc["y"][robot]
    q = dijkstra_moves(m, sc["goals"], sc["vias"], unit)
    if lab in argmins(q): c["onshort"] += 1; continue
    c["off"] += 1
    if sc["vias"]:
        c["off_hasvia"] += 1
        if lab in argmins(dijkstra_moves(m, sc["goals"], None, unit)): c["goal_only"] += 1
        if lab in argmins(dijkstra_moves(m, sc["vias"], None, unit)): c["via_only"] += 1
    if len(sc["goals"]) > 1: c["off_dupgoal"] += 1
    hit = [t for t, rc in m.lm_list if lab in argmins(dijkstra_moves(m, [rc], None, unit))]
    c["any_lm_short"] += bool(hit)
print(dict(c))
