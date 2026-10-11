import sys
from exactB_common import *
from exactB_paths import enum_paths, path_feats, bfs_lb
robot, split, cond, n = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
keys = sys.argv[5].split(",")
D = load(split); k = 0
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
    def mins(d):
        P = enum_paths(m, sc["goals"], sc["vias"], d, cap=2000)
        fs = [path_feats(m, p, sc["goals"], sc["vias"], m.heading) for p in P]
        ks = set().union(*fs)
        r = {kk: min(f.get(kk, 0) for f in fs) for kk in ks}; r["firstU"] = int(relative_turn(m.heading, d) == 2); return r
    a = mins(lab)
    expl = False
    for d in L:
        if L[d] != min(L.values()): continue
        b = mins(d)
        if any(a.get(kk, 0) < b.get(kk, 0) for kk in keys): expl = True
    if expl: continue
    k += 1
    print(render(sc), "style", sc["s"]["style"])
    for d in L:
        b = mins(d); print("  ", NAMES[d], "SRUL"[relative_turn(m.heading, d)], {kk: b.get(kk, 0) for kk in ["E", "crowded", "covered", "lm", "Uturn", "Lturn", "Rturn", "inter"]}, "<-LABEL" if d == lab else "")
    if k >= n: break
