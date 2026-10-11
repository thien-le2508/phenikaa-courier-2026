"""Enumerate shortest (by edges) paths per first move; rich features."""
from collections import deque
from exactB_common import *


def bfs_lb(m, goals, vias):
    goals = set(goals); vias = set(vias) if vias else None
    stages = (0, 1) if vias else (1,)
    rev = defaultdict(list)
    for u in m.nodes:
        for d, v, status, st in m.adj[u]:
            if st: continue
            for s_ in stages:
                s2 = 1 if (s_ == 1 or v in vias) else 0
                rev[(v, s2)].append((u, s_))
    lb = {}; dq = deque()
    for g in goals:
        lb[(g, 1)] = 0; dq.append((g, 1))
    while dq:
        x = dq.popleft()
        for p in rev[x]:
            if p not in lb:
                lb[p] = lb[x] + 1; dq.append(p)
    return lb


def enum_paths(m, goals, vias, first, extra=0, cap=3000):
    """All paths starting with move `first` with length <= lb_via_first + extra. Returns list of [(d, u, v, status)]"""
    goals_s = set(goals); vias_s = set(vias) if vias else None
    lb = bfs_lb(m, goals, vias)
    s0 = 1 if (not vias_s or m.start in vias_s) else 0
    step = {d: (v, st, sti) for d, v, st, sti in m.adj[m.start] if not sti}
    if first not in step: return []
    v, st, _ = step[first]
    s1 = 1 if (s0 == 1 or (vias_s and v in vias_s)) else 0
    if (v, s1) not in lb: return []
    L = 1 + lb[(v, s1)] + extra
    out = []
    def rec(u, s_, path):
        if len(out) >= cap: return
        if u in goals_s and s_ == 1:
            out.append(list(path)); return
        for d, w, status, sti in m.adj[u]:
            if sti: continue
            s2 = 1 if (s_ == 1 or (vias_s and w in vias_s)) else 0
            if (w, s2) not in lb or len(path) + 1 + lb[(w, s2)] > L: continue
            path.append((d, u, w, status)); rec(w, s2, path); path.pop()
    rec(v, s1, [(first, m.start, v, st)])
    return out


def path_feats(m, path, goals, vias, h0):
    avoid = m.lm_cells - set(goals) - set(vias or [])
    lt = {rc: t for t, rc in m.lm_list}
    f = defaultdict(int)
    h = h0
    for k, (d, u, v, status) in enumerate(path):
        f["E"] += 1
        f[status] += 1
        if v in avoid:
            f["lm"] += 1; f["lm_" + lt[v]] += 1
        t = relative_turn(h, d)
        if k > 0:
            f[("S", "R", "U", "L")[t] + "turn"] += 1
        h = d
        deg = len({x[1] for x in m.adj[v]})
        if deg >= 3: f["inter"] += 1
        f["deg"] += deg
    return dict(f)
