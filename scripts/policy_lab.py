"""Analysis helpers for reverse-engineering robot policies (train/validation only)."""
import heapq
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / os.environ.get("COURIER_DATA", "delivery_public")
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
HEADING = {"UP": UP, "DOWN": DOWN, "LEFT": LEFT, "RIGHT": RIGHT}
CW = [UP, RIGHT, DOWN, LEFT]
NAMES = "UDLR"


def rel(h, d):
    """0 straight, 1 right, 2 uturn, 3 left"""
    return (CW.index(d) - CW.index(h)) % 4


def load(split):
    S = json.loads((DATA / split / "scenes.json").read_text(encoding="utf-8"))
    Y = json.loads((DATA / split / "labels.json").read_text(encoding="utf-8"))
    out = []
    for i, s in enumerate(S):
        out.append(Scene(s, Y[10 * i:10 * i + 10]))
    return out


class Scene:
    def __init__(self, s, y):
        self.s = s
        self.y = y
        self.id = s["scene_id"]
        self.nodes = {tuple(n["rc"]) for n in s["nodes"]}
        self.R, self.C = s["grid"]["rows"], s["grid"]["cols"]
        self.adj = defaultdict(list)  # u -> [(d, v, status, stairs, oneway)]
        self.edge = {}
        for e in s["edges"]:
            a, b = tuple(e["a"]), tuple(e["b"])
            self.edge[frozenset((a, b))] = e
            if e["status"] == "closed":
                continue
            ow = tuple(e["oneway_to"]) if e.get("oneway_to") else None
            for u, v in ((a, b), (b, a)):
                if ow is not None and ow != v:
                    continue
                d = next(k for k, dd in DELTA.items() if dd == (v[0] - u[0], v[1] - u[1]))
                self.adj[u].append((d, v, e["status"], bool(e.get("stairs")), ow is not None))
        self.lms = [(l["type"], tuple(l["rc"])) for l in s["landmarks"]]
        self.lm = defaultdict(list)
        for t, rc in self.lms:
            self.lm[t].append(rc)
        self.start = tuple(s["robot"]["rc"])
        self.heading = HEADING[s["robot"]["heading"]]
        self.rain = s["weather"] == "rain"
        m = s["mission"]
        self.m = m
        self.urgent, self.fragile = m["urgent"], m["fragile"]
        self.goal_type = m["goal"]
        self.via_type = m["via"]
        self.goals = [tuple(m["goal_ref"]["rc"])] if m["goal_ref"] else sorted(self.lm[m["goal"]])
        if m["via"]:
            self.vias = [tuple(m["via_ref"]["rc"])] if m["via_ref"] else sorted(self.lm[m["via"]])
        else:
            self.vias = None


def first_move_costs(sc, goals, vias, ecost, tcost=None, legged=False, start=None, heading=None):
    """ecost(status, stairs, u, v, d) -> cost; tcost(h, d) -> cost. Returns {d: total}."""
    start = sc.start if start is None else start
    heading = sc.heading if heading is None else heading
    goals = set(goals)
    vias = set(vias) if vias else None
    headings = range(4) if tcost else (0,)
    stages = (0, 1) if vias else (1,)
    rev = defaultdict(list)
    for u in sc.nodes:
        for d, v, status, st, ow in sc.adj[u]:
            if st and not legged:
                continue
            c0 = ecost(status, st, u, v, d)
            for h in headings:
                c = c0 + (tcost(h, d) if tcost else 0)
                for s in stages:
                    s2 = 1 if (s == 1 or v in vias) else 0
                    rev[(v, d if tcost else 0, s2)].append(((u, h, s), c))
    dist = {}
    pq = [(0, (g, h, 1)) for g in goals for h in headings]
    heapq.heapify(pq)
    while pq:
        c, s = heapq.heappop(pq)
        if s in dist:
            continue
        dist[s] = c
        for p, w in rev[s]:
            if p not in dist:
                heapq.heappush(pq, (c + w, p))
    s0 = 1 if (not vias or start in vias) else 0
    q = {}
    for d, v, status, st, ow in sc.adj[start]:
        if st and not legged:
            continue
        c = ecost(status, st, start, v, d) + (tcost(heading, d) if tcost else 0)
        s2 = 1 if (s0 == 1 or v in vias) else 0
        rest = dist.get((v, d if tcost else 0, s2))
        if rest is not None:
            q[d] = c + rest
    return q


def W(normal=10, crowded=10, covered=10, stairs=0):
    t = {"normal": normal, "crowded": crowded, "covered": covered}
    return lambda status, st, u, v, d: t[status] + (stairs if st else 0)


def T(right, left, uturn):
    t = [0, right, uturn, left]
    return lambda h, d: t[rel(h, d)]


def argmins(q):
    if not q:
        return set()
    b = min(q.values())
    return {d for d, c in q.items() if c == b}


ABBR = {"library": "TV", "dorm": "KT", "sports": "TT", "clinic": "YT", "canteen": "CA", "parking": "XE",
        "lecture": "GD", "lab": "TN", "office": "HC", "gate": "CT"}


def render(sc):
    """ASCII map. Node cells 3 chars wide; edges: '-' normal, '=' crowded, '~' covered, ' ' none/closed('x'),
    'S' stairs, '>' '<' '^' 'v' oneway."""
    lines = []
    lmat = {rc: ABBR[t] for t, rc in sc.lms}
    for r in range(sc.R):
        row, below = "", ""
        for c in range(sc.C):
            u = (r, c)
            if u not in sc.nodes:
                cell = "   "
            elif u == sc.start:
                cell = "R" + "^v<>"[sc.heading] + " "
            elif u in lmat:
                cell = lmat[u] + " "
            else:
                cell = " o "
            if u in sc.goals:
                cell = cell[:2] + "*"
            elif sc.vias and u in sc.vias:
                cell = cell[:2] + "+"
            row += cell
            e = sc.edge.get(frozenset((u, (r, c + 1))))
            row += sym(e, (r, c + 1), ">", "<") if e else " "
            e = sc.edge.get(frozenset((u, (r + 1, c))))
            below += " " + (sym(e, (r + 1, c), "v", "^") if e else " ") + "  "
        lines.append(row)
        lines.append(below)
    return "\n".join(lines)


def sym(e, fwd, a, b):
    if e["status"] == "closed":
        return "x"
    if e.get("stairs"):
        return "S"
    if e.get("oneway_to"):
        return a if tuple(e["oneway_to"]) == fwd else b
    return {"normal": "-", "crowded": "=", "covered": "~"}[e["status"]]


def describe(sc):
    m = sc.m
    return (f"{sc.id} {sc.s['style']} rain={sc.rain} urgent={sc.urgent} fragile={sc.fragile} goal={m['goal']} {m['goal_ref']} "
            f"via={m['via']} {m['via_ref']} start={sc.start} head={NAMES[sc.heading]} labels={''.join(NAMES[x] for x in sc.y)}")


FEATS = ["normal", "crowded", "covered", "stairs", "lm", "oneway", "right", "left", "uturn", "f_right", "f_left", "f_uturn"]


def step_feats(sc, status, st, ow, v, avoid, h, d):
    f = [0.0] * len(FEATS)
    f[{"normal": 0, "crowded": 1, "covered": 2}[status]] = 1
    if st:
        f[3] = 1
    if v in avoid:
        f[4] = 1
    if ow:
        f[5] = 1
    t = rel(h, d)
    if t == 1:
        f[6] = 1
    elif t == 3:
        f[7] = 1
    elif t == 2:
        f[8] = 1
    return f


def first_move_feats(sc, w, legged=False, featfn=step_feats, goals=None, vias=None):
    """Reverse Dijkstra with linear cost w·f; returns {d: (cost, featvec)} for first moves."""
    goals = set(sc.goals if goals is None else goals)
    vias = sc.vias if vias is None else vias
    vias = set(vias) if vias else None
    avoid = {p for _, p in sc.lms} - goals - (vias or set())
    stages = (0, 1) if vias else (1,)
    rev = defaultdict(list)
    for u in sc.nodes:
        for d, v, status, st, ow in sc.adj[u]:
            if st and not legged:
                continue
            for h in range(4):
                f = featfn(sc, status, st, ow, v, avoid, h, d)
                c = sum(a * b for a, b in zip(w, f))
                for s in stages:
                    s2 = 1 if (s == 1 or v in vias) else 0
                    rev[(v, d, s2)].append(((u, h, s), c, f))
    dist = {}
    feat = {}
    pq = []
    for g in goals:
        for h in range(4):
            pq.append((0.0, (g, h, 1), None))
    heapq.heapify(pq)
    cnt = 0
    while pq:
        c, s, fv = heapq.heappop(pq)[:3]
        if s in dist:
            continue
        dist[s] = c
        feat[s] = fv if fv is not None else [0.0] * len(FEATS)
        for p, wc, f in rev[s]:
            if p not in dist:
                cnt += 1
                heapq.heappush(pq, (c + wc, p, [a + b for a, b in zip(feat[s], f)]))
    s0 = 1 if (not vias or sc.start in vias) else 0
    q = {}
    for d, v, status, st, ow in sc.adj[sc.start]:
        if st and not legged:
            continue
        f = featfn(sc, status, st, ow, v, avoid, sc.heading, d)
        t = rel(sc.heading, d)
        if t:
            f[{1: 9, 3: 10, 2: 11}[t]] += 1
        s2 = 1 if (s0 == 1 or v in vias) else 0
        key = (v, d, s2)
        if key in dist:
            q[d] = (sum(a * b for a, b in zip(w, f)) + dist[key], [a + b for a, b in zip(f, feat[key])])
    return q


def best_paths(sc, ecost, legged=False, k=3):
    """For each first move: the k cheapest simple-ish paths (by ecost(status,st,u,v,d)) to target,
    via handled. Uses DFS with bound = best+slack. Returns {d: [(cost, [(v,status,...)])]}"""
    goals = set(sc.goals)
    vias = set(sc.vias) if sc.vias else None
    # reverse dijkstra distances for pruning
    q = first_move_costs(sc, sc.goals, sc.vias, ecost, None, legged)
    return q


def path_strings(sc, ecost, legged=False, slack=0):
    """Enumerate all optimal (within slack) paths for each first move via DFS on (node, stage) w/ dist pruning."""
    goals = set(sc.goals)
    vias = set(sc.vias) if sc.vias else None
    stages = (0, 1) if vias else (1,)
    rev = defaultdict(list)
    for u in sc.nodes:
        for d, v, status, st, ow in sc.adj[u]:
            if st and not legged:
                continue
            c = ecost(status, st, u, v, d)
            for s in stages:
                s2 = 1 if (s == 1 or v in vias) else 0
                rev[(v, s2)].append(((u, s), c))
    dist = {}
    pq = [(0, (g, 1)) for g in goals]
    heapq.heapify(pq)
    while pq:
        c, s = heapq.heappop(pq)
        if s in dist:
            continue
        dist[s] = c
        for p, w in rev[s]:
            if p not in dist:
                heapq.heappush(pq, (c + w, p))
    out = {}
    s0 = 1 if (not vias or sc.start in vias) else 0
    for d, v, status, st, ow in sc.adj[sc.start]:
        if st and not legged:
            continue
        c0 = ecost(status, st, sc.start, v, d)
        s2 = 1 if (s0 == 1 or v in vias) else 0
        if (v, s2) not in dist:
            continue
        tot = c0 + dist[(v, s2)]
        # follow one optimal path greedily
        path = [(NAMES[d], v, status[:2] + ("S" if st else "") + ("1" if ow else ""))]
        node, stg, rem = v, s2, dist[(v, s2)]
        guard = 0
        while not (node in goals and stg == 1) and guard < 200:
            guard += 1
            for dd, vv, sst, sts, oww in sc.adj[node]:
                if sts and not legged:
                    continue
                cc = ecost(sst, sts, node, vv, dd)
                ns = 1 if (stg == 1 or (vias and vv in vias)) else 0
                if (vv, ns) in dist and dist[(vv, ns)] + cc == rem:
                    path.append((NAMES[dd], vv, sst[:2] + ("S" if sts else "") + ("1" if oww else "")))
                    node, stg, rem = vv, ns, dist[(vv, ns)]
                    break
        out[d] = (tot, path)
    return out


def first_move_costs_h(sc, goals, vias, cfn, legged=False, start=None, heading=None):
    """General: cfn(h, d, status, stairs, u, v, ow) -> cost, h = heading before the move. Returns {d: total}."""
    start = sc.start if start is None else start
    heading = sc.heading if heading is None else heading
    goals = set(goals)
    vias = set(vias) if vias else None
    stages = (0, 1) if vias else (1,)
    rev = defaultdict(list)
    for u in sc.nodes:
        for d, v, status, st, ow in sc.adj[u]:
            if st and not legged:
                continue
            for h in range(4):
                c = cfn(h, d, status, st, u, v, ow)
                for s in stages:
                    s2 = 1 if (s == 1 or v in vias) else 0
                    rev[(v, d, s2)].append(((u, h, s), c))
    dist = {}
    pq = [(0, (g, h, 1)) for g in goals for h in range(4)]
    heapq.heapify(pq)
    while pq:
        c, s = heapq.heappop(pq)
        if s in dist:
            continue
        dist[s] = c
        for p, w in rev[s]:
            if p not in dist:
                heapq.heappush(pq, (c + w, p))
    s0 = 1 if (not vias or start in vias) else 0
    q = {}
    for d, v, status, st, ow in sc.adj[start]:
        if st and not legged:
            continue
        c = cfn(heading, d, status, st, start, v, ow)
        s2 = 1 if (s0 == 1 or v in vias) else 0
        rest = dist.get((v, d, s2))
        if rest is not None:
            q[d] = c + rest
    return q
