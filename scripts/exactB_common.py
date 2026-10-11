"""Shared helpers for EXACT-B analysis (R6-R9)."""
import json, os, sys, heapq
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from courier.policy import Map, graph_from_scene, mission_from_scene, targets, relative_turn, rel_order, CLOCKWISE  # noqa
DATA = ROOT / os.environ.get("COURIER_DATA", "delivery_public")
NAMES = "UDLR"


def load(split):
    S = json.loads((DATA / split / "scenes.json").read_text(encoding="utf-8"))
    Y = json.loads((DATA / split / "labels.json").read_text(encoding="utf-8"))
    out = []
    for i, s in enumerate(S):
        g, mi = graph_from_scene(s), mission_from_scene(s)
        m = Map(g)
        goals, vias = targets(m, mi)
        fix = os.environ.get("FIX", "none")
        if fix != "none":
            e2 = lambda a, b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2  # noqa: E731
            if vias and len(vias) > 1:
                vias = [min(sorted(vias), key=lambda t: e2(m.start, t))]
            if goals and len(goals) > 1:
                ref = vias[0] if (vias and fix == "chain") else m.start
                goals = [min(sorted(goals), key=lambda t: e2(ref, t))]
        out.append(dict(i=i, s=s, m=m, mi=mi, goals=goals, vias=vias, y=Y[10 * i:10 * i + 10]))
    return out


def render(sc):
    s, m = sc["s"], sc["m"]
    R, C = s["grid"]["rows"], s["grid"]["cols"]
    ABBR = {"library": "TV", "dorm": "KT", "sports": "TT", "clinic": "YT", "canteen": "CA", "parking": "XE",
            "lecture": "GD", "lab": "TN", "office": "HC", "gate": "CT"}
    edge = {frozenset((tuple(e["a"]), tuple(e["b"]))): e for e in s["edges"]}
    lmat = {rc: ABBR[t] for t, rc in m.lm_list}

    def sym(e, fwd, a, b):
        if e["status"] == "closed":
            return "x"
        if e.get("stairs"):
            return "S"
        if e.get("oneway_to"):
            return a if tuple(e["oneway_to"]) == fwd else b
        return {"normal": "-", "crowded": "=", "covered": "~"}[e["status"]]
    lines = ["    " + "".join(f"{c:<4d}" for c in range(C))]
    for r in range(R):
        row, below = f"{r:2d}  ", "    "
        for c in range(C):
            u = (r, c)
            if u not in m.nodes:
                cell = "   "
            elif u == m.start:
                cell = "R" + "^v<>"[m.heading] + " "
            elif u in lmat:
                cell = lmat[u] + " "
            else:
                cell = " o "
            if u in sc["goals"]:
                cell = cell[:2] + "*"
            elif sc["vias"] and u in sc["vias"]:
                cell = cell[:2] + "+"
            row += cell
            e = edge.get(frozenset((u, (r, c + 1))))
            row += sym(e, (r, c + 1), ">", "<") if e else " "
            e = edge.get(frozenset((u, (r + 1, c))))
            below += " " + (sym(e, (r + 1, c), "v", "^") if e else " ") + "  "
        lines.append(row)
        lines.append(below)
    mi = s["mission"]
    hdr = (f"#{sc['i']} rain={m.rain} urg={mi['urgent']} frag={mi['fragile']} goal={mi['goal']} {mi['goal_ref']} via={mi['via']} {mi['via_ref']} "
           f"head={NAMES[m.heading]} labels={''.join(NAMES[x] for x in sc['y'])}")
    return hdr + "\n" + "\n".join(lines)


def dijkstra_moves(m, goals, vias, cfn, legged=False):
    """cfn(h, d, status, stairs, u, v) -> cost (h heading before move). Returns {d: total} for first moves."""
    goals = set(goals)
    vias = set(vias) if vias else None
    stages = (0, 1) if vias else (1,)
    rev = defaultdict(list)
    for u in m.nodes:
        for d, v, status, st in m.adj[u]:
            if st and not legged:
                continue
            for h in range(4):
                c = cfn(h, d, status, st, u, v)
                for s_ in stages:
                    s2 = 1 if (s_ == 1 or v in vias) else 0
                    rev[(v, d, s2)].append(((u, h, s_), c))
    dist = {}
    pq = [(0, (g, h, 1)) for g in goals for h in range(4)]
    heapq.heapify(pq)
    while pq:
        c, st_ = heapq.heappop(pq)
        if st_ in dist:
            continue
        dist[st_] = c
        for p, w in rev[st_]:
            if p not in dist:
                heapq.heappush(pq, (c + w, p))
    s0 = 1 if (not vias or m.start in vias) else 0
    q = {}
    for d, v, status, st in m.adj[m.start]:
        if st and not legged:
            continue
        c = cfn(m.heading, d, status, st, m.start, v)
        s2 = 1 if (s0 == 1 or v in vias) else 0
        rest = dist.get((v, d, s2))
        if rest is not None:
            q[d] = c + rest
    return q


def argmins(q):
    if not q:
        return set()
    b = min(q.values())
    return {d for d, c in q.items() if c == b}


import re as _re
import unicodedata as _ud


def strip_acc(s):
    s = _ud.normalize("NFD", s.lower())
    s = "".join(c for c in s if _ud.category(c) != "Mn")
    return s.replace("đ", "d")


URG_PHR = {"dang can gap lam": "gaplam", "gap nhe": "gapnhe", "can ngay trong 5 phut": "5phut", "hoa toc": "hoatoc",
           "viec nay khan cap": "khancap", "cang nhanh cang tot": "cangnhanh", "cu tu tu": "tutu",
           "khong gap dau": "khonggap", "chieu nay": "chieunay"}


def urg_phrase(text):
    t = strip_acc(text)
    hits = [v for k, v in URG_PHR.items() if k in t]
    return hits[0] if len(hits) == 1 else ("none" if not hits else "+".join(hits))


FRAG_NEG = ("do khong vo duoc dau", "hang chac chan, khong de vo")
URG_NEG = ("cu tu tu", "khong gap dau", "chieu nay")


def env_of(sc):
    t = strip_acc(sc["s"]["mission"]["text"])
    mi = sc["mi"]
    return dict(U=mi["urgent"], F=mi["fragile"], Fr=mi["fragile"], R=sc["m"].rain, V=bool(sc["vias"]),
                N=sc["s"]["style"] == "night", S=sc["s"]["style"],
                UX=(not mi["urgent"]) and any(k in t for k in URG_NEG),
                FX=(not mi["fragile"]) and any(k in t for k in FRAG_NEG), K=urg_phrase(sc["s"]["mission"]["text"]))
