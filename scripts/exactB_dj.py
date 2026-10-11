"""Parallel Dijkstra-based evaluation of linear cost configs with rich features.
Feature names per transition: E, crowded, covered, normal, lm, lm_<type>, tR, tL, tU (later turns), oneway.
First-move turn costs: fR, fL, fU.  goal choice: all copies (min).  """
import heapq, os, sys
import numpy as np
from multiprocessing import Pool
from exactB_common import *

LMT = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
_G = {}


def _init(split):
    _G["D"] = load(split)


def scene_q(sc, w):
    m, goals, vias = sc["m"], set(sc["goals"]), set(sc["vias"]) if sc["vias"] else None
    avoid = m.lm_cells - goals - (vias or set())
    lt = {rc: t for t, rc in m.lm_list}
    st_cost = {"normal": w.get("normal", 0), "crowded": w.get("crowded", 0), "covered": w.get("covered", 0)}
    E = w.get("E", 100)
    turn = [0, w.get("tR", 0), w.get("tU", 0), w.get("tL", 0)]
    lmw = w.get("lm", 0)
    stages = (0, 1) if vias else (1,)

    def ec(u, v, status):
        c = E + st_cost[status]
        if v in avoid:
            c += lmw + w.get("lm_" + lt[v], 0)
        return c
    rev = defaultdict(list)
    for u in m.nodes:
        for d, v, status, st in m.adj[u]:
            if st:
                continue
            c0 = ec(u, v, status)
            for h in range(4):
                c = c0 + turn[relative_turn(h, d)]
                for s_ in stages:
                    s2 = 1 if (s_ == 1 or v in vias) else 0
                    rev[(v, d, s2)].append(((u, h, s_), c))
    dist = {}
    pq = [(0, (g, h, 1)) for g in goals for h in range(4)]
    heapq.heapify(pq)
    while pq:
        c, s = heapq.heappop(pq)
        if s in dist:
            continue
        dist[s] = c
        for p, wc in rev[s]:
            if p not in dist:
                heapq.heappush(pq, (c + wc, p))
    s0 = 1 if (not vias or m.start in vias) else 0
    first = [0, w.get("fR", 0), w.get("fU", 0), w.get("fL", 0)]
    q = {}
    for d, v, status, st in m.adj[m.start]:
        if st:
            continue
        s2 = 1 if (s0 == 1 or v in vias) else 0
        rest = dist.get((v, d, s2))
        if rest is not None:
            q[d] = ec(m.start, v, status) + first[relative_turn(m.heading, d)] + rest
    return q


def _work(args):
    w, idxs = args
    D = _G["D"]
    return [scene_q(D[i], w) for i in idxs]


class Evaluator:
    def __init__(self, split, procs=16):
        self.D = load(split)
        self.pool = Pool(procs, initializer=_init, initargs=(split,))
        self.procs = procs

    def qs(self, w, idxs):
        chunks = [idxs[k::self.procs] for k in range(self.procs)]
        res = self.pool.map(_work, [(w, c) for c in chunks])
        out = {}
        for c, r in zip(chunks, res):
            out.update(zip(c, r))
        return out

    def many(self, ws, idxs):
        jobs = []
        for wi, w in enumerate(ws):
            for k in range(0, len(idxs), 64):
                jobs.append((wi, w, idxs[k:k + 64]))
        res = self.pool.map(_work, [(w, c) for _, w, c in jobs])
        out = [dict() for _ in ws]
        for (wi, w, c), r in zip(jobs, res):
            out[wi].update(zip(c, r))
        return out


def stats(D, Q, robot):
    n = len(Q); inn = uni = 0
    for i, q in Q.items():
        am = argmins(q)
        lab = D[i]["y"][robot]
        inn += lab in am
        uni += am == {lab}
    return inn / n, uni / n
