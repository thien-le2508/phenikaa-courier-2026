"""ML2 core: batched (GPU) Bellman-Ford over (node, heading, via-stage) states with linear edge costs.

State = (node, heading = direction of last move, stage = via visited). Terminal = (goal, *, 1).
Each transition carries a feature vector (FEATS); cost = W . feat. Q[d] = cost of best path whose
first move is d. Used for inverse optimisation (search / perceptron over W) and as ranker features.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from courier.policy import (LM_TYPES, Map, mission_from_scene, graph_from_scene, targets,  # noqa: E402
                            relative_turn)

DATA = ROOT / os.environ.get("COURIER_DATA", "delivery_public")

FEATS = (["edge", "crowded", "covered", "stairs", "oneway", "lm"] + ["lm_" + t for t in LM_TYPES]
         + ["t_right", "t_uturn", "t_left", "lm_sametype", "lm_goaltype", "lm_viatype"])
F = len(FEATS)
FI = {n: i for i, n in enumerate(FEATS)}


def load_split(split):
    S = json.loads((DATA / split / "scenes.json").read_text(encoding="utf-8"))
    Y = json.loads((DATA / split / "labels.json").read_text(encoding="utf-8"))
    Y = np.asarray(Y, np.int64).reshape(-1, 10)
    return S, Y


def build_scene(graph, mission):
    """Numpy description of the state graph of one scene."""
    m = Map(graph)
    goals, vias = targets(m, mission)
    goals = set(goals)
    vset = set(vias or [])
    avoid = m.lm_cells - goals - vset
    ltype = {rc: t for t, rc in m.lm_list}
    gtype = {ltype.get(g) for g in goals}
    vtype = {ltype.get(v) for v in vset}
    ow = set()
    for e in graph["edges"]:
        if e.get("oneway_to") and e["status"] != "closed":
            ow.add((tuple(e["a"]), tuple(e["b"])))
            ow.add((tuple(e["b"]), tuple(e["a"])))
    nodes = sorted(m.nodes)
    nid = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    S = N * 8
    succ = np.full((S, 4), -1, np.int32)
    feat = np.zeros((S, 4, F), np.int8)
    stairs = np.zeros((S, 4), bool)
    term = np.zeros(S, bool)

    def sid(n, h, s):
        return (nid[n] * 4 + h) * 2 + s
    for g in goals:
        for h in range(4):
            term[sid(g, h, 1)] = True
    for u in nodes:
        for d, v, status, st in m.adj[u]:
            base = np.zeros(F, np.int8)
            base[0] = 1
            base[1] = status == "crowded"
            base[2] = status == "covered"
            base[3] = st
            base[4] = (u, v) in ow
            if v in avoid:
                base[5] = 1
                base[6 + LM_TYPES.index(ltype[v])] = 1
                base[FI["lm_sametype"]] = ltype[v] in gtype or ltype[v] in vtype
                base[FI["lm_goaltype"]] = ltype[v] in gtype
                base[FI["lm_viatype"]] = ltype[v] in vtype
            for h in range(4):
                rt = relative_turn(h, d)
                fv = base.copy()
                if rt:
                    fv[16 + rt - 1] = 1
                for s in (0, 1):
                    s2 = 1 if (s == 1 or v in vset) else 0
                    x = sid(u, h, s)
                    if term[x]:
                        continue
                    succ[x, d] = sid(v, d, s2)
                    feat[x, d] = fv
                    stairs[x, d] = st
    s0 = 1 if (not vias or m.start in vset) else 0
    start = sid(m.start, m.heading, s0)
    return dict(succ=succ, feat=feat, stairs=stairs, term=term, start=start, heading=m.heading,
                N=N, goals=goals, vias=vset, nodes=nodes)


def cond_of(scene):
    ms = scene["mission"]
    return (scene["weather"] == "rain", bool(ms["urgent"]), bool(ms["fragile"]))


def build_split(split):
    S, Y = load_split(split)
    out = [build_scene(graph_from_scene(s), mission_from_scene(s)) for s in S]
    conds = np.asarray([cond_of(s) for s in S], bool)
    has_via = np.asarray([bool(o["vias"]) for o in out])
    return dict(scenes=out, Y=Y, conds=conds, has_via=has_via, style=[s["style"] for s in S])


def cache(split):
    p = ROOT / "work" / "ml2" / f"states_{split}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    d = build_split(split)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(d))
    return d


def pack(scenes, device="cuda"):
    """Pad scenes into batched torch tensors. Transition features are stored as indices into a
    codebook of unique feature rows (code 0 = invalid transition)."""
    import torch
    B = len(scenes)
    Smax = max(o["succ"].shape[0] for o in scenes)
    SINK = Smax
    succ = np.full((B, Smax, 4), SINK, np.int64)
    code = np.zeros((B, Smax, 4), np.int64)
    term = np.zeros((B, Smax + 1), bool)
    start = np.zeros(B, np.int64)
    head = np.zeros(B, np.int64)
    book = {}
    rows = [np.zeros(F + 1, np.int8)]  # last column: invalid flag
    rows[0][F] = 1
    for i, o in enumerate(scenes):
        n = o["succ"].shape[0]
        sc = o["succ"]
        succ[i, :n] = np.where(sc >= 0, sc, SINK)
        fl = o["feat"].reshape(-1, F)
        keys = [r.tobytes() for r in fl]
        cc = np.zeros(len(keys), np.int64)
        for j, (k, r) in enumerate(zip(keys, fl)):
            if k not in book:
                book[k] = len(rows)
                rows.append(np.concatenate([r, [0]]).astype(np.int8))
            cc[j] = book[k]
        cc = cc.reshape(n, 4)
        cc[sc < 0] = 0
        code[i, :n] = cc
        term[i, :n] = o["term"]
        start[i] = o["start"]
        head[i] = o["heading"]
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return dict(succ=t(succ), code=t(code), book=t(np.stack(rows)), term=t(term),
                start=t(start), head=t(head), B=B, S=Smax)


INF = 1e9


def unit_len(P, legged):
    """[B] number of edges of the shortest legal route (inf if none)."""
    import torch
    w = torch.zeros(1, F, dtype=torch.float64, device=P["succ"].device)
    w[0, FI["edge"]] = 1.0
    Q = solve(P, w, legged=legged)[0]
    return Q.min(-1).values


def solve(P, W, legged=False, max_iter=600, return_dist=False, group=None, budget=None, lstar=None):
    """W: torch [K, F] weights (float32 exact for integer costs < 2^24, else float64).
    Returns Q [K, B, 4] cost of the best path starting with each move (INF if illegal/unreachable)."""
    import torch
    K = W.shape[0]
    B, S = P["B"], P["S"]
    dev = W.device
    book = P["book"].to(W.dtype)
    badrow = book[:, F] > 0
    if not legged:
        badrow = badrow | (book[:, FI["stairs"]] > 0)
    if W.dim() == 2:
        cb = book[:, :F] @ W.T  # [U, K]
        cb = cb.masked_fill(badrow.unsqueeze(1), INF).T.contiguous()  # [K, U]
        cost = cb[:, P["code"].view(-1)].view(K, B, S, 4)
    else:  # per-group weights W [K, G, F], group [B]
        U = book.shape[0]
        cb = torch.einsum("uf,kgf->kgu", book[:, :F], W).masked_fill(badrow.view(1, 1, -1), INF)
        cb = cb.reshape(K, -1)
        idx = (group.view(B, 1, 1) * U + P["code"]).view(-1)
        cost = cb[:, idx].view(K, B, S, 4)
    term = P["term"].unsqueeze(0).expand(K, B, S + 1)
    dist = torch.full((K, B, S + 1), INF, device=dev, dtype=W.dtype).masked_fill(term, 0.0)
    succ = P["succ"].view(1, B, S * 4).expand(K, B, S * 4)
    pad = torch.full((K, B, 1), INF, device=dev, dtype=W.dtype)
    if budget is not None:
        # Route length limited to T = L* + budget[k] edges (resource-constrained shortest path).
        st = P["start"].view(1, B, 1, 1).expand(K, B, 1, 4)
        c0 = torch.gather(cost, 2, st).squeeze(2)
        sn = torch.gather(P["succ"], 1, P["start"].view(B, 1, 1).expand(B, 1, 4)).squeeze(1)
        snk = sn.unsqueeze(0).expand(K, B, 4)
        ls = torch.where(torch.isfinite(lstar) & (lstar < 1e6), lstar, torch.zeros_like(lstar)).long()
        T = (ls.view(1, B) + budget.view(K, 1).long())  # [K,B]
        maxT = int(T.max())
        Q = torch.full((K, B, 4), INF, device=dev, dtype=W.dtype)
        for t in range(1, maxT + 1):  # dist == V_{t-1}
            Qt = c0 + torch.gather(dist, 2, snk)
            Q = torch.where((T == t).unsqueeze(-1), Qt, Q)
            nxt = torch.gather(dist, 2, succ).view(K, B, S, 4)
            new = (cost + nxt).min(-1).values.clamp(max=INF)
            new = torch.cat([new, pad], 2).masked_fill(term, 0.0)
            dist = torch.minimum(dist, new)
        return Q.masked_fill(Q >= INF / 2, INF)
    for it in range(max_iter):
        nxt = torch.gather(dist, 2, succ).view(K, B, S, 4)
        new = (cost + nxt).min(-1).values.clamp(max=INF)
        new = torch.cat([new, pad], 2).masked_fill(term, 0.0)
        nd = torch.minimum(dist, new)
        if it % 3 == 2 and torch.equal(nd, dist):
            break
        dist = nd
    st = P["start"].view(1, B, 1, 1).expand(K, B, 1, 4)
    c0 = torch.gather(cost, 2, st).squeeze(2)
    sn = torch.gather(P["succ"], 1, P["start"].view(B, 1, 1).expand(B, 1, 4)).squeeze(1)
    d0 = torch.gather(dist, 2, sn.unsqueeze(0).expand(K, B, 4))
    Q = (c0 + d0)
    Q = Q.masked_fill(Q >= INF / 2, INF)
    if return_dist:
        return Q, dist, cost
    return Q


def rel_rank_table(order):
    """order: tuple of relative turns (0 straight, 1 right, 2 U, 3 left), best first.
    Returns int [4 headings, 4 dirs] rank (lower = preferred)."""
    tab = np.zeros((4, 4), np.int64)
    for h in range(4):
        for d in range(4):
            tab[h, d] = order.index(relative_turn(h, d))
    return tab


def abs_rank_table(order):
    """order: tuple of absolute dirs best first."""
    tab = np.zeros((4, 4), np.int64)
    for h in range(4):
        for d in range(4):
            tab[h, d] = order.index(d)
    return tab


def pick(Q, head, order_tab, tol=1e-6):
    """Q [K,B,4]; order_tab torch [4,4] (heading, dir) -> rank. Returns [K,B] argmin with tie-break."""
    import torch
    rank = order_tab[head]  # [B,4]
    best = Q.min(-1, keepdim=True).values
    tie = Q <= best + tol
    score = torch.where(tie, rank.unsqueeze(0).to(Q.dtype), torch.full_like(Q, 99.0))
    return score.argmin(-1)
