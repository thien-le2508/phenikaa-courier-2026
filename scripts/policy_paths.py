"""Enumerate candidate routes per first move (length <= unit-shortest + SLACK, revisits allowed) and store
deduplicated path feature vectors. Analysis cache for exact reverse-engineering.

    python scripts/policy_paths.py train 6
"""
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_lab import load, rel, DELTA  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
FN = ["normal", "crowded", "covered", "stairs"] + ["lm_" + t for t in TYPES] + \
    ["oneway", "right", "left", "uturn", "f_right", "f_left", "f_uturn", "lm_tgtcopy", "near_lm", "revisit", "f_crowded", "f_covered", "f_lm", "goal_early"]
NF = len(FN)
SLACK = 6
CAP = 20000


def unit_dist(sc, legged):
    """stage-aware unit distance to finish: dist[(node, stage)]"""
    goals = set(sc.goals)
    vias = set(sc.vias) if sc.vias else None
    stages = (0, 1) if vias else (1,)
    rev = defaultdict(list)
    for u in sc.nodes:
        for d, v, st, s, ow in sc.adj[u]:
            if s and not legged:
                continue
            for g in stages:
                g2 = 1 if (g == 1 or v in vias) else 0
                rev[(v, g2)].append((u, g))
    dist = {(g, 1): 0 for g in goals}
    frontier = list(dist)
    while frontier:
        nxt = []
        for s in frontier:
            for p in rev[s]:
                if p not in dist:
                    dist[p] = dist[s] + 1
                    nxt.append(p)
        frontier = nxt
    return dist


def enum_scene(args):
    sc, legged = args
    goals = set(sc.goals)
    vias = set(sc.vias) if sc.vias else None
    tgt = goals | (vias or set())
    ltype = {rc: t for t, rc in sc.lms}
    # copies of goal/via type that are not the chosen target
    gtypes = {ltype[g] for g in tgt if g in ltype}
    dist = unit_dist(sc, legged)
    s0 = 1 if (not vias or sc.start in vias) else 0
    if (sc.start, s0) not in dist:
        return {}, -1
    for slack in range(SLACK, -1, -1):
        res = _enum(sc, legged, goals, vias, tgt, ltype, gtypes, dist, s0, dist[(sc.start, s0)] + slack)
        if res is not None:
            return {d: np.array(sorted(v), dtype=np.int16) for d, v in res.items()}, slack
    return {}, -1


def _enum(sc, legged, goals, vias, tgt, ltype, gtypes, dist, s0, bound):
    out = defaultdict(set)
    count = [0]

    lmc = {rc for rc in ltype if rc not in tgt}
    near = {u for u in sc.nodes if u not in ltype and any((u[0] + a, u[1] + b) in lmc for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))}
    visited = {sc.start: 1}

    def dfs(u, stage, h, length, f, first):
        if count[0] > CAP * 4:
            return
        if u in goals and stage == 1:
            out[first].add(tuple(f))
            count[0] += 1
            return
        for d, v, st, s, ow in sc.adj[u]:
            if s and not legged:
                continue
            g2 = 1 if (stage == 1 or (vias and v in vias)) else 0
            if (v, g2) not in dist or length + 1 + dist[(v, g2)] > bound:
                continue
            f2 = list(f)
            f2[{"normal": 0, "crowded": 1, "covered": 2}[st]] += 1
            if s:
                f2[3] += 1
            if v in ltype and v not in tgt:
                if ltype[v] in gtypes:
                    f2[21] += 1
                f2[4 + TYPES.index(ltype[v])] += 1
            if ow:
                f2[14] += 1
            if v in near:
                f2[22] += 1
            if visited.get(v):
                f2[23] += 1
            if first is None:
                if st == "crowded":
                    f2[24] += 1
                if st == "covered":
                    f2[25] += 1
                if v in lmc:
                    f2[26] += 1
            if vias and stage == 0 and v in goals:
                f2[27] += 1
            t = rel(h, d)
            if first is None:
                if t:
                    f2[{1: 18, 3: 19, 2: 20}[t]] += 1
            elif t:
                f2[{1: 15, 3: 16, 2: 17}[t]] += 1
            visited[v] = visited.get(v, 0) + 1
            dfs(v, g2, d, length + 1, f2, d if first is None else first)
            visited[v] -= 1

    dfs(sc.start, s0, sc.heading, 0, [0] * NF, None)
    if count[0] > CAP * 4:
        return None
    return out


if __name__ == "__main__":
    split = sys.argv[1]
    if len(sys.argv) > 2:
        SLACK = int(sys.argv[2])
    S = load(split)
    with Pool(15) as pool:
        res0 = pool.map(enum_scene, [(sc, False) for sc in S], chunksize=4)
        res1 = pool.map(enum_scene, [(sc, True) for sc in S], chunksize=4)
    sl0 = [x[1] for x in res0]
    sl1 = [x[1] for x in res1]
    res0 = [x[0] for x in res0]
    res1 = [x[0] for x in res1]
    print("slack hist", np.bincount(np.array(sl0) + 1))
    Y = np.array([sc.y for sc in S])
    tot = sum(len(a) for r in res0 for a in r.values())
    print(split, "paths(nonlegged)", tot, "max per scene", max(sum(len(a) for a in r.values()) for r in res0))
    with open(ROOT / "work" / f"paths_{split}.pkl", "wb") as f:
        pickle.dump({"feat_names": FN, "nonlegged": res0, "legged": res1, "Y": Y,
                     "ids": [sc.id for sc in S], "slack": SLACK, "slack0": sl0, "slack1": sl1}, f)
