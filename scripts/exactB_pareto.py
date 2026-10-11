"""Per first move, Pareto front of path feature vectors (layered forward DP).
Features: [edges, crowded, noncovered, covered, lm, right, left, uturn, oneway]  (edges handled by layer)
Saves work/exactB_pareto_{split}.pkl : list over scenes of {d: np.array(K, F)}"""
import pickle, sys
import numpy as np
from multiprocessing import Pool
from exactB_common import *
from collections import deque

SLACK = int(sys.argv[2]) if len(sys.argv) > 2 else 6
FN = ["edges", "crowded", "noncov", "lm", "right", "left", "uturn"]


def dominated(a, b):  # b dominates a
    return all(x <= y for x, y in zip(b, a))


def add(front, vec):
    for f in front:
        if all(x <= y for x, y in zip(f, vec)):
            return False
    front[:] = [f for f in front if not all(x <= y for x, y in zip(vec, f))]
    front.append(vec)
    return True


def fronts(sc, slack=SLACK):
    m, goals, vias = sc["m"], set(sc["goals"]), set(sc["vias"]) if sc["vias"] else None
    avoid = m.lm_cells - goals - (vias or set())
    # reverse BFS on (node, stage) for lower bound
    rev = defaultdict(list)
    stages = (0, 1) if vias else (1,)
    for u in m.nodes:
        for d, v, status, st in m.adj[u]:
            if st:
                continue
            for s_ in stages:
                s2 = 1 if (s_ == 1 or v in vias) else 0
                rev[(v, s2)].append((u, s_))
    lb = {}
    dq = deque()
    for g in goals:
        lb[(g, 1)] = 0
        dq.append((g, 1))
    while dq:
        x = dq.popleft()
        for p in rev[x]:
            if p not in lb:
                lb[p] = lb[x] + 1
                dq.append(p)
    s0 = 1 if (not vias or m.start in vias) else 0
    if (m.start, s0) not in lb or (m.start in goals and s0 == 1):
        return {}
    L = lb[(m.start, s0)] + slack
    out = defaultdict(list)
    layer = {}
    for d, v, status, st in m.adj[m.start]:
        if st:
            continue
        s2 = 1 if (s0 == 1 or v in vias) else 0
        if (v, s2) not in lb or 1 + lb[(v, s2)] > L:
            continue
        t = relative_turn(m.heading, d)
        vec = (1, int(status == "crowded"), int(status != "covered"), int(v in avoid),
               int(t == 1), int(t == 3), int(t == 2))
        layer.setdefault((v, d, s2, d), []).append(vec)
    step = 1
    while layer and step <= L:
        nxt = {}
        for (u, h, s_, fm), vecs in layer.items():
            if u in goals and s_ == 1:
                for vec in vecs:
                    add(out[fm], vec)
                continue
            for d, v, status, st in m.adj[u]:
                if st:
                    continue
                s2 = 1 if (s_ == 1 or v in vias) else 0
                if (v, s2) not in lb or step + 1 + lb[(v, s2)] > L:
                    continue
                t = relative_turn(h, d)
                inc = (1, int(status == "crowded"), int(status != "covered"), int(v in avoid),
                       int(t == 1), int(t == 3), int(t == 2))
                key = (v, d, s2, fm)
                fr = nxt.setdefault(key, [])
                for vec in vecs:
                    add(fr, tuple(a + b for a, b in zip(vec, inc)))
        layer = nxt
        step += 1
    # final prune per move
    res = {}
    for fm, vecs in out.items():
        fr = []
        for v in sorted(vecs):
            add(fr, v)
        res[fm] = np.array(fr, dtype=np.int32)
    return res


def work(sc):
    return fronts(sc)


if __name__ == "__main__":
    split = sys.argv[1]
    D = load(split)
    with Pool(16) as p:
        F = p.map(work, D, chunksize=8)
    pickle.dump(F, open(ROOT / f"work/exactB_pareto_{split}_s{SLACK}.pkl", "wb"))
    print(split, "sizes", np.mean([sum(len(v) for v in f.values()) for f in F]), max(sum(len(v) for v in f.values()) for f in F))
