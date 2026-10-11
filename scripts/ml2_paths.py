"""CPU helpers: best path per first move under weights w (dict), with summed feature counts."""
import heapq, sys
sys.path.insert(0, 'scripts')
import numpy as np
import ml2_core as c


def wfull(w, E=1000.0):
    v = np.zeros(c.F); v[c.FI['edge']] = E
    for k, x in w.items():
        if k in c.FI: v[c.FI[k]] = x
    return v


def best_paths(o, w, legged=False):
    v = wfull(w)
    succ, feat, term = o['succ'], o['feat'], o['term']
    S = succ.shape[0]
    cost = feat.astype(float) @ v
    # reverse dijkstra
    rev = [[] for _ in range(S)]
    for x in range(S):
        for d in range(4):
            y = succ[x, d]
            if y >= 0 and (legged or not o['stairs'][x, d]):
                rev[y].append((x, d))
    dist = np.full(S, np.inf); nxt = np.full(S, -1)
    pq = [(0.0, x) for x in np.where(term)[0]]
    for _, x in pq: dist[x] = 0
    heapq.heapify(pq)
    while pq:
        dd, y = heapq.heappop(pq)
        if dd > dist[y]: continue
        for x, d in rev[y]:
            nd = dd + cost[x, d]
            if nd < dist[x] - 1e-9:
                dist[x] = nd; nxt[x] = d; heapq.heappush(pq, (nd, x))
    out = {}
    s0 = o['start']
    for d in range(4):
        y = succ[s0, d]
        if y < 0 or (o['stairs'][s0, d] and not legged) or not np.isfinite(dist[y]): continue
        f = feat[s0, d].astype(int).copy(); path = [d]; x = y; n = 0
        while not term[x] and n < 300:
            dd = nxt[x]; f += feat[x, dd]; path.append(dd); x = succ[x, dd]; n += 1
        out[d] = (cost[s0, d] + dist[y], ''.join('UDLR'[k] for k in path), {c.FEATS[i]: int(f[i]) for i in np.nonzero(f)[0]})
    return out
