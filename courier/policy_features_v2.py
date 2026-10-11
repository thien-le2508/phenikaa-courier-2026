"""Structured route-summary features for the learned policy ranker.

The graph solver remains exact. ML only learns how each robot values the
resulting path summaries and local lookahead signals.
"""
import heapq
import math

from courier.policy import (LM_TYPES, Map, _man, move_features, relative_turn,
                            targets)


# A compact set of deliberately different route objectives. The old feature
# bank still remains in move_features(); these objectives expose *why* a route
# wins by returning counts along its selected path.
SUMMARY_BANK = [
    {},
    {"cr": 30},
    {"cr": 60, "lm": 20},
    {"cv": 2},
    {"cv": 2, "cr": 30, "lm": 20},
    {"lm": 40},
    {"tr": 15, "tl": 15, "tu": 150},
    {"cr": 40, "lm": 20, "tr": 10, "tl": 10, "tu": 100},
]


def _edge_cost(cfg, status, stairs, dest, avoid, lm_type, heading, direction):
    cost = cfg.get("cr", 10) if status == "crowded" else cfg.get("cv", 10) if status == "covered" else 10
    if dest in avoid:
        cost += cfg.get("lm", 0) + cfg.get("lmt", {}).get(lm_type.get(dest), 0)
    rel = relative_turn(heading, direction)
    cost += (0, cfg.get("tr", 0), cfg.get("tu", 0), cfg.get("tl", 0))[rel]
    cost += cfg.get("stairs", 0) if stairs else 0
    return float(cost)


def _add_stats(stats, status, stairs, dest, avoid, lm_type, heading, direction):
    z = list(stats)
    z[0] += 1
    z[1 + ("normal", "crowded", "covered").index(status)] += 1
    z[4] += int(stairs)
    z[5] += int(dest in avoid)
    rel = relative_turn(heading, direction)
    z[6 + rel] += 1
    typ = lm_type.get(dest) if dest in avoid else None
    if typ in LM_TYPES:
        z[10 + LM_TYPES.index(typ)] += 1
    return tuple(z)


def _route_summary(m, goals, vias, first, cfg, legged):
    """Best completion after a fixed first edge: (cost, 20 path counters)."""
    goals = set(goals)
    via_set = set(vias or [])
    avoid = m.lm_cells - goals - via_set
    lm_type = {rc: typ for typ, rc in m.lm_list}
    direction, node, status, stairs = first
    if stairs and not legged:
        return None
    stage = not via_set or m.start in via_set or node in via_set
    zero = (0,) * (10 + len(LM_TYPES))
    stats = _add_stats(zero, status, stairs, node, avoid, lm_type, m.heading, direction)
    cost = _edge_cost(cfg, status, stairs, node, avoid, lm_type, m.heading, direction)
    seq = 0
    pq = [(cost, 1, seq, node, direction, int(stage), stats)]
    best = {}
    while pq:
        cur_cost, steps, _, u, heading, done, cur_stats = heapq.heappop(pq)
        state = (u, heading, done)
        if state in best and best[state] <= (cur_cost, steps):
            continue
        best[state] = (cur_cost, steps)
        if done and u in goals:
            return cur_cost, cur_stats
        for nd, v, st, is_stairs in m.adj[u]:
            if is_stairs and not legged:
                continue
            next_done = int(done or v in via_set)
            nc = cur_cost + _edge_cost(cfg, st, is_stairs, v, avoid, lm_type, heading, nd)
            ns = _add_stats(cur_stats, st, is_stairs, v, avoid, lm_type, heading, nd)
            seq += 1
            heapq.heappush(pq, (nc, steps + 1, seq, v, nd, next_done, ns))
    return None


def _local_features(m, goals, vias, first):
    direction, v, status, stairs = first
    active = list(vias) if vias and m.start not in set(vias) else list(goals)
    if not active:
        active = [m.start]
    man = [_man(v, g) for g in active]
    euc = [math.hypot(v[0] - g[0], v[1] - g[1]) for g in active]
    start_man = min(_man(m.start, g) for g in active)
    start_euc = min(math.hypot(m.start[0] - g[0], m.start[1] - g[1]) for g in active)
    outgoing = [(d, w, st, ss) for d, w, st, ss in m.adj[v] if not ss]
    two_man = min((_man(w, g) for _, w, _, _ in outgoing for g in active), default=min(man))
    two_euc = min((math.hypot(w[0] - g[0], w[1] - g[1]) for _, w, _, _ in outgoing for g in active), default=min(euc))
    nearest = min(active, key=lambda g: (_man(v, g), g))
    rows = [p[0] for p in m.nodes]
    cols = [p[1] for p in m.nodes]
    span_r = max(max(rows) - min(rows), 1)
    span_c = max(max(cols) - min(cols), 1)
    return [
        min(man) / 10, max(man) / 10, min(euc) / 10, max(euc) / 10,
        (min(man) - start_man) / 5, (min(euc) - start_euc) / 5,
        two_man / 10, two_euc / 10,
        (two_man - min(man)) / 5, (two_euc - min(euc)) / 5,
        (nearest[0] - v[0]) / span_r, (nearest[1] - v[1]) / span_c,
        len(outgoing) / 4, float(len(outgoing) <= 1),
        sum(st == "normal" for _, _, st, _ in outgoing) / 4,
        sum(st == "crowded" for _, _, st, _ in outgoing) / 4,
        sum(st == "covered" for _, _, st, _ in outgoing) / 4,
        float(status == "normal"), float(status == "crowded"), float(status == "covered"),
        float(stairs), float(v in set(active)), float(direction),
    ]


def move_features_v2(graph, mission):
    base = move_features(graph, mission)
    m = Map(graph)
    goals, vias = targets(m, mission)
    if not goals:
        return base
    shared = [
        len(m.nodes) / 30, sum(len(x) for x in m.adj.values()) / 60,
        len(m.lm_list) / 20, len(m.adj[m.start]) / 4,
        len(goals) / 2, len(vias or []) / 2,
    ]
    out_by_legged = {}
    for legged in (False, True):
        representative = 4 if legged else 1
        firsts = {d: (d, v, status, stairs) for d, v, status, stairs in m.adj[m.start]
                  if legged or not stairs}
        summaries = []
        for cfg in SUMMARY_BANK:
            summaries.append({d: _route_summary(m, goals, vias, edge, cfg, legged)
                              for d, edge in firsts.items()})
        rows = {}
        for direction, old in base[representative].items():
            extra = []
            for bank in summaries:
                result = bank.get(direction)
                valid = [x for x in bank.values() if x is not None]
                if result is None or not valid:
                    extra.extend([50.0] + [0.0] * (2 + 10 + len(LM_TYPES)))
                    continue
                cost, stats = result
                best_cost = min(x[0] for x in valid)
                costs = sorted(x[0] for x in valid)
                margin = costs[1] - costs[0] if len(costs) > 1 else 0.0
                steps = max(stats[0], 1)
                # Relative cost, winner margin, then normalized path counters.
                extra.extend([(cost - best_cost) / 10, margin / 10, cost / (10 * steps)])
                extra.extend([stats[0] / 10])
                extra.extend([stats[i] / steps for i in range(1, 10)])
                extra.extend([stats[i] / steps for i in range(10, 10 + len(LM_TYPES))])
            extra.extend(_local_features(m, goals, vias, firsts[direction]))
            rows[direction] = list(old) + extra + shared
        out_by_legged[legged] = rows
    return {robot: out_by_legged[robot == 4] for robot in range(10)}
