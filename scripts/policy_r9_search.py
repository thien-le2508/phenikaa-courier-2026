"""Brute-force interpretable one-step policies for robot 9 (research only)."""
import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from courier.data import load_split
from courier.policy import Map, graph_from_scene, mission_from_scene, relative_turn, targets

ORDERS = list(itertools.permutations(range(4)))


def dist(a, b, metric):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    if metric == "man":
        return dr + dc
    if metric == "euclid":
        return math.hypot(dr, dc)
    if metric == "sq":
        return dr * dr + dc * dc
    if metric == "cheb":
        return max(dr, dc)
    raise ValueError(metric)


def make_rows(scenes):
    result = []
    for scene in scenes:
        m = Map(graph_from_scene(scene))
        goals, vias = targets(m, mission_from_scene(scene))
        tgts = vias if vias and m.start not in vias else goals
        cands = [(d, v, status, v in m.lm_cells) for d, v, status, stairs in m.adj[m.start] if not stairs]
        result.append((m.heading, m.start, tgts, cands))
    return result


def predict(row, cfg):
    heading, start, tgts, cands = row
    target_mode, select_metric, move_metric, crowded, covered, landmark, turn, order = cfg
    if target_mode == "select":
        tgt = min(sorted(tgts), key=lambda p: (dist(start, p, select_metric), p))
        chosen = [tgt]
    else:
        chosen = tgts
    rank = {rel: i for i, rel in enumerate(order)}
    vals = []
    for direction, nxt, status, is_lm in cands:
        base = min(dist(nxt, tgt, move_metric) for tgt in chosen)
        edge = crowded if status == "crowded" else covered if status == "covered" else 0.
        rel = relative_turn(heading, direction)
        vals.append((base + edge + landmark * is_lm + turn * (rel != 0), rank[rel], direction))
    return min(vals)[2]


def arrays(rows, target_mode, select_metric, move_metric):
    shape = (len(rows), 4)
    base = np.full(shape, np.inf)
    cr = np.zeros(shape)
    cv = np.zeros(shape)
    lm = np.zeros(shape)
    turn = np.zeros(shape)
    rels = np.zeros(shape, dtype=np.int8)
    for i, (heading, start, tgts, cands) in enumerate(rows):
        if target_mode == "select":
            chosen = [min(sorted(tgts), key=lambda p: (dist(start, p, select_metric), p))]
        else:
            chosen = tgts
        for direction, nxt, status, is_lm in cands:
            base[i, direction] = min(dist(nxt, tgt, move_metric) for tgt in chosen)
            cr[i, direction] = status == "crowded"
            cv[i, direction] = status == "covered"
            lm[i, direction] = is_lm
            rels[i, direction] = relative_turn(heading, direction)
            turn[i, direction] = rels[i, direction] != 0
    return base, cr, cv, lm, turn, rels


def main():
    _, train_labels, train_scenes = load_split("train")
    _, val_labels, val_scenes = load_split("validation")
    yt = np.asarray(train_labels).reshape(-1, 10)[:, 9]
    yv = np.asarray(val_labels).reshape(-1, 10)[:, 9]
    rt, rv = make_rows(train_scenes), make_rows(val_scenes)
    top = []
    for target_mode, select_metric, move_metric in itertools.product(
            ("select", "any"), ("man", "euclid"), ("man", "euclid", "sq", "cheb")):
        aa = arrays(rt, target_mode, select_metric, move_metric)
        base, cr, cv, lm, turn, rels = aa
        for crowded, covered, landmark, turn_cost, order in itertools.product(
                (0., .5, 1., 2.), (0., -.5, .5), (0., .5, 1.), (0., .5, 1.), ORDERS):
            cfg = (target_mode, select_metric, move_metric, crowded, covered,
                   landmark, turn_cost, order)
            rank = np.asarray([order.index(k) for k in range(4)])
            cost = base + crowded * cr + covered * cv + landmark * lm + turn_cost * turn
            pt = np.argmin(cost + rank[rels] * 1e-7, axis=1)
            score = float(np.mean(pt == yt))
            if len(top) < 20 or score > top[0][0]:
                top.append((score, cfg))
                top.sort(key=lambda x: x[0])
                top = top[-20:]
    print("top train-selected formulas")
    for train_score, cfg in reversed(top):
        pv = np.fromiter((predict(row, cfg) for row in rv), dtype=np.int8)
        print(f"train={train_score:.4f} val={np.mean(pv == yv):.4f} cfg={cfg}")


if __name__ == "__main__":
    main()
