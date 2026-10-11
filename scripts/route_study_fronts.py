"""Build route Pareto fronts for the isolated structured-route study.

Only public ``scenes.json``/``labels.json`` from the requested split are read.
The output is deliberately separate from the production policy and existing
EXACT-B caches.  A front retains paths up to shortest-length + ``slack`` and
tracks enough counters to test lexicographic and detour-budget rules.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict, deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from exactB_common import load, relative_turn  # noqa: E402

FEATURES = ("edges", "crowded", "noncovered", "landmark", "right", "left", "uturn", "stairs", "oneway")


def _add(front, vec):
    """Insert a vector if it is not Pareto dominated (all counters minimized)."""
    for old in front:
        if all(x <= y for x, y in zip(old, vec)):
            return
    front[:] = [old for old in front if not all(x <= y for x, y in zip(vec, old))]
    front.append(vec)


def _one(args):
    scene, slack, legged = args
    m = scene["m"]
    goals = set(scene["goals"])
    vias = set(scene["vias"] or [])
    staged = bool(vias)
    avoid = m.lm_cells - goals - vias

    # Directed adjacency already applies the one-way constraint.  This set is
    # only a path counter, not a feasibility rule.
    directed_edges = set()
    for edge in scene["s"]["edges"]:
        if edge.get("oneway_to") and edge.get("status") != "closed":
            a, b = tuple(edge["a"]), tuple(edge["b"])
            directed_edges.add((a, b))
            directed_edges.add((b, a))

    stages = (0, 1) if staged else (1,)
    reverse = defaultdict(list)
    for u in m.nodes:
        for _, v, _, stairs in m.adj[u]:
            if stairs and not legged:
                continue
            for stage in stages:
                stage2 = int(stage == 1 or v in vias)
                reverse[(v, stage2)].append((u, stage))
    lower = {}
    queue = deque()
    for goal in goals:
        lower[(goal, 1)] = 0
        queue.append((goal, 1))
    while queue:
        state = queue.popleft()
        for prev in reverse[state]:
            if prev not in lower:
                lower[prev] = lower[state] + 1
                queue.append(prev)

    stage0 = int(not staged or m.start in vias)
    if (m.start, stage0) not in lower:
        return {"fronts": {}, "shortest": None, "ambiguous_goal": len(goals) > 1,
                "has_via": staged, "goal_count": len(goals)}
    shortest = lower[(m.start, stage0)]
    limit = shortest + slack
    completed = defaultdict(list)
    layer = {}
    for direction, node, status, stairs in m.adj[m.start]:
        if stairs and not legged:
            continue
        stage2 = int(stage0 == 1 or node in vias)
        if (node, stage2) not in lower or 1 + lower[(node, stage2)] > limit:
            continue
        turn = relative_turn(m.heading, direction)
        vec = (1, int(status == "crowded"), int(status != "covered"), int(node in avoid),
               int(turn == 1), int(turn == 3), int(turn == 2), int(stairs),
               int((m.start, node) in directed_edges))
        layer[(node, direction, stage2, direction)] = [vec]

    step = 1
    while layer and step <= limit:
        next_layer = {}
        for (node, heading, stage, first), vectors in layer.items():
            if node in goals and stage == 1:
                for vector in vectors:
                    _add(completed[first], vector)
                continue
            for direction, dest, status, stairs in m.adj[node]:
                if stairs and not legged:
                    continue
                stage2 = int(stage == 1 or dest in vias)
                if (dest, stage2) not in lower or step + 1 + lower[(dest, stage2)] > limit:
                    continue
                turn = relative_turn(heading, direction)
                inc = (1, int(status == "crowded"), int(status != "covered"), int(dest in avoid),
                       int(turn == 1), int(turn == 3), int(turn == 2), int(stairs),
                       int((node, dest) in directed_edges))
                front = next_layer.setdefault((dest, direction, stage2, first), [])
                for vector in vectors:
                    _add(front, tuple(a + b for a, b in zip(vector, inc)))
        layer = next_layer
        step += 1
    fronts = {}
    for first, vectors in completed.items():
        front = []
        for vector in sorted(vectors):
            _add(front, vector)
        fronts[first] = np.asarray(front, dtype=np.int16)
    return {"fronts": fronts, "shortest": shortest, "ambiguous_goal": len(goals) > 1,
            "has_via": staged, "goal_count": len(goals)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("train", "validation"))
    parser.add_argument("--slack", type=int, default=6)
    parser.add_argument("--legged", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    scenes = load(args.split)
    jobs = [(scene, args.slack, args.legged) for scene in scenes]
    with Pool(args.workers) as pool:
        fronts = pool.map(_one, jobs, chunksize=8)
    kind = "legged" if args.legged else "wheeled"
    output = ROOT / "work" / "ml2" / f"route_study_fronts_{args.split}_{kind}_s{args.slack}.pkl"
    payload = {"features": FEATURES, "split": args.split, "legged": args.legged,
               "slack": args.slack, "fronts": fronts}
    output.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    sizes = [sum(len(x) for x in item["fronts"].values()) for item in fronts]
    print(f"wrote {output} scenes={len(fronts)} mean_vectors={np.mean(sizes):.2f} max={max(sizes)}")


if __name__ == "__main__":
    main()
