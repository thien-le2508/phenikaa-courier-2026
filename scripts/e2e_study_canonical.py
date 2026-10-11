"""Check policy invariance to harmless CV graph ordering on validation only."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from courier.data import load_split  # noqa: E402
from courier.policy import graph_from_scene, mission_from_scene, predict_all_ml  # noqa: E402


def canonical_graph(graph):
    """Normalize container order without changing graph semantics."""
    g = dict(graph)
    g["nodes"] = sorted(map(tuple, graph["nodes"]))
    edges = []
    for src in graph["edges"]:
        e = dict(src)
        e["a"], e["b"] = tuple(e["a"]), tuple(e["b"])
        e["oneway_to"] = tuple(e["oneway_to"]) if e.get("oneway_to") else None
        edges.append(e)
    g["edges"] = sorted(edges, key=lambda e: (e["a"], e["b"]))
    g["landmarks"] = sorted(((t, tuple(rc)) for t, rc in graph["landmarks"]),
                            key=lambda x: (x[1], x[0]))
    g["robot"] = tuple(graph["robot"])
    return g


def main():
    work = ROOT / "work"
    with (work / "e2e_study_policy_trainonly.pkl").open("rb") as f:
        bundle = pickle.load(f)
    with (work / "e2e_study_cv_graphs.pkl").open("rb") as f:
        cv_graphs, _ = pickle.load(f)
    _, flat_labels, scenes = load_split("validation")
    labels = np.asarray(flat_labels).reshape(len(scenes), 10)
    raw, canon = [], []
    for s, pg in zip(scenes, cv_graphs):
        m = mission_from_scene(s, drop_map_goal_type=True)
        style = s["style"]
        raw.append(predict_all_ml(dict(pg, style=style), m, bundle))
        canon.append(predict_all_ml(dict(canonical_graph(pg), style=style), m, bundle))
        # Validate exact canonical equality; then no third policy pass is needed.
        gg = canonical_graph(graph_from_scene(s))
        pc = canonical_graph(pg)
        assert all(pc[k] == gg[k] for k in
                   ("nodes", "edges", "landmarks", "robot", "heading", "rain"))
    raw, canon = map(np.asarray, (raw, canon))
    oracle = canon
    changed_raw = np.argwhere(raw != oracle)
    changed_canon = np.argwhere(canon != oracle)
    out = {
        "scope": "public validation only",
        "raw_cv_accuracy": float(np.mean(raw == labels)),
        "canonical_cv_accuracy": float(np.mean(canon == labels)),
        "canonical_oracle_accuracy": float(np.mean(oracle == labels)),
        "raw_vs_canonical_oracle": [
            {"scene": int(i), "robot": int(r), "raw": int(raw[i, r]),
             "oracle": int(oracle[i, r]), "label": int(labels[i, r])}
            for i, r in changed_raw],
        "canonical_cv_vs_canonical_oracle": [
            {"scene": int(i), "robot": int(r), "cv": int(canon[i, r]),
             "oracle": int(oracle[i, r])} for i, r in changed_canon],
    }
    path = work / "e2e_study_canonical.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
