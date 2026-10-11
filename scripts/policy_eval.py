"""Eval policy on GT graph + GT mission (no GT rc). Also checks resolver vs goal_ref.rc / via_ref.rc,
and the variant where the goal type is dropped for map-described goals.
    COURIER_DATA=delivery_public python scripts/policy_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.data import load_split, macro_accuracy  # noqa: E402
from courier.policy import Map, graph_from_scene, mission_from_scene, predict_all, targets  # noqa: E402

for split in ("train", "validation"):
    rows, labels, scenes = load_split(split)
    bad_ref = 0
    for drop in (False, True):
        preds = []
        for s in scenes:
            g, mi = graph_from_scene(s), mission_from_scene(s, drop_map_goal_type=drop)
            m = Map(g)
            goals, vias = targets(m, mi)
            sm = s["mission"]
            if sm["goal_ref"] and goals != [tuple(sm["goal_ref"]["rc"])]:
                bad_ref += 1
            if sm["via_ref"] and vias != [tuple(sm["via_ref"]["rc"])]:
                bad_ref += 1
            preds += predict_all(g, mi)
        score, per = macro_accuracy(rows, labels, preds)
        print(f"{split} drop_goal_type={drop}: macro {score:.4f}", {r: round(v, 4) for r, v in per.items()})
    print(f"{split}: ref resolution mismatches (both variants) = {bad_ref}")
