"""Chấm khâu chiến thuật trên đồ thị + yêu cầu chuẩn của scenes.json.

    python scripts/eval_policy.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.data import load_split, macro_accuracy  # noqa: E402
from courier.policy import graph_from_scene, mission_from_scene, predict_all  # noqa: E402

for split in ("train", "validation"):
    rows, labels, scenes = load_split(split)
    preds = []
    for s in scenes:
        preds += predict_all(graph_from_scene(s), mission_from_scene(s))
    score, per = macro_accuracy(rows, labels, preds)
    print(f"{split}: macro {score:.4f}", {r: round(v, 4) for r, v in per.items()})
