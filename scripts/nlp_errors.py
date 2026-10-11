"""In các câu validation bị đọc sai (sau giải mã ràng buộc theo bản đồ) và lỗi đó có làm sai hướng đi không.

    python scripts/nlp_errors.py work/nlp_holdout.valpred.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.data import load_split  # noqa: E402
from courier.policy import graph_from_scene, mission_from_scene, predict_all  # noqa: E402


def nref(r):
    return (r["kind"], r["anchor"] if r["kind"] in ("near", "far") else None) if r else None


preds = json.load(open(sys.argv[1], encoding="utf-8"))
rows, labels, scenes = load_split("validation")
fields = ("goal", "via", "goal_ref", "via_ref", "urgent", "fragile")
cnt, cnt_move = Counter(), Counter()
for i, (s, p) in enumerate(zip(scenes, preds)):
    g = mission_from_scene(s)
    bad = [f for f in fields if (nref(p[f]) != nref(g[f]) if f.endswith("ref") else p[f] != g[f])]
    if not bad:
        continue
    mv = predict_all(graph_from_scene(s), p)
    wrong_moves = sum(a != b for a, b in zip(mv, labels[10 * i:10 * i + 10]))
    for f in bad:
        cnt[f] += 1
        cnt_move[f] += wrong_moves > 0
    print(f"#{i} sai {bad} -> {wrong_moves} robot sai")
    print("   ", s["mission"]["text"])
    for f in bad:
        print(f"    {f}: đoán {p[f]}  | đúng {g[f]}")
print("số câu sai theo trường:", dict(cnt))
print("trong đó làm sai hướng đi:", dict(cnt_move))
