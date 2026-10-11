"""Chấm khâu CV trên validation: từng phần của đồ thị và macro accuracy hướng đi.

    python scripts/eval_cv.py --heat work/heat.pt --crops work/crops.pt [--nlp work/nlp_holdout.pt]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.cv.reader import MapReader  # noqa: E402
from courier.data import DATA, load_split, macro_accuracy  # noqa: E402
from courier.policy import graph_from_scene, mission_from_scene, predict_all  # noqa: E402


def edge_key(e):
    a, b = tuple(e["a"]), tuple(e["b"])
    ow = tuple(e["oneway_to"]) if e.get("oneway_to") else None
    return (min(a, b), max(a, b)), (e["status"], bool(e["stairs"]), ow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heat", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--nlp")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--dump")
    ap.add_argument("--recover", type=float, default=0.05, help="ngưỡng nhặt lại giao lộ, 0 = tắt")
    a = ap.parse_args()
    reader = MapReader(a.heat, a.crops, recover_thr=a.recover)
    rows, labels, scenes = load_split(a.split)
    ok = Counter()
    edge_ok = edge_n = 0
    preds_gt_mission, graphs, bad = [], [], []
    for i, s in enumerate(scenes):
        g, aux = reader.read(str(DATA / a.split / s["image"]))
        gold = graph_from_scene(s)
        graphs.append(g)
        chk = {
            "nodes": sorted(map(tuple, g["nodes"])) == sorted(gold["nodes"]),
            "robot": tuple(g["robot"]) == gold["robot"] and g["heading"] == gold["heading"],
            "landmarks": sorted(g["landmarks"]) == sorted((t, tuple(rc)) for t, rc in gold["landmarks"]),
            "weather": g["rain"] == gold["rain"],
            "legend": all(aux["look_to_status"].get(s["road_look"].get(st, st)) == st
                          for st in ("normal", "crowded", "covered")),
        }
        ge = dict(edge_key(e) for e in gold["edges"])
        pe = dict(edge_key(e) for e in g["edges"])
        chk["edges"] = ge == pe
        for k in set(ge) | set(pe):
            edge_n += 1
            edge_ok += ge.get(k) == pe.get(k)
        chk["graph"] = all(chk.values())
        for k, v in chk.items():
            ok[k] += v
        if not chk["graph"]:
            bad.append((i, s["style"], [k for k, v in chk.items() if not v and k != "graph"]))
        preds_gt_mission += predict_all(g, mission_from_scene(s))
    n = len(scenes)
    print("cảnh đúng hoàn toàn theo phần:", {k: round(v / n, 4) for k, v in ok.items()})
    print(f"đoạn đường đúng: {edge_ok / edge_n:.4f} ({edge_n - edge_ok} sai / {edge_n})")
    print("cảnh sai (chỉ số, kiểu vẽ, phần sai):", bad[:40])
    print("theo kiểu vẽ:", Counter(st for _, st, _ in bad))
    macro, per = macro_accuracy(rows, labels, preds_gt_mission)
    print(f"[CV dự đoán + yêu cầu chuẩn] macro={macro:.4f}", {r: round(v, 3) for r, v in per.items()})
    if a.nlp:
        from courier.nlp.infer import MissionReader
        mr = MissionReader(a.nlp)
        missions = mr.read([s["mission"]["text"] for s in scenes],
                           [[t for t, _ in g["landmarks"]] for g in graphs])
        preds = []
        for g, m in zip(graphs, missions):
            preds += predict_all(g, m)
        macro, per = macro_accuracy(rows, labels, preds)
        print(f"[CV dự đoán + NLP dự đoán] macro={macro:.4f}", {r: round(v, 3) for r, v in per.items()})
    if a.dump:
        json.dump(bad, open(a.dump, "w"))


if __name__ == "__main__":
    main()
