"""Chạy toàn bộ pipeline (ảnh + yêu cầu -> hướng đi của 10 robot) và ghi file nộp.

    python scripts/predict.py --split test --out predictions.json
    python scripts/predict.py --split validation --out work/val_pred.json      # có nhãn thì in điểm

Chỉ đọc ảnh, câu yêu cầu và robot_id; không dùng scenes.json.
"""
import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.cv.reader import MapReader  # noqa: E402
from courier.data import DATA, load_split, macro_accuracy, scene_rows  # noqa: E402
from courier.nlp.infer import MissionReader  # noqa: E402
from courier.policy import HEADING, predict_all, predict_all_ml  # noqa: E402
from courier.style import predict_style  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--heat", default="work/heat_strong.pt")
    ap.add_argument("--crops", default="work/crops_final.pt")
    ap.add_argument("--nlp", default="work/nlp_final3.pt")
    ap.add_argument("--policy-ml", help="bundle policy học được; bỏ trống để dùng luật viết tay")
    ap.add_argument("--tta", action="store_true", help="TTA đảo thứ tự câu cho NLP (bản v3 dùng; model nlp v3 thì tắt)")
    ap.add_argument("--no-mid-filter", action="store_true", help="tắt bộ lọc đỉnh giữa đoạn (để tái tạo đúng bản v3)")
    a = ap.parse_args()

    rows, labels, _ = load_split(a.split)
    scenes = scene_rows(rows)
    crop_paths = a.crops.split(",") if "," in a.crops else a.crops
    mapr = MapReader(a.heat, crop_paths, opts={"mid_filter": not a.no_mid_filter})
    mr = MissionReader(a.nlp)
    if a.policy_ml:
        with open(a.policy_ml, "rb") as f:
            policy_bundle = pickle.load(f)
    else:
        policy_bundle = None

    t0 = time.time()
    graphs = []
    for i, (image, _) in enumerate(scenes):
        try:
            image_path = str(DATA / a.split / image)
            graph = mapr.read(image_path)[0]
            if policy_bundle is not None:
                graph["style"] = predict_style(policy_bundle["style_model"], image_path)
            graphs.append(graph)
        except Exception:  # không bao giờ bỏ trống một cảnh
            traceback.print_exc()
            graphs.append(None)
        if (i + 1) % 100 == 0:
            print(f"đã đọc {i + 1}/{len(scenes)} ảnh, {time.time() - t0:.0f}s", flush=True)

    texts = [m for _, m in scenes]
    lms = [[t for t, _ in g["landmarks"]] if g else None for g in graphs]
    missions = mr.read(texts, lms, tta=a.tta)

    preds = []
    for g, m in zip(graphs, missions):
        try:
            p = predict_all_ml(g, m, policy_bundle) if policy_bundle is not None else predict_all(g, m)
        except Exception:
            traceback.print_exc()
            try:
                p = predict_all(g, m) if g is not None else None
            except Exception:
                traceback.print_exc()
                p = None
        if p is None:
            h = HEADING[g["heading"]] if g else 0
            p = [h] * 10
        preds += [int(x) for x in p]

    assert len(preds) == len(rows) and all(x in (0, 1, 2, 3) for x in preds)
    Path(a.out).write_text(json.dumps(preds), encoding="utf-8")
    print(f"đã ghi {len(preds)} dự đoán vào {a.out} ({time.time() - t0:.0f}s)")
    if labels:
        macro, per = macro_accuracy(rows, labels, preds)
        print(f"macro accuracy = {macro:.4f}", {r: round(v, 3) for r, v in per.items()})


if __name__ == "__main__":
    main()
