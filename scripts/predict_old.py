"""Bản nộp nhanh vòng private: CV + NLP của vòng trước (code và checkpoint cũ, không train lại)
ghép với chiến thuật tạm courier/policy_quick.py của vòng này.

    python scripts/predict_old.py --split test_private --out predictions.json
    python scripts/predict_old.py --split validation --out work/val_pred_old.json   # có nhãn thì in điểm
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OLD = Path(r"Z:\hackathon\Phenikaa_Campus_Courier_2026\Phenikaa_Campus_Courier_2026")
os.environ.setdefault("COURIER_DATA", str(HERE / "delivery_public"))
sys.path.insert(0, str(OLD))  # courier.cv / courier.nlp / courier.data của vòng trước

from courier.cv.reader import MapReader  # noqa: E402
from courier.data import DATA, load_split, macro_accuracy, scene_rows  # noqa: E402
from courier.nlp.infer import MissionReader  # noqa: E402

_spec = importlib.util.spec_from_file_location("policy_quick", HERE / "courier" / "policy_quick.py")
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_private")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--heat", default=str(OLD / "work/heat_strong.pt"))
    ap.add_argument("--crops", default=str(OLD / "work/crops_final.pt"))
    ap.add_argument("--nlp", default=str(OLD / "work/nlp_final3.pt"))
    a = ap.parse_args()

    rows, labels, _ = load_split(a.split)
    scenes = scene_rows(rows)
    mapr = MapReader(a.heat, a.crops)
    mr = MissionReader(a.nlp)

    t0 = time.time()
    graphs = []
    for i, (image, _) in enumerate(scenes):
        try:
            graphs.append(mapr.read(str(DATA / a.split / image))[0])
        except Exception:  # không bao giờ bỏ trống một cảnh
            traceback.print_exc()
            graphs.append(None)
        if (i + 1) % 100 == 0:
            print(f"đã đọc {i + 1}/{len(scenes)} ảnh, {time.time() - t0:.0f}s", flush=True)

    texts = [m for _, m in scenes]
    lms = [[t for t, _ in g["landmarks"]] if g else None for g in graphs]
    missions = mr.read(texts, lms)

    preds = []
    for g, m in zip(graphs, missions):
        try:
            p = policy.predict_all(g, m)
        except Exception:
            traceback.print_exc()
            p = None
        if p is None:
            h = policy.HEADING.get(g["heading"], 0) if g and isinstance(g["heading"], str) else 0
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
