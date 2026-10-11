"""Đọc dữ liệu và chấm điểm giống BTC."""
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# COURIER_DATA trỏ tới bộ dữ liệu khác (vd. bộ vòng private test) mà không cần sửa code
DATA = Path(os.environ["COURIER_DATA"]) if os.environ.get("COURIER_DATA") else ROOT / "delivery_public"
LANDMARKS = ["library", "dorm", "sports", "clinic", "canteen", "parking", "lecture", "lab", "office", "gate"]
REF_KINDS = ["north", "south", "west", "east", "near", "far"]


def _read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_split(split):
    """Trả về (rows, labels|None, scenes|None) của một split."""
    d = DATA / split
    rows = _read(d / "observations.json")
    labels = _read(d / "labels.json") if (d / "labels.json").exists() else None
    scenes = _read(d / "scenes.json") if (d / "scenes.json").exists() else None
    return rows, labels, scenes


def scene_rows(rows):
    """Nhóm 10 dòng liên tiếp thành một cảnh: [(image, mission), ...]."""
    out = []
    for i in range(0, len(rows), 10):
        grp = rows[i:i + 10]
        assert [r["robot_id"] for r in grp] == list(range(10))
        assert len({(r["image"], r["mission"]) for r in grp}) == 1
        out.append((grp[0]["image"], grp[0]["mission"]))
    return out


def macro_accuracy(rows, labels, preds):
    per = defaultdict(list)
    for row, y, p in zip(rows, labels, preds):
        per[row["robot_id"]].append(y == p)
    scores = {r: sum(v) / len(v) for r, v in sorted(per.items())}
    return sum(scores.values()) / len(scores), scores
