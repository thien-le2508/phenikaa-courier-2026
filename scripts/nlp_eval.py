"""Chấm bộ phân tích yêu cầu trên một split (mặc định validation), ràng buộc theo địa điểm thật trên bản đồ.

    python scripts/nlp_eval.py --ckpt work/nlp_v3_holdout.pt [--split validation] [--errors 30]

In: độ chính xác từng trường (goal theo nhãn NLP: None nếu chỉ mô tả qua bản đồ và câu không lộ loại),
khớp toàn bộ (exact) và "target" (giải tham chiếu trên bản đồ chuẩn: đúng tập đích + điểm ghé + gấp + dễ vỡ).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.data import load_split  # noqa: E402
from courier.nlp import infer  # noqa: E402
from courier.nlp.labels import nlp_target  # noqa: E402
from courier.nlp.metrics import FIELDS, field_eq, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--errors", type=int, default=20)
    ap.add_argument("--free", action="store_true", help="không ràng buộc theo bản đồ")
    ap.add_argument("--rules", action="store_true", help="bật luật phủ định (mặc định tắt: hại trên train)")
    a = ap.parse_args()
    _, _, scenes = load_split(a.split)
    reader = infer.load(a.ckpt)
    texts = [s["mission"]["text"] for s in scenes]
    lts = None if a.free else [[l["type"] for l in s["landmarks"]] for s in scenes]
    preds = infer.parse(reader, texts, lts, rules=a.rules)
    gold = [nlp_target(s["mission"]) for s in scenes]
    acc, ok = score(preds, gold, scenes)
    print(f"[{a.split}] n={len(scenes)} " + " ".join(f"{k}={v:.4f}" for k, v in acc.items()))
    shown = 0
    for p, g, o, t in zip(preds, gold, ok, texts):
        if o or shown >= a.errors:
            continue
        shown += 1
        bad = {f: (p[f], g[f]) for f in FIELDS if not field_eq(f, p, g)}
        print("-", t, "\n   pred/gold:", json.dumps(bad, ensure_ascii=False))


if __name__ == "__main__":
    main()
