"""Phân tích lỗi trên validation theo kiểu vẽ và theo robot, tách nguồn lỗi CV / NLP.

    python scripts/error_breakdown.py --heat work/heat.pt --crops work/crops.pt --nlp work/nlp_holdout.pt
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.cv.reader import MapReader  # noqa: E402
from courier.data import DATA, load_split  # noqa: E402
from courier.nlp.infer import MissionReader  # noqa: E402
from courier.policy import graph_from_scene, mission_from_scene, predict_all  # noqa: E402

SETTINGS = {"CV": "ảnh dự đoán + câu chuẩn", "NLP": "ảnh chuẩn + câu dự đoán", "ALL": "cả hai dự đoán"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heat", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--nlp", required=True)
    a = ap.parse_args()
    rows, labels, scenes = load_split("validation")
    reader = MapReader(a.heat, a.crops)
    graphs = [reader.read(str(DATA / "validation" / s["image"]))[0] for s in scenes]
    mr = MissionReader(a.nlp)
    texts = [s["mission"]["text"] for s in scenes]
    m_pred = mr.read(texts, [[t for t, _ in g["landmarks"]] for g in graphs])
    m_pred_gt = mr.read(texts, [[l["type"] for l in s["landmarks"]] for s in scenes])

    wrong = {k: defaultdict(int) for k in SETTINGS}   # (style, robot) -> số dòng sai
    scenes_bad = {k: defaultdict(int) for k in SETTINGS}
    n_style = defaultdict(int)
    for i, s in enumerate(scenes):
        st = s["style"]
        n_style[st] += 1
        y = labels[10 * i:10 * i + 10]
        gg, gm = graph_from_scene(s), mission_from_scene(s)
        preds = {"CV": predict_all(graphs[i], gm), "NLP": predict_all(gg, m_pred_gt[i]),
                 "ALL": predict_all(graphs[i], m_pred[i])}
        for k, p in preds.items():
            bad = [r for r in range(10) if p[r] != y[r]]
            for r in bad:
                wrong[k][(st, r)] += 1
            scenes_bad[k][st] += bool(bad)

    styles = sorted(n_style)
    for k, desc in SETTINGS.items():
        print(f"\n=== {k}: {desc} ===")
        print("theo kiểu vẽ: " + " | ".join(
            f"{st} ({n_style[st]} cảnh): acc={1 - sum(wrong[k][(st, r)] for r in range(10)) / (10 * n_style[st]):.4f},"
            f" {scenes_bad[k][st]} cảnh có lỗi" for st in styles))
        n = len(scenes)
        print("theo robot:   " + " ".join(
            f"R{r}={1 - sum(wrong[k][(st, r)] for st in styles) / n:.3f}" for r in range(10)))
        print("số dòng sai (kiểu vẽ x robot):")
        print("          " + "".join(f"R{r:<4}" for r in range(10)))
        for st in styles:
            print(f"  {st:<8}" + "".join(f"{wrong[k][(st, r)]:<5}" for r in range(10)))


if __name__ == "__main__":
    main()
