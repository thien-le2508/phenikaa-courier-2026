"""Run the selected V2 end-to-end pipeline and write a submission JSON."""
import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import conditional_models as cm  # noqa: E402
import ml2_core as core  # noqa: E402
import ml2_nip as nip  # noqa: E402
from e2e_study_canonical import canonical_graph  # noqa: E402
from ml2_inv2 import group_ids  # noqa: E402
from courier.cv.reader import MapReader  # noqa: E402
from courier.data import DATA, load_split, scene_rows  # noqa: E402
from courier.nlp.infer import MissionReader  # noqa: E402
from courier.exact import r9 as r9_exact  # noqa: E402
from courier.policy import HEADING, Map, fallback_move, move_features, predict_all, targets  # noqa: E402
from courier.policy_features_v2 import move_features_v2  # noqa: E402
from courier.style import predict_style  # noqa: E402


def candidate_probs(model, rows):
    out = np.zeros(4, dtype=float)
    directions = sorted(rows)
    if directions:
        p = model.predict_proba(np.asarray([rows[d] for d in directions], np.float32))[:, 1]
        out[directions] = p
    return out


def normalise(p):
    s = float(np.sum(p))
    return p / s if s > 0 else p


def calibrate(p, temperature):
    legal = p > 0
    q = normalise(p)
    out = np.zeros(4, dtype=float)
    if legal.any():
        z = np.power(np.maximum(q[legal], 1e-7), 1.0 / temperature)
        out[legal] = z / z.sum()
    return out


def conditional9_probs(bundle, rows, graph, mission):
    spec = bundle["conditional9"]
    base = np.zeros((1, 4, spec["base_width"]), dtype=np.float64)
    legal = np.zeros((1, 4), dtype=bool)
    for direction, row in rows.items():
        base[0, direction] = np.asarray(row[:spec["base_width"]], dtype=np.float64)
        legal[0, direction] = True
    ctx = np.asarray([[float(graph.get("rain", False)), float(bool(mission.get("urgent"))),
                       float(bool(mission.get("fragile"))), float(bool(mission.get("via")))]])
    style = np.asarray([[graph.get("style") == name for name in cm.STYLES]], dtype=float)
    design, _ = cm.design(base, ctx, style, spec["variant"])
    raw = cm.utilities(spec["model"], spec["scaler"], spec["mult"],
                       design, legal, np.asarray([0]))
    return cm.softmax(raw, legal)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_private")
    ap.add_argument("--out", default="predictionsv2.json")
    ap.add_argument("--bundle", default="work/policy_v2_final.pkl")
    ap.add_argument("--heat", default="work/heat_v3.pt")
    ap.add_argument("--crops", default="work/crops_v3.pt,work/crops_v3b.pt")
    ap.add_argument("--nlp", default="work/nlp_v3_final.pt")
    ap.add_argument("--r9", choices=("exact", "ml"), default="exact",
                    help="exact = luật courier/exact/r9.py (100%% train/val); ml = mô hình conditional cũ")
    args = ap.parse_args()

    bundle = pickle.loads((ROOT / args.bundle).read_bytes())
    rows, _, _ = load_split(args.split)
    scenes = scene_rows(rows)
    reader = MapReader(args.heat, args.crops.split(","))
    mission_reader = MissionReader(args.nlp)
    graphs = []
    started = time.time()
    for i, (image, _) in enumerate(scenes):
        try:
            image_path = str(DATA / args.split / image)
            graph = reader.read(image_path)[0]
            graph["style"] = predict_style(bundle["style_model"], image_path)
            graphs.append(canonical_graph(graph))
        except Exception:
            traceback.print_exc()
            graphs.append(None)
        if (i + 1) % 100 == 0:
            print(f"read {i + 1}/{len(scenes)} images ({time.time() - started:.0f}s)", flush=True)

    texts = [text for _, text in scenes]
    landmarks = [[t for t, _ in g["landmarks"]] if g else None for g in graphs]
    missions = mission_reader.read(texts, landmarks, tta=False)
    # cache đồ thị + nhiệm vụ đã đọc để đổi chiến thuật mà không phải chạy lại CV/NLP
    (ROOT / "work" / f"cache_{args.split}.pkl").write_bytes(pickle.dumps({"graphs": graphs, "missions": missions}))

    valid_idx = [i for i, (g, m) in enumerate(zip(graphs, missions)) if g is not None and m is not None]
    built = [core.build_scene(graphs[i], missions[i]) for i in valid_idx]
    d = {
        "scenes": built,
        "conds": np.asarray([[bool(graphs[i].get("rain")), bool(missions[i].get("urgent")),
                               bool(missions[i].get("fragile"))] for i in valid_idx]),
        "has_via": np.asarray([bool(missions[i].get("via")) for i in valid_idx]),
        "style": [graphs[i].get("style", "classic") for i in valid_idx],
    }
    packed = core.pack(built, "cuda")
    groups = group_ids(d, nip.KEYS)
    nip_probs = {}
    for robot in (4, 5, 6):
        model = nip.NIP(robot == 4).cuda()
        model.load_state_dict(bundle["nip_states"][robot])
        nip_probs[robot] = nip.predict(model, packed, groups)
    del packed
    torch.cuda.empty_cache()

    valid_pos = {scene_idx: j for j, scene_idx in enumerate(valid_idx)}
    predictions = []
    for i, (graph, mission) in enumerate(zip(graphs, missions)):
        if graph is None or mission is None:
            heading = HEADING[graph["heading"]] if graph else 0
            predictions.extend([heading] * 10)
            continue
        try:
            base_exact = predict_all(graph, mission)
            all_v2 = move_features_v2(graph, mission)
            v1_rows = {r: {direction: row[:bundle["v1_dim"]] for direction, row in all_v2[r].items()}
                       for r in range(1, 10)}
            out = [base_exact[0]]
            pos = valid_pos[i]
            for robot in range(1, 10):
                if robot in (1, 2, 3, 8):
                    p = candidate_probs(bundle["v1_models"][robot], v1_rows[robot])
                elif robot == 4:
                    p = normalise(candidate_probs(bundle["v2_models"][robot], all_v2[robot])) + \
                        normalise(nip_probs[robot][pos])
                elif robot in (5, 6):
                    parts = {
                        "v1": candidate_probs(bundle["v1_models"][robot], v1_rows[robot]),
                        "v2": candidate_probs(bundle["v2_models"][robot], all_v2[robot]),
                        "nip": nip_probs[robot][pos],
                    }
                    p = sum(weight * calibrate(parts[name], bundle["temperatures"][robot][
                        "baseline_" + name if name in ("v1", "v2") else name])
                            for name, weight in bundle["blend_weights"][robot].items())
                elif robot == 7:
                    p = candidate_probs(bundle["v2_models"][robot], all_v2[robot])
                else:
                    p = conditional9_probs(bundle, v1_rows[robot], graph, mission)
                out.append(int(np.argmax(p)))
            if args.r9 == "exact":
                m = Map(graph)
                goals, vias = targets(m, mission)
                mv = r9_exact.move(m, mission, goals, vias, style=graph.get("style"))
                out[9] = mv if mv is not None else fallback_move(m)
            predictions.extend(out)
        except Exception:
            traceback.print_exc()
            predictions.extend([int(x) for x in predict_all(graph, mission)])
        if (i + 1) % 100 == 0:
            print(f"policy {i + 1}/{len(scenes)} ({time.time() - started:.0f}s)", flush=True)

    assert len(predictions) == len(rows)
    assert all(x in (0, 1, 2, 3) for x in predictions)
    path = ROOT / args.out
    path.write_text(json.dumps(predictions), encoding="utf-8")
    print(f"wrote {len(predictions)} predictions to {path} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
