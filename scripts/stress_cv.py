"""CV robustness: extra-degrade validation images and measure the fraction of scenes whose graph is read perfectly.

    python scripts/stress_cv.py --heat work/heat.pt work/heat_strong.pt --crops work/crops_v3.pt --levels clean blur2.0 rot5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from collections import Counter  # noqa: E402

from courier.cv.datasets import load_image  # noqa: E402
from courier.cv.reader import MapReader  # noqa: E402
from courier.data import DATA, load_split  # noqa: E402
from cv_eval import compare, gold_graph  # noqa: E402

LEVELS = {
    "clean": {}, "jpeg25": {"jpeg": 25}, "blur1.5": {"blur": 1.5}, "blur2.0": {"blur": 2.0}, "blur2.5": {"blur": 2.5},
    "rot3": {"rot": 3}, "rot5": {"rot": 5}, "rot7": {"rot": 7}, "scale0.6": {"scale": 0.6}, "noise16": {"noise": 16},
    "hard": {"rot": 4, "scale": 0.75, "blur": 1.2, "jpeg": 40},
    "harder": {"rot": 5, "scale": 0.65, "blur": 1.6, "jpeg": 30},
}


def stress(img, cfg, rng):
    h, w = img.shape[:2]
    if cfg.get("rot"):
        ang = cfg["rot"] * rng.choice([-1, 1]) * rng.uniform(0.7, 1.0)
        R = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        c, s = abs(R[0, 0]), abs(R[0, 1])
        nw, nh = int(w * c + h * s), int(w * s + h * c)
        R[0, 2] += nw / 2 - w / 2
        R[1, 2] += nh / 2 - h / 2
        bg = tuple(int(x) for x in np.median(np.r_[img[0], img[-1]], 0))
        img = cv2.warpAffine(img, R, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
        h, w = nh, nw
    if cfg.get("scale"):
        img = cv2.resize(img, (int(round(w * cfg["scale"])), int(round(h * cfg["scale"]))), interpolation=cv2.INTER_AREA)
    if cfg.get("blur"):
        img = cv2.GaussianBlur(img, (0, 0), cfg["blur"])
    if cfg.get("noise"):
        img = np.clip(img.astype(np.float32) + rng.normal(0, cfg["noise"], img.shape), 0, 255).astype(np.uint8)
    if cfg.get("jpeg"):
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpeg"])])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    return np.ascontiguousarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heat", nargs="+", required=True)
    ap.add_argument("--crops", nargs="+", default=["work/crops_v3.pt"], help="one CropNet or an ensemble")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--opts", default="{}")
    a = ap.parse_args()
    _, _, scenes = load_split(a.split)
    if a.n:
        scenes = scenes[:a.n]
    imgs = [load_image(str(DATA / a.split / s["image"])) for s in scenes]
    golds = [gold_graph(s) for s in scenes]
    crops = a.crops if len(a.crops) > 1 else a.crops[0]
    readers = {h: MapReader(h, crops, device=a.device, opts=json.loads(a.opts)) for h in a.heat}
    print("level".ljust(10) + "".join(Path(h).stem.ljust(34) for h in a.heat))
    for lv in a.levels:
        line = lv.ljust(10)
        for h, r in readers.items():
            ok, errs = 0, Counter()
            for i, s in enumerate(scenes):
                img = stress(imgs[i], LEVELS[lv], np.random.default_rng(1000 + i))
                res, e = compare(r.read_img(img)[0], golds[i])
                ok += res["graph"]
                errs.update(x.split()[0] for x in e)
            top = ",".join(f"{k}:{v}" for k, v in errs.most_common(3))
            line += f"{ok / len(scenes):.3f} {top}".ljust(34)
        print(line, flush=True)


if __name__ == "__main__":
    main()
