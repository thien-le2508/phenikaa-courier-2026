"""Huấn luyện HeatmapNet (tâm giao lộ, ô mẫu chú giải, ô thời tiết) và chấm trên validation.

    python scripts/train_heatmap.py --out work/heat.pt
    python scripts/train_heatmap.py --train train validation --out work/heat_final.pt --no-eval
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from courier.cv import geometry as G  # noqa: E402
from courier.cv.datasets import HEAT_SIZE, HeatmapDataset, split_items  # noqa: E402
from courier.cv.labels import scene_targets  # noqa: E402
from courier.cv.models import HeatmapNet, focal_loss, normalize  # noqa: E402
from courier.cv.peaks import find_peaks  # noqa: E402


def match(pred, gold, tol):
    if len(pred) == 0 or len(gold) == 0:
        return 0
    d, _ = cKDTree(pred).query(gold)
    return int((d < tol).sum())


@torch.no_grad()
def evaluate(model, items, log):
    model.eval()
    ds = HeatmapDataset(items, train=False)
    tp = fp = fn = 0
    sw_ok = w_ok = rc_ok = 0
    errs = []
    for i in range(len(ds)):
        img, _ = ds[i]
        s = items[i][1]
        scale = HEAT_SIZE / max(s["width"], s["height"])
        with torch.autocast("cuda", dtype=torch.float16):
            prob = torch.sigmoid(model(normalize(img[None].cuda()))[0].float())
        t = scene_targets(s)
        gold = np.array([p for p, _, _ in t["nodes"]])
        sp = G.spacing(gold)
        nodes = find_peaks(prob[0], scale)
        m = match(nodes[:, :2], gold, 0.25 * sp)
        tp += m
        fn += len(gold) - m
        fp += len(nodes) - m
        if len(nodes) == len(gold) == m:
            rcs, _ = G.assign_rc(nodes[:, :2])
            _, idx = cKDTree(nodes[:, :2]).query(gold)
            pred_rc = [rcs[j] for j in idx]
            rc_ok += pred_rc == [rc for _, rc, _ in t["nodes"]]
        else:
            errs.append((i, len(gold) - m, len(nodes) - m))
        sws = find_peaks(prob[1], scale)
        gs = np.array([p for p, _, _ in t["legend"]])
        sw_ok += match(sws[:, :2], gs, 0.02 * s["width"]) == len(gs) == len(sws)
        wp = find_peaks(prob[2], scale, thr=0.05)
        if len(wp):
            best = wp[np.argmax(wp[:, 2]), :2]
            w_ok += np.linalg.norm(best - t["weather"][0]) < 0.02 * s["width"]
    n = len(items)
    log(f"[val] nodes P={tp / max(1, tp + fp):.4f} R={tp / max(1, tp + fn):.4f} | scenes rc exact={rc_ok / n:.4f}"
        f" | legend rows all found={sw_ok / n:.4f} | weather found={w_ok / n:.4f}")
    log("   node errors (scene, missed, extra):", errs[:15])
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=["train"])
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--init", help="checkpoint to start from (fine-tune)")
    ap.add_argument("--aug", default=None, help='JSON degrade params, e.g. {"rot": 6, "blur": 2.0, "noise": 10}')
    ap.add_argument("--pct-start", type=float, default=0.05)
    ap.add_argument("--strong", action="store_true", help="làm xấu ảnh mạnh hơn: xoay ±7°, mờ tới 2.5, nhiễu hạt")
    a = ap.parse_args()
    aug = {"rot": 7.0, "blur": 2.5, "noise": 15.0} if a.strong else {}
    if a.aug:
        aug = json.loads(a.aug)
    logf = open(Path(a.out).with_suffix(".log"), "w", encoding="utf-8")

    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    items = [it for sp in a.train for it in split_items(sp)]
    val = split_items("validation")
    dl = DataLoader(HeatmapDataset(items, train=True, aug=aug), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    persistent_workers=True, drop_last=True)
    model = HeatmapNet(pretrained=not a.init)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
    model = model.cuda()
    log("params (M)", round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=a.pct_start)
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    for ep in range(a.epochs):
        tot = 0.0
        for img, hm in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(normalize(img.cuda(non_blocking=True)))
            hm = hm.cuda(non_blocking=True)
            loss = sum(focal_loss(out[:, c], hm[:, c]) for c in range(3))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        log(f"epoch {ep} loss {tot / len(dl):.4f} {time.time() - t0:.0f}s")
        torch.save(model.state_dict(), a.out)
        if not a.no_eval and (ep % 4 == 3 or ep == a.epochs - 1):
            evaluate(model, val, log)
    log("saved", a.out)


if __name__ == "__main__":
    main()
