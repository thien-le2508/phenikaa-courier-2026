"""Huấn luyện CropNet (giao lộ, đoạn đường, dòng chú giải, thời tiết) và chấm trên validation (vị trí chuẩn).

    python scripts/train_crops.py --out work/crops.pt
    python scripts/train_crops.py --train train validation --out work/crops_final.pt --no-eval
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from courier.cv.datasets import CropDataset, collate_crops, split_items  # noqa: E402
from courier.cv.labels import IGNORE  # noqa: E402
from courier.cv.models import TASK_HEADS, CropNet, normalize  # noqa: E402


def run_batch(model, batch, ce=None):
    """Trả về (loss, {head: (pred, label)})."""
    loss, res = 0.0, {}
    for task, (x, y) in batch.items():
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(normalize(x.cuda(non_blocking=True)), task)
        y = y.cuda(non_blocking=True)
        for j, h in enumerate(TASK_HEADS[task]):
            if ce is not None:
                loss = loss + ce(out[h].float(), y[:, j])
            res[h] = (out[h].argmax(-1), y[:, j])
    return loss, res


@torch.no_grad()
def evaluate(model, items, log):
    model.eval()
    dl = DataLoader(CropDataset(items, train=False), batch_size=8, num_workers=6, collate_fn=collate_crops)
    ok, n = defaultdict(int), defaultdict(int)
    for batch in dl:
        _, res = run_batch(model, batch)
        for h, (p, y) in res.items():
            m = y != IGNORE
            ok[h] += int((p[m] == y[m]).sum())
            n[h] += int(m.sum())
    log("[val] " + " ".join(f"{h}={ok[h] / max(1, n[h]):.4f}({n[h] - ok[h]} sai)" for h in n))
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=["train"])
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--init", help="checkpoint to start from (fine-tune)")
    ap.add_argument("--aug", default=None, help='JSON degrade params, e.g. {"rot": 6, "blur": 2.0, "noise": 10}')
    ap.add_argument("--pct-start", type=float, default=0.05)
    a = ap.parse_args()
    logf = open(Path(a.out).with_suffix(".log"), "w", encoding="utf-8")

    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    items = [it for sp in a.train for it in split_items(sp)]
    val = split_items("validation")
    dl = DataLoader(CropDataset(items, train=True, aug=json.loads(a.aug) if a.aug else None), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    persistent_workers=True, drop_last=True, collate_fn=collate_crops)
    model = CropNet(pretrained=not a.init)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
    model = model.cuda()
    log("params (M)", round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=a.pct_start)
    scaler = torch.amp.GradScaler()
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE, label_smoothing=0.02)
    t0 = time.time()
    for ep in range(a.epochs):
        tot = 0.0
        for batch in dl:
            loss, _ = run_batch(model, batch, ce)
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
