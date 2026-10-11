"""Huấn luyện bộ phân tích yêu cầu v3 (PhoBERT-base-v2 + 8 đầu) và chấm trên validation.

    # đo trung thực: train + dữ liệu sinh (không có cụm của val v3)
    python scripts/train_nlp.py --synth work/nlp/synth_holdout.json --real train --out work/nlp_v3_holdout.pt
    # bản nộp: train + val + dữ liệu sinh
    python scripts/train_nlp.py --synth work/nlp/synth_full.json --real train validation --no-eval --out work/nlp_v3_final.pt
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from courier.data import load_split  # noqa: E402
from courier.nlp.decode import decode  # noqa: E402
from courier.nlp.generate import add_noise  # noqa: E402
from courier.nlp.labels import nlp_target  # noqa: E402
from courier.nlp.metrics import score  # noqa: E402
from courier.nlp.model import HEADS, MissionParser, encode_labels, get_tokenizer, predict_probs  # noqa: E402


def real_missions(split):
    _, _, scenes = load_split(split)
    return [nlp_target(s["mission"]) for s in scenes]


def evaluate(model, tok, split="validation", log=print, max_len=192):
    _, _, scenes = load_split(split)
    texts = [s["mission"]["text"] for s in scenes]
    gold = [nlp_target(s["mission"]) for s in scenes]
    probs = predict_probs(model, tok, texts, max_len=max_len)
    preds = [decode(p, [l["type"] for l in s["landmarks"]]) for p, s in zip(probs, scenes)]
    acc, _ = score(preds, gold, scenes)
    log(f"[{split}] " + " ".join(f"{k}={v:.4f}" for k, v in acc.items()))
    return acc, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vinai/phobert-base-v2")
    ap.add_argument("--synth", nargs="*", default=[])
    ap.add_argument("--real", nargs="*", default=["train"])
    ap.add_argument("--real-repeat", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="chặn an toàn theo thời gian")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth", type=float, default=0.05)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    logf = open(Path(a.out).with_suffix(".log"), "w", encoding="utf-8")

    def log(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log("args", vars(a))
    synth = [m for f in a.synth for m in json.load(open(f, encoding="utf-8"))]
    real = [m for sp in a.real for m in real_missions(sp)]
    log(f"synthetic {len(synth)}  real {len(real)} x{a.real_repeat}")

    tok = get_tokenizer(a.encoder)
    model = MissionParser(a.encoder).cuda()
    log("params (M)", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))

    pool = [(m, False) for m in synth] + [(m, True) for m in real] * a.real_repeat
    steps_per_epoch = len(pool) // a.bs
    total = int(steps_per_epoch * a.epochs)
    if a.max_steps:
        total = min(total, a.max_steps)
    log("total steps", total)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr},
                             {"params": model.heads.parameters(), "lr": 1e-3}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, 1e-3], total_steps=total, pct_start=0.06)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss(label_smoothing=a.smooth)
    model.train()
    t0, step = time.time(), 0
    while step < total:
        rng.shuffle(pool)
        for i in range(0, len(pool) - a.bs + 1, a.bs):
            if step >= total or (a.max_minutes and time.time() - t0 > 60 * a.max_minutes):
                total = step
                break
            batch = pool[i:i + a.bs]
            texts = [add_noise(m["text"], rng, 0.1, 0.15, 0.05, 0.2, 0.05) if is_real else m["text"] for m, is_real in batch]
            y = [encode_labels(m) for m, _ in batch]
            b = tok(texts, padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(b["input_ids"].cuda(), b["attention_mask"].cuda())
            loss = sum(ce(out[h].float(), torch.tensor([t[h] for t in y], device="cuda")) for h in HEADS)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                log(f"step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s")
            if not a.no_eval and step % a.eval_every == 0 and step < total:
                evaluate(model, tok, log=log, max_len=a.max_len)
                model.train()
    torch.save({"encoder": a.encoder, "heads": {h: v for h, v in HEADS.items()},
                "state_dict": {k: v.half() for k, v in model.state_dict().items()}}, a.out)
    log("saved", a.out, f"{time.time() - t0:.0f}s")
    if not a.no_eval:
        acc, preds = evaluate(model, tok, log=log, max_len=a.max_len)
        json.dump(preds, open(Path(a.out).with_suffix(".valpred.json"), "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
