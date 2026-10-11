"""Parallel inverse optimisation: independent integer coordinate searches for G condition groups
evaluated in one GPU batch (each scene uses its own group's weights).

    python scripts/ml2_inv2.py 1,2,3 rain,urgent,fragile,night tag [fold k/5]
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import ml2_core as c  # noqa: E402
from ml2_inverse import GRIDS, ORDERS, OUT, REL, nz, order_tabs, wvec  # noqa: E402

CORE = ["crowded", "covered", "lm", "t_right", "t_left", "t_uturn", "oneway", "f_right", "f_left", "f_uturn"]
EXTRA = ["lm_goaltype", "lm_sametype"] + ["lm_" + t for t in c.LM_TYPES]
CH = 6
INITS = ({}, {"lm": 10}, {"crowded": 500, "lm": 100}, {"t_right": 100, "t_left": 100, "t_uturn": 500})


def gscore(P, W, y, legged, tabs, group, G):
    """W [K,G,F+3]. Returns acc [K,G,O], cover [K,G], Q [K,B,4], pred-by-order [K,O,B]."""
    rel = torch.as_tensor(REL, device=W.device)[P["head"]]  # [B,4]
    Q = c.solve(P, W[..., :c.F].contiguous(), legged=legged, group=group)
    fw = torch.cat([torch.zeros_like(W[..., :1]), W[..., c.F:c.F + 3]], -1)  # [K,G,4] by rel
    add = torch.gather(fw[:, group], 2, rel.unsqueeze(0).expand(W.shape[0], -1, -1))
    Q = torch.where(Q < c.INF / 2, Q + add, Q)
    best = Q.min(-1, keepdim=True).values
    tie = Q <= best + 1e-6
    yy = torch.as_tensor(y, device=W.device)
    rank = tabs[:, P["head"]]  # [O,B,4]
    sc = torch.where(tie.unsqueeze(1), rank.unsqueeze(0).to(Q.dtype),
                     torch.full((1, 1, 1, 1), 99.0, device=W.device, dtype=Q.dtype))
    pred = sc.argmin(-1)  # [K,O,B]
    hit = (pred == yy.view(1, 1, -1)).to(Q.dtype)
    oh = torch.nn.functional.one_hot(group, G).to(Q.dtype)
    cnt = oh.sum(0).clamp(min=1)
    acc = (hit @ oh / cnt).permute(0, 2, 1)
    tieh = tie.gather(2, yy.view(1, -1, 1).expand(Q.shape[0], -1, 1)).squeeze(2).to(Q.dtype)
    return acc, tieh @ oh / cnt, Q, pred


def search(P, y, legged, feats, tabs, group, G, curs, sweeps=4):
    dev = P["succ"].device
    cur = [dict(d) for d in curs]
    for g in range(G):
        for f in feats:
            cur[g].setdefault(f, 0)
    acc, _, _, _ = gscore(P, torch.stack([wvec(d, dev) for d in cur]).unsqueeze(0), y, legged, tabs, group, G)
    best = acc[0].max(-1).values.cpu().numpy()
    for _ in range(sweeps):
        improved = False
        for f in feats:
            grid = GRIDS[f]
            W = torch.stack([torch.stack([wvec({**cur[g], f: v}, dev) for g in range(G)]) for v in grid])
            a = np.concatenate([gscore(P, W[i:i + CH], y, legged, tabs, group, G)[0].max(-1).values.cpu().numpy()
                                for i in range(0, len(grid), CH)])
            for g in range(G):
                k = int(np.argmax(a[:, g]))
                if a[k, g] > best[g] + 1e-9:
                    best[g] = a[k, g]
                    cur[g] = {**cur[g], f: grid[k]}
                    improved = True
        if not improved:
            break
    return cur


def finalize(P, y, legged, tabs, group, G, cur):
    dev = P["succ"].device
    acc, cov, Q, pred = gscore(P, torch.stack([wvec(d, dev) for d in cur]).unsqueeze(0), y, legged, tabs, group, G)
    o = acc[0].argmax(-1).cpu().numpy()
    return [float(acc[0, g, o[g]]) for g in range(G)], [float(cov[0, g]) for g in range(G)], o


def group_ids(d, keys):
    cols = {"rain": d["conds"][:, 0], "urgent": d["conds"][:, 1], "fragile": d["conds"][:, 2],
            "night": np.asarray([st == "night" for st in d["style"]]), "via": d["has_via"]}
    g = np.zeros(len(d["scenes"]), np.int64)
    for k in keys:
        g = g * 2 + cols[k].astype(np.int64)
    return g


def fit_robot(Ptr, ytr, gtr, G, r, tabs, extra=True):
    """Returns per-group weight dicts and tie orders fitted on (Ptr, ytr)."""
    dev = Ptr["succ"].device
    legged = r == 4
    feats = CORE + (["stairs"] if legged else [])
    gt = torch.as_tensor(gtr, device=dev)
    best = None
    for init in INITS:
        cur = search(Ptr, ytr, legged, feats, tabs, gt, G, [dict(init) for _ in range(G)])
        acc, cov, o = finalize(Ptr, ytr, legged, tabs, gt, G, cur)
        if best is None:
            best = [[cur[g], acc[g]] for g in range(G)]
        else:
            for g in range(G):
                if acc[g] > best[g][1]:
                    best[g] = [cur[g], acc[g]]
    cur = [b[0] for b in best]
    if extra:
        cur = search(Ptr, ytr, legged, feats + EXTRA, tabs, gt, G, cur, sweeps=3)
    acc, cov, o = finalize(Ptr, ytr, legged, tabs, gt, G, cur)
    return cur, acc, cov, o


def main():
    robots = [int(x) for x in sys.argv[1].split(",")]
    keys = [k for k in sys.argv[2].split(",") if k]
    tag = sys.argv[3]
    dev = "cuda"
    tabs = order_tabs(dev)
    tr, va = c.cache("train"), c.cache("validation")
    G = 2 ** len(keys)
    gtr, gva = group_ids(tr, keys), group_ids(va, keys)
    Ptr, Pva = c.pack(tr["scenes"], dev), c.pack(va["scenes"], dev)
    resp = OUT / f"inv2_{tag}.json"
    out = json.loads(resp.read_text()) if resp.exists() else {}
    for r in robots:
        t0 = time.time()
        cur, acc, cov, o = fit_robot(Ptr, tr["Y"][:, r], gtr, G, r, tabs)
        vacc, vcov, _, _ = gscore(Pva, torch.stack([wvec(d, dev) for d in cur]).unsqueeze(0), va["Y"][:, r],
                                  r == 4, tabs, torch.as_tensor(gva, device=dev), G)
        ntr, nva = np.bincount(gtr, minlength=G), np.bincount(gva, minlength=G)
        res = {}
        for g in range(G):
            res[format(g, f"0{max(1, len(keys))}b")] = dict(
                w=nz(cur[g]), order=ORDERS[int(o[g])], train_acc=round(acc[g], 4), train_cover=round(cov[g], 4),
                val_acc=round(float(vacc[0, g, o[g]]), 4), val_cover=round(float(vcov[0, g]), 4),
                ntr=int(ntr[g]), nva=int(nva[g]))
        rs = list(res.values())
        ttr = sum(x["train_acc"] * x["ntr"] for x in rs) / len(gtr)
        tva = sum(x["val_acc"] * x["nva"] for x in rs) / len(gva)
        out[f"R{r}"] = dict(keys=keys, train_acc=round(ttr, 4), val_acc=round(tva, 4), groups=res)
        print(f"R{r} keys={keys} train={ttr:.4f} val={tva:.4f} ({time.time() - t0:.0f}s)", flush=True)
        for g, x in res.items():
            print(f"   {g} {x}", flush=True)
        resp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
