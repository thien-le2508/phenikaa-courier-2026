"""Inverse optimisation by direct accuracy search (train only).

For each robot and each condition cell (rain, urgent, fragile) find integer edge-cost weights W
(edge weight fixed to E) and a first-move tie order such that argmin_d Q_W[d] reproduces labels.
Coordinate descent over a value grid, all candidate values evaluated in one GPU batch.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml2_core as c  # noqa: E402

ROOT = c.ROOT
OUT = ROOT / "work" / "ml2"
E = 1000.0
POS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000,
       5000, 10000, 30000, 100000]
COV = [-999, -900, -700, -500, -300, -200, -100, -70, -50, -30, -20, -10, -5, -3, -2, -1] + POS[:-3]
TURN = POS[:-4]
GRIDS = {"crowded": POS, "covered": COV, "lm": POS, "t_right": TURN, "t_left": TURN, "t_uturn": TURN,
         "oneway": POS[:-6], "stairs": POS, "lm_goaltype": POS[:-6], "lm_sametype": POS[:-6]}
for t in c.LM_TYPES:
    GRIDS["lm_" + t] = POS[:-6]
for t in ("f_right", "f_uturn", "f_left"):
    GRIDS[t] = TURN

ORDERS = ([("rel", o) for o in itertools.permutations(range(4))]
          + [("abs", o) for o in itertools.permutations(range(4))])


def order_tabs(dev):
    tabs = []
    for kind, o in ORDERS:
        tabs.append(c.rel_rank_table(o) if kind == "rel" else c.abs_rank_table(o))
    return torch.tensor(np.stack(tabs), device=dev)  # [O,4,4]


def score(P, W, y, legged, tabs):
    """Returns acc [K, O] (with tie orders) and cover [K] (label among minimisers)."""
    Q = c.solve(P, W[:, :c.F].contiguous(), legged=legged)
    # first-move-only turn costs (relative to robot heading): W[:, F:F+3] = f_right, f_uturn, f_left
    rel = torch.as_tensor(REL, device=W.device)[P["head"]]  # [B,4] relative turn of each dir
    fw = torch.cat([torch.zeros_like(W[:, :1]), W[:, c.F:c.F + 3]], 1)  # [K,4] by rel turn
    Q = torch.where(Q < c.INF / 2, Q + fw[:, rel], Q)
    best = Q.min(-1, keepdim=True).values
    tie = Q <= best + 1e-6  # [K,B,4]
    yy = torch.as_tensor(y, device=W.device)
    cover = tie.gather(2, yy.view(1, -1, 1).expand(Q.shape[0], -1, 1)).squeeze(2).float().mean(1)
    rank = tabs[:, P["head"]]  # [O,B,4]
    sc = torch.where(tie.unsqueeze(1), rank.unsqueeze(0).to(Q.dtype), torch.full((1, 1, 1, 1), 99.0,
                     device=W.device, dtype=Q.dtype))  # [K,O,B,4]
    pred = sc.argmin(-1)  # [K,O,B]
    acc = (pred == yy.view(1, 1, -1)).float().mean(-1)
    return acc, cover, Q


REL = [[c.relative_turn(h, d) for d in range(4)] for h in range(4)]
FIRST = {"f_right": 0, "f_uturn": 1, "f_left": 2}


def wvec(d, dev):
    w = torch.zeros(c.F + 3, dtype=torch.float64, device=dev)
    w[c.FI["edge"]] = E
    for k, v in d.items():
        w[c.F + FIRST[k] if k in FIRST else c.FI[k]] = v
    return w


def coord_search(P, y, legged, feats, tabs, init=None, sweeps=4, log=None):
    dev = P["succ"].device
    cur = dict(init or {})
    for f in feats:
        cur.setdefault(f, 0)
    acc, cov, _ = score(P, wvec(cur, dev).unsqueeze(0), y, legged, tabs)
    best_acc = float(acc.max())
    for sw in range(sweeps):
        improved = False
        for f in feats:
            vals = [v for v in GRIDS[f] if v != cur[f]]
            cands = []
            for v in vals:
                d = dict(cur)
                d[f] = v
                cands.append(d)
            W = torch.stack([wvec(d, dev) for d in cands])
            acc, cov, _ = score(P, W, y, legged, tabs)
            a = acc.max(1).values
            i = int(a.argmax())
            if float(a[i]) > best_acc + 1e-9:
                best_acc = float(a[i])
                cur = cands[i]
                improved = True
        if log:
            log(f"   sweep {sw} acc={best_acc:.4f} {nz(cur)}")
        if not improved:
            break
    W = wvec(cur, dev).unsqueeze(0)
    acc, cov, _ = score(P, W, y, legged, tabs)
    o = int(acc[0].argmax())
    return cur, float(acc[0, o]), float(cov[0]), o


def nz(d):
    return {k: v for k, v in d.items() if v}


def subset(scenes, idx):
    return [scenes[i] for i in idx]


def main():
    robots = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 9))
    mode = sys.argv[2] if len(sys.argv) > 2 else "cell"
    tr, va = c.cache("train"), c.cache("validation")
    dev = "cuda"
    tabs = order_tabs(dev)
    logf = open(OUT / f"inverse_{mode}.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    results = {}
    resp = OUT / f"inverse_{mode}.json"
    if resp.exists():
        results = json.loads(resp.read_text())
    core = ["crowded", "covered", "lm", "t_right", "t_left", "t_uturn", "oneway", "f_right", "f_left", "f_uturn"]
    extra = ["lm_goaltype", "lm_sametype"] + ["lm_" + t for t in c.LM_TYPES]
    for d in (tr, va):
        night = np.asarray([st == "night" for st in d["style"]])[:, None]
        d["conds4"] = np.concatenate([d["conds"], night], 1)
    if mode == "cell":
        cells = list(itertools.product([False, True], repeat=3))
    elif mode == "cellN":
        cells = list(itertools.product([False, True], repeat=4))
    elif mode == "night":
        cells = [("N", False), ("N", True)]
    else:
        cells = [None]
    for r in robots:
        legged = r == 4
        feats = core + (["stairs"] if legged else [])
        for cell in cells:
            if cell is None:
                itr = np.arange(len(tr["scenes"]))
                iva = np.arange(len(va["scenes"]))
            elif cell[0] == "N":
                itr = np.where(tr["conds4"][:, 3] == cell[1])[0]
                iva = np.where(va["conds4"][:, 3] == cell[1])[0]
            else:
                key = "conds" if len(cell) == 3 else "conds4"
                itr = np.where((tr[key] == np.array(cell)).all(1))[0]
                iva = np.where((va[key] == np.array(cell)).all(1))[0]
            Ptr = c.pack(subset(tr["scenes"], itr), dev)
            Pva = c.pack(subset(va["scenes"], iva), dev)
            ytr, yva = tr["Y"][itr, r], va["Y"][iva, r]
            t0 = time.time()
            log(f"R{r} cell={cell} ntr={len(itr)} nva={len(iva)}")
            best = None
            for init in ({}, {"lm": 10}, {"crowded": 500, "lm": 100}, {"t_right": 100, "t_left": 100, "t_uturn": 500}):
                cur, acc, cov, o = coord_search(Ptr, ytr, legged, feats, tabs, init=init, log=None)
                if best is None or acc > best[1]:
                    best = (cur, acc, cov, o)
            cur, acc, cov, o = coord_search(Ptr, ytr, legged, feats + extra, tabs, init=best[0], log=None)
            if acc < best[1]:
                cur, acc, cov, o = best
            vacc, vcov, _ = score(Pva, wvec(cur, dev).unsqueeze(0), yva, legged, tabs)
            res = dict(w=nz(cur), train_acc=acc, train_cover=cov, order=ORDERS[o],
                       val_acc=float(vacc[0, o]), val_cover=float(vcov[0]), ntr=len(itr), nva=len(iva))
            log(f"  -> {res}  ({time.time() - t0:.0f}s)")
            results[f"R{r}|{cell}"] = res
            resp.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
