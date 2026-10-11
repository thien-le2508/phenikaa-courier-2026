"""ML2 candidate features: per legal first move, route-cost summaries under a bank of cost hypotheses
(evaluated by the batched Bellman-Ford solver in ml2_core) plus local/scene context.

Usage: python scripts/ml2_feats.py   -> work/ml2/feats_{train,validation}.pkl
"""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml2_core as c  # noqa: E402

E = 1000.0
REL = np.array([[c.relative_turn(h, d) for d in range(4)] for h in range(4)])
FIRST = {"f_right": 1, "f_uturn": 2, "f_left": 3}


def make_bank():
    """List of (name, weight dict). Units: one edge = 1000."""
    bank = []
    # generic grid over the main factors
    for cr, lm, cv, turn in itertools.product([0, 2000, 6000], [0, 500, 3000, 100000], [0, -400, 1000],
                                              [None, "soft", "hard"]):
        w = {"crowded": cr, "lm": lm, "covered": cv}
        if turn == "soft":
            w.update(t_right=100, t_left=100, t_uturn=300)
        elif turn == "hard":
            w.update(t_right=300, t_left=300, t_uturn=3000)
        bank.append(w)
    # lexicographic R0-style (edges >> landmarks >> statuses)
    bank.append({"edge": 1e6, "lm": 1e4, "crowded": 100})
    bank.append({"edge": 1e6 + 100, "lm": 1e4, "covered": -100, "crowded": 1})
    # turn-averse variants
    for tr, tl, tu in [(3000, 3000, 30000), (500, 4500, 9000), (4500, 500, 9000), (1000, 1000, 3000)]:
        bank.append({"t_right": tr, "t_left": tl, "t_uturn": tu})
        bank.append({"t_right": tr, "t_left": tl, "t_uturn": tu, "lm": 3000, "crowded": 3000})
    # per-type landmark aversion
    for t in c.LM_TYPES:
        bank.append({"lm": 500, "lm_" + t: 5000})
    bank.append({"lm_goaltype": 5000, "lm": 500})
    bank.append({"oneway": 2000})
    bank.append({"f_uturn": 2000, "t_uturn": 3000, "lm": 1000})
    bank.append({"stairs": 3000})
    bank.append({"stairs": 3000, "crowded": 3000, "lm": 3000})
    return bank


def wvec(d):
    w = np.zeros(c.F + 4)
    w[c.FI["edge"]] = E
    for k, v in d.items():
        if k in FIRST:
            w[c.F + FIRST[k]] = v
        else:
            w[c.FI[k]] = v
    return w


def bank_Q(P, bank, legged, chunk=8, dev="cuda"):
    """Q [H, B, 4] (np.float64; inf for illegal)."""
    W = torch.tensor(np.stack([wvec(d) for d in bank]), dtype=torch.float64, device=dev)
    out = []
    head = P["head"].cpu().numpy()
    rel = REL[head]  # [B,4]
    for i in range(0, len(bank), chunk):
        Q = c.solve(P, W[i:i + chunk, :c.F].contiguous(), legged=legged).cpu().numpy()
        fw = W[i:i + chunk, c.F:].cpu().numpy()  # [k,4] by rel
        Q = np.where(Q < c.INF / 2, Q + fw[:, rel], np.inf)
        out.append(Q)
    return np.concatenate(out)


def move_matrix(Q, scale=E):
    """Q [H,B,4] -> features [B,4,3H]: relative cost (edges), is-min, number of minimisers."""
    best = np.min(Q, axis=2, keepdims=True)
    rel = np.where(np.isfinite(Q), np.minimum((Q - best) / scale, 30.0), 30.0)
    ismin = (Q <= best + 1e-6) & np.isfinite(Q)
    nmin = np.broadcast_to(ismin.sum(2, keepdims=True), Q.shape).astype(float)
    f = np.stack([rel, ismin.astype(float), nmin], -1)  # [H,B,4,3]
    return f.transpose(1, 2, 0, 3).reshape(Q.shape[1], 4, -1)


STYLES = ["classic", "night", "print", "sketch"]


def context(d):
    """Scene/move context [B,4,k]."""
    B = len(d["scenes"])
    out = np.zeros((B, 4, 27))
    for i, o in enumerate(d["scenes"]):
        h = o["heading"]
        rain, urg, frag = d["conds"][i]
        sc = [float(rain), float(urg), float(frag), float(bool(o["vias"]))] + \
             [float(d["style"][i] == s) for s in STYLES]
        for dd in range(4):
            r = REL[h, dd]
            out[i, dd] = sc + [float(r == k) for k in range(4)] + [float(dd == k) for k in range(4)] + \
                [float(len(o["goals"]) > 1), float(len(o["vias"]) > 1), float(o["N"]) / 81, 0.0, 0.0] + \
                [float(h == k) for k in range(4)] + [0.0, 0.0]
    return out


def legal_mask(P, legged):
    """[B,4] legal first moves."""
    B = P["B"]
    book = P["book"].cpu().numpy()
    code = P["code"].cpu().numpy()
    st = P["start"].cpu().numpy()
    c0 = code[np.arange(B), st]  # [B,4]
    ok = book[c0, c.F] == 0
    if not legged:
        ok &= book[c0, c.FI["stairs"]] == 0
    return ok


def first_step(P):
    """Features of the first edge itself [B,4,k]: status one-hot, stairs, oneway, lm-step."""
    B = P["B"]
    book = P["book"].cpu().numpy().astype(float)
    code = P["code"].cpu().numpy()
    st = P["start"].cpu().numpy()
    c0 = code[np.arange(B), st]
    cols = [c.FI[k] for k in ("crowded", "covered", "stairs", "oneway", "lm", "lm_goaltype")]
    return book[c0][..., cols]


def build(split, bank, dev="cuda"):
    d = c.cache(split)
    P = c.pack(d["scenes"], dev)
    ctx = context(d)
    fs = first_step(P)
    res = {}
    for legged in (False, True):
        Q = bank_Q(P, bank, legged, dev=dev)
        X = np.concatenate([move_matrix(Q), ctx, fs], -1).astype(np.float32)
        res[legged] = dict(X=X, legal=legal_mask(P, legged))
    return dict(res=res, Y=d["Y"], conds=d["conds"], style=d["style"], has_via=d["has_via"])


def main():
    bank = make_bank()
    print("bank size", len(bank), flush=True)
    for split in ("train", "validation"):
        out = build(split, bank)
        out["bank"] = bank
        (c.ROOT / "work" / "ml2" / f"feats_{split}.pkl").write_bytes(pickle.dumps(out))
        print(split, out["res"][False]["X"].shape, flush=True)


if __name__ == "__main__":
    main()
