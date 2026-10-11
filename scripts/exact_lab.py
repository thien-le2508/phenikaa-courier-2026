"""Fast evaluator of linear route-cost hypotheses over enumerated route features (work/paths_<split>.pkl)."""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class Paths:
    def __init__(self, split, legged=False):
        d = pickle.load(open(ROOT / "work" / f"paths_{split}.pkl", "rb"))
        self.names = d["feat_names"]
        self.Y = d["Y"]
        self.ids = d["ids"]
        res = d["legged" if legged else "nonlegged"]
        mats, sc, mv = [], [], []
        for i, r in enumerate(res):
            for m, a in sorted(r.items()):
                mats.append(a)
                sc.append(np.full(len(a), i, np.int32))
                mv.append(np.full(len(a), m, np.int8))
        self.M = np.concatenate(mats).astype(np.float32)
        self.sc = np.concatenate(sc)
        self.mv = np.concatenate(mv)
        key = self.sc.astype(np.int64) * 4 + self.mv
        self.starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
        self.seg_sc = self.sc[self.starts]
        self.seg_mv = self.mv[self.starts]
        self.n = len(res)
        del d

    def fidx(self, name):
        return self.names.index(name)

    def qtable(self, w):
        """(n_scenes, 4) min cost per first move (inf if absent)."""
        c = self.M @ np.asarray(w, np.float32)
        mins = np.minimum.reduceat(c, self.starts)
        Q = np.full((self.n, 4), np.inf, np.float32)
        Q[self.seg_sc, self.seg_mv] = mins
        return Q

    def status(self, w, r, eps=2e-4):
        """per scene: 0 ok(unique), 1 tie(label in argmin set), 2 out; also argmin mask"""
        Q = self.qtable(w)
        b = Q.min(1, keepdims=True)
        A = Q <= b + eps
        y = self.Y[:, r]
        iny = A[np.arange(self.n), y]
        st = np.where(iny, np.where(A.sum(1) == 1, 0, 1), 2)
        return st, A, Q


def wvec(names, **kw):
    w = np.zeros(len(names), np.float32)
    w[0] = w[1] = w[2] = 10
    for k, v in kw.items():
        if k == "lm":
            for i, n in enumerate(names):
                if n.startswith("lm_") and n != "lm_tgtcopy":
                    w[i] += v
        elif k == "turn":
            for n in ("right", "left", "uturn"):
                w[names.index(n)] += v
        else:
            w[names.index(k)] = v
    return w


from itertools import permutations as _perms

PERMS = list(_perms(range(4)))
_CW = [0, 3, 1, 2]  # UP, RIGHT, DOWN, LEFT (codes U0 D1 L2 R3)


def rel_choice(A, heading, perm):
    """first legal-optimal move following relative order perm (0 S,1 R,2 U,3 L)."""
    n = len(A)
    hi = np.array([_CW.index(h) for h in heading])
    ch = np.full(n, -1)
    for k in reversed(perm):
        d = np.array(_CW)[(hi + k) % 4]
        ok = A[np.arange(n), d]
        ch = np.where(ok, d, ch)
    return ch


def strict_errors(P, w, r, mask, heading):
    """min over 24 relative tie orders of #errors on mask; returns (err, perm)."""
    st, A, Q = P.status(w, r)
    y = P.Y[:, r]
    best = None
    for p in PERMS:
        ch = rel_choice(A[mask], heading[mask], p)
        e = int((ch != y[mask]).sum())
        if best is None or e < best[0]:
            best = (e, p)
    return best


def qtable_groups(P, groups):
    """groups: list of (scene_mask, w). Returns Q for all scenes (inf for scenes not in any group)."""
    Q = np.full((P.n, 4), np.inf, np.float32)
    for mask, w in groups:
        rows = mask[P.sc]
        c = P.M[rows] @ np.asarray(w, np.float32)
        starts_mask = mask[P.seg_sc]
        st = np.r_[0, np.cumsum(np.diff(np.r_[P.starts, len(P.M)])[starts_mask])[:-1]]
        mins = np.minimum.reduceat(c, st) if len(c) else c
        Q[P.seg_sc[starts_mask], P.seg_mv[starts_mask]] = mins
    return Q


def status_Q(Q, y, eps=2e-4):
    b = Q.min(1, keepdims=True)
    A = Q <= b + eps
    iny = A[np.arange(len(Q)), y]
    st = np.where(iny, np.where(A.sum(1) == 1, 0, 1), 2)
    return st, A


def abs_choice(A, perm):
    n = len(A)
    ch = np.full(n, -1)
    for d in reversed(perm):
        ch = np.where(A[:, d], d, ch)
    return ch


def strict_errors2(P, w, r, mask, heading, sec=None):
    """min over 24 rel + 24 abs orders; sec: optional secondary weight vector (lexicographic eps)."""
    if sec is not None:
        w = np.asarray(w, np.float32) + 1e-3 * np.asarray(sec, np.float32)
    st, A, Q = P.status(w, r)
    y = P.Y[mask, r]
    Am, hm = A[mask], heading[mask]
    best = None
    for p in PERMS:
        e = int((rel_choice(Am, hm, p) != y).sum())
        if best is None or e < best[0]:
            best = (e, 'rel:' + ''.join('SRUL'[k] for k in p))
        e = int((abs_choice(Am, p) != y).sum())
        if e < best[0]:
            best = (e, 'abs:' + ''.join('UDLR'[k] for k in p))
    return best
