"""Evaluate a conditional linear route model on enumerated routes (fast).
Model file (python) defines: ROBOT, weights(c) -> dict for wvec, order(c) -> 'SRLU'-style string.
c = dict(rain, urgent, fragile, night, style, gtype, vtype, via, heading)

    python scripts/exact_eval_model.py model.py [show_n]
"""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_lab import Paths, wvec, qtable_groups, status_Q, rel_choice  # noqa: E402
from policy_lab import load, NAMES  # noqa: E402

_CACHE = {}
ABS = 'UDLR'


def choose(A, hd, od):
    if od.startswith('abs:'):
        n = len(A)
        ch = np.full(n, -1)
        for ch_ in reversed(od[4:]):
            d = ABS.index(ch_)
            ch = np.where(A[:, d], d, ch)
        return ch
    od = od[4:] if od.startswith('rel:') else od
    return rel_choice(A, hd, ['SRUL'.index(c) for c in od])


def best_orders(A, hd, y, idx):
    from itertools import permutations
    res = []
    for p in permutations('SRUL'):
        o = 'rel:' + ''.join(p)
        res.append((int((choose(A[idx], hd[idx], o) == y[idx]).sum()), o))
    for p in permutations('UDLR'):
        o = 'abs:' + ''.join(p)
        res.append((int((choose(A[idx], hd[idx], o) == y[idx]).sum()), o))
    res.sort(reverse=True)
    return res[:3]


def ctx(s):
    lt = {rc: t for t, rc in s.lms}
    return dict(rain=s.rain, urgent=s.urgent, fragile=s.fragile, night=s.s['style'] == 'night', style=s.s['style'],
                gtype=lt.get(s.goals[0]) if s.goals else s.goal_type, vtype=s.via_type, via=bool(s.vias),
                heading=s.heading)


def evaluate(mod, split):
    r = mod.ROBOT
    key = (split, r == 4)
    if key not in _CACHE:
        _CACHE[key] = (Paths(split, legged=(r == 4)), load(split))
    P, S = _CACHE[key]
    cs = [ctx(s) for s in S]
    groups = defaultdict(list)
    for i, c in enumerate(cs):
        groups[(repr(sorted(mod.weights(c).items())), mod.order(c))].append(i)
    Q = np.full((P.n, 4), np.inf, np.float32)
    pred = np.full(P.n, -1)
    for (wk, od), idx in groups.items():
        mask = np.zeros(P.n, bool)
        mask[idx] = True
        w = wvec(P.names, **dict(eval(wk)))
        Qg = qtable_groups(P, [(mask, w)])
        Q[mask] = Qg[mask]
    st, A = status_Q(Q, P.Y[:, r])
    hd = np.array([s.heading for s in S])
    for (wk, od), idx in groups.items():
        idx = np.array(idx)
        pred[idx] = choose(A[idx], hd[idx], od)
    err = pred != P.Y[:, r]
    return err, st, Q, S


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('m', sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    show = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = open(sys.argv[3], 'w', encoding='utf-8') if len(sys.argv) > 3 else sys.stdout
    print = lambda *a, **k: __builtins__.print(*a, file=out, flush=True)  # noqa: E731
    for split in ('train', 'validation'):
        err, st, Q, S = evaluate(mod, split)
        if hasattr(mod, 'ogroup'):
            P_, _ = _CACHE[(split, mod.ROBOT == 4)]
            b_ = Q.min(1, keepdims=True); A_ = Q <= b_ + 2e-4
            hd_ = np.array([s.heading for s in S])
            keys = [mod.ogroup(ctx(s)) for s in S]
            for k in sorted(set(keys), key=str):
                idx = np.array([i for i, kk in enumerate(keys) if kk == k])
                print('  order group', k, len(idx), best_orders(A_, hd_, P_.Y[:, mod.ROBOT], idx))
        print(f'{split}: acc {1 - err.mean():.4f}  errors {err.sum()}  outs {(st == 2).sum()}  ties-wrong {(err & (st == 1)).sum()}',
              )
        bad = np.nonzero(err)[0]
        for i in bad[:show]:
            s = S[i]
            q = {NAMES[d]: round(float(Q[i, d]), 3) for d in range(4) if np.isfinite(Q[i, d])}
            print('  ', s.id, 'st', st[i], 'label', NAMES[s.y[mod.ROBOT]], q, ctx(s)['style'], 'rain' if s.rain else '',
                  'urg' if s.urgent else '', 'frag' if s.fragile else '', ctx(s)['gtype'])
