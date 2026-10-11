"""Search on pooled (robot, condition) items. Items given as list of (robot, cond-expr).
Primary weights fixed (json), then CD over secondary weights (scaled by 1e-3) minimizing strict errors.

    python scripts/exact_pool.py "1:s.s['style']=='night';6:s.urgent" '{"lm_dorm":20,"lm_canteen":20,"f_uturn":20}' [primary_cd]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_lab import Paths, wvec  # noqa: E402
from policy_lab import load, CW, rel  # noqa: E402
from itertools import permutations  # noqa: E402

VALS = [0, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]
SVALS = [-3, -2, -1, 0, 1, 2, 3]


class Pool_:
    def __init__(self, split, spec):
        self.P = {lg: Paths(split, legged=lg) for lg in (False, True) if any((r == 4) == lg for r, _ in spec)}
        S = load(split)
        self.S = S
        self.items = []
        for r, cond in spec:
            m = np.array([bool(eval(cond, {'s': s})) for s in S])
            self.items.append((r, m))

    def status(self, w):
        out = []
        for r, m in self.items:
            st, A, Q = self.P[r == 4].status(w, r)
            out.append((r, m, st, A))
        return out

    def counts(self, w):
        o = t = 0
        for r, m, st, A in self.status(w):
            o += int((st[m] == 2).sum())
            t += int((st[m] == 1).sum())
        return o, t


def parse(spec):
    out = []
    for part in spec.split(';'):
        r, c = part.split(':', 1)
        out.append((int(r), c))
    return out


def best_order(pool, w):
    """best relative / absolute fixed tie order on remaining ties"""
    res = []
    for kind in ('rel', 'abs'):
        for p in permutations(range(4)):
            ok = n = 0
            for r, m, st, A in pool.status(w):
                for i in np.nonzero(m & (st == 1))[0]:
                    h = pool.S[i].heading
                    order = [CW[(CW.index(h) + k) % 4] for k in p] if kind == 'rel' else list(p)
                    d = next(d for d in order if A[i, d])
                    ok += d == pool.S[i].y[r]
                    n += 1
            res.append((ok, n, kind + ''.join(('SRUL' if kind == 'rel' else 'UDLR')[k] for k in p)))
    res.sort(reverse=True)
    return res[:3]


if __name__ == '__main__':
    spec = parse(sys.argv[1])
    prim = json.loads(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else 'secondary'
    pool = Pool_('train', spec)
    names = next(iter(pool.P.values())).names
    w = wvec(names, **prim)
    print('primary', pool.counts(w), flush=True)
    if mode == 'primary':
        best = pool.counts(w)
        improved = True
        while improved:
            improved = False
            for k in range(1, len(w)):
                for v in VALS:
                    w2 = w.copy(); w2[k] = v
                    o = pool.counts(w2)
                    if o < best:
                        best, w, improved = o, w2, True
                        print(names[k], v, best, flush=True)
        print('final', json.dumps({n: float(x) for n, x in zip(names, w) if x}))
    sec = np.zeros(len(w), np.float32)
    best = sum(pool.counts(w))
    improved = True
    while improved:
        improved = False
        for k in range(len(w)):
            for v in SVALS:
                s2 = sec.copy(); s2[k] = v
                o = sum(pool.counts(w + 1e-3 * s2))
                if o < best:
                    best, sec, improved = o, s2, True
                    print('sec', names[k], v, pool.counts(w + 1e-3 * s2), flush=True)
    print('secondary', {n: float(x) for n, x in zip(names, sec) if x})
    wf = w + 1e-3 * sec
    print('final counts', pool.counts(wf), 'tie orders', best_order(pool, wf))
    pv = Pool_('validation', spec)
    print('validation counts', pv.counts(wf), best_order(pv, wf))
