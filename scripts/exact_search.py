"""Coordinate-descent search of linear route weights for a (robot, scene-subset) on enumerated routes.

    python scripts/exact_search.py R COND [start-json]
COND: python expression over scene `s` (policy_lab.Scene), e.g. "s.s['style']!='night'".
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_lab import Paths, wvec  # noqa: E402
from policy_lab import load  # noqa: E402

VALS = [0, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150]


def search(P, r, mask, w, keys=None, vals=VALS, verbose=True):
    keys = keys if keys is not None else list(range(len(w)))

    def obj(w):
        st = P.status(w, r)[0][mask]
        return (int((st == 2).sum()), int((st == 1).sum()))
    best = obj(w)
    if verbose:
        print('start', best, flush=True)
    improved = True
    while improved:
        improved = False
        for k in keys:
            if k == 0:
                continue
            for v in vals:
                w2 = w.copy()
                w2[k] = v
                o = obj(w2)
                if o < best:
                    best, w, improved = o, w2, True
                    if verbose:
                        print(P.names[k], v, best, flush=True)
    return w, best


if __name__ == '__main__':
    r = int(sys.argv[1])
    cond = sys.argv[2]
    P = Paths('train', legged=(r == 4))
    S = load('train')
    mask = np.array([bool(eval(cond, {'s': s})) for s in S])
    start = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    w = wvec(P.names, **start)
    w, best = search(P, r, mask, w)
    print('final', {n: float(x) for n, x in zip(P.names, w) if x}, best, 'of', int(mask.sum()))
    PV = Paths('validation', legged=(r == 4))
    SV = load('validation')
    mv = np.array([bool(eval(cond, {'s': s})) for s in SV])
    st = PV.status(w, r)[0][mv]
    print('validation', np.bincount(st, minlength=3), 'of', int(mv.sum()))
