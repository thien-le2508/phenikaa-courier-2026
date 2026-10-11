"""CD search minimizing strict errors (best relative tie order) for robot R on scenes matching COND.
    python scripts/exact_strict.py R COND START_JSON [keys]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_lab import Paths, wvec, strict_errors2 as strict_errors, rel_choice, abs_choice  # noqa: E402
from policy_lab import load  # noqa: E402

import os
SKIP = set(os.environ.get('SKIP', '').split(','))
VALS = [0, 1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 21, 22, 25, 30, 35, 40, 50, 60, 80, 100, 200]


def run(r, cond, start, P, S, PV, SV, verbose=True):
    mask = np.array([bool(eval(cond, {'s': s})) for s in S])
    hd = np.array([s.heading for s in S])
    w = wvec(P.names, **start)
    best = strict_errors(P, w, r, mask, hd)
    if verbose:
        print('start', best, 'of', int(mask.sum()), flush=True)
    improved = True
    while improved:
        improved = False
        for k in [i for i, n in enumerate(P.names) if i > 0 and n not in SKIP]:
            for v in VALS:
                w2 = w.copy()
                w2[k] = v
                o = strict_errors(P, w2, r, mask, hd)
                if o[0] < best[0]:
                    best, w, improved = o, w2, True
                    if verbose:
                        print(P.names[k], v, best, flush=True)
    mv = np.array([bool(eval(cond, {'s': s})) for s in SV])
    hv = np.array([s.heading for s in SV])
    st, A, Q = PV.status(w, r)
    o = best[1]
    ch = rel_choice(A[mv], hv[mv], ['SRUL'.index(c) for c in o[4:]]) if o.startswith('rel') else abs_choice(A[mv], ['UDLR'.index(c) for c in o[4:]])
    verr = int((ch != PV.Y[mv, r]).sum())
    return w, best, (verr, int(mv.sum())), int(mask.sum())


if __name__ == '__main__':
    r = int(sys.argv[1])
    P = Paths('train', legged=(r == 4))
    PV = Paths('validation', legged=(r == 4))
    S, SV = load('train'), load('validation')
    for cond in sys.argv[2].split('||'):
        w, best, v, n = run(r, cond, json.loads(sys.argv[3]), P, S, PV, SV)
        print('COND', cond, 'train err', best[0], '/', n, 'order', best[1],
              'val err', v, json.dumps({k: float(x) for k, x in zip(P.names, w) if x}), flush=True)
