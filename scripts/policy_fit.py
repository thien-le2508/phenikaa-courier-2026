"""Averaged structured perceptron on pooled (robot, condition) items. Analysis only.
python policy_fit.py SPEC epochs [w0 json]"""
import sys, json, random
from multiprocessing import Pool
sys.path.insert(0, '.')
sys.path.insert(0, r'C:\Users\LENOVO\AppData\Local\Temp\claude\z--hackathon\3adc1e89-ecee-4a85-af1d-35cadbcaff9d\scratchpad')
from policy_lab import *  # noqa
from fit import parse, F  # noqa
SC = None
ITEMS = None


def init(spec):
    global SC, ITEMS
    SC = load('train')
    ITEMS = [(i, r) for i, sc in enumerate(SC) for r, cs in parse(spec) if all(F[c](sc) for c in cs)]


def work(args):
    w, part = args
    out = []
    for k in part:
        i, r = ITEMS[k]
        q = first_move_feats(SC[i], w, legged=(r == 4))
        out.append((k, q))
    return out


if __name__ == '__main__':
    spec = sys.argv[1]
    epochs = int(sys.argv[2])
    nF = len(FEATS)
    w = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [10.0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [float(x) for x in w]
    init(spec)
    n = len(ITEMS)
    with Pool(15, initializer=init, initargs=(spec,)) as pool:
        best = None
        for ep in range(epochs):
            parts = [list(range(k, n, 60)) for k in range(60)]
            res = {}
            for part in pool.map(work, [(w, p) for p in parts]):
                for k, q in part:
                    res[k] = q
            viol = 0
            grad = [0.0] * nF
            for k, q in res.items():
                i, r = ITEMS[k]
                y = SC[i].y[r]
                if y not in q:
                    viol += 1
                    continue
                cy, fy = q[y]
                oth = [(c, f) for d, (c, f) in q.items() if d != y]
                if not oth:
                    continue
                co, fo = min(oth, key=lambda x: x[0])
                if cy > co + 1e-6:
                    viol += 1
                    for j in range(nF):
                        grad[j] += fy[j] - fo[j]
            print(ep, 'viol', viol, 'of', n, 'w', [round(x, 2) for x in w], flush=True)
            if best is None or viol < best[0]:
                best = (viol, list(w))
            if viol == 0:
                break
            step = 2.0 / (1 + ep * 0.3)
            norm = max(1e-9, max(abs(g) for g in grad))
            w = [w[j] - step * grad[j] / norm for j in range(nF)]
            w[0] = 10.0
            w = [max(0.0, x) for x in w]
    print('best', best)
