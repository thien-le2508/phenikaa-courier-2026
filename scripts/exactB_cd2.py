"""Coordinate descent with Dijkstra evaluator. args: robot cond split init_dict [keys] [lam]"""
import sys
from exactB_dj import *
if __name__ == "__main__":
    robot, cond, split = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    cur = eval(sys.argv[4])
    keys = sys.argv[5].split(",") if len(sys.argv) > 5 else ["crowded", "covered", "normal", "lm", "tR", "tL", "tU", "fR", "fL", "fU"] + ["lm_" + t for t in LMT]
    lam = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
    ev = Evaluator(split)
    D = ev.D
    from exactB_wsearch import cond_mask
    idxs = [i for i, ok in enumerate(cond_mask(D, cond)) if ok]
    vals = [0, 1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 500, 1000]
    def obj(w):
        a, u = stats(D, ev.qs(w, idxs), robot)
        return a + lam * u, a, u
    best = obj(cur)
    print("start", cur, best, flush=True)
    for it in range(4):
        imp = False
        for k in keys:
            ws = [dict(cur, **{k: v}) for v in vals]
            Qs = ev.many(ws, idxs)
            for w, Q in zip(ws, Qs):
                a, u = stats(D, Q, robot)
                if a + lam * u > best[0] + 1e-12:
                    best = (a + lam * u, a, u); cur = w; imp = True
            print(" ", k, "->", cur.get(k, 0), f"in={best[1]:.4f} uni={best[2]:.4f}", flush=True)
        print("iter", it, cur, best, flush=True)
        if not imp:
            break
