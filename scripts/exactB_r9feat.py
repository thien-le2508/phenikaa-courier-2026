"""Per-move local features for R9; save pickle."""
import pickle, sys, math
from exactB_common import *
from exactB_paths import bfs_lb


def feats(sc):
    m, goals, vias = sc["m"], sc["goals"], sc["vias"]
    vias_s = set(vias) if vias else set()
    s0 = 1 if (not vias or m.start in vias_s) else 0
    tg = list(vias) if s0 == 0 else list(goals)
    mi = sc["s"]["mission"]
    # ignore-ref targets
    gt = m.landmarks.get(mi["goal"], []) if mi["goal"] else goals
    vt = m.landmarks.get(mi["via"], []) if mi["via"] else []
    tg2 = list(vt) if (vt and m.start not in vt) else list(gt)
    lb = bfs_lb(m, goals, vias)
    avoid = m.lm_cells - set(goals) - vias_s
    lt = {rc: t for t, rc in m.lm_list}
    out = {}
    def man(v, T): return min(abs(v[0] - t[0]) + abs(v[1] - t[1]) for t in T) if T else 0
    def euc(v, T): return min((v[0] - t[0]) ** 2 + (v[1] - t[1]) ** 2 for t in T) if T else 0
    for d, v, status, st in m.adj[m.start]:
        s1 = 1 if (s0 or v in vias_s) else 0
        f = dict(
            man=man(v, tg), man0=man(m.start, tg), euc2=euc(v, tg),
            man2=man(v, tg2), euc22=euc(v, tg2), mang=man(v, goals), eucg=euc(v, goals),
            cheb=min(max(abs(v[0] - t[0]), abs(v[1] - t[1])) for t in tg) if tg else 0,
            bfs=lb.get((v, s1), 999),
            lm=int(v in avoid), lmt=lt.get(v) if v in avoid else None, lmany=int(v in m.lm_cells),
            crowded=int(status == "crowded"), covered=int(status == "covered"), stairs=int(st),
            rel=relative_turn(m.heading, d), d=d, deg=len(m.adj[v]), istarget=int(v in set(tg)),
            dr=v[0] - m.start[0], dc=v[1] - m.start[1],
        )
        out[d] = f
    return out


if __name__ == "__main__":
    for split in ("train", "validation"):
        D = load(split)
        pickle.dump([feats(sc) for sc in D], open(ROOT / f"work/exactB_r9feat_{split}.pkl", "wb"))
