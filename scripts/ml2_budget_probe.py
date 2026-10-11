"""Probe length-budget (resource-constrained) hypotheses: max objective within L*+b edges."""
import sys, itertools
sys.path.insert(0, 'scripts')
import numpy as np, torch
import ml2_core as c
from ml2_inverse import order_tabs, ORDERS
r = int(sys.argv[1])
tr = c.cache('train')
night = np.array([s == 'night' for s in tr['style']])
P = c.pack(tr['scenes'])
legged = r == 4
ls = c.unit_len(P, legged)
tabs = order_tabs('cuda')
y = torch.as_tensor(tr['Y'][:, r], device='cuda')
def W(**kw):
    w = torch.zeros(c.F, dtype=torch.float64, device='cuda')
    for k, v in kw.items(): w[c.FI[k]] = v
    return w
hyps = {
 'cov_max': W(edge=1001, covered=-1000),
 'cov_max_lm': W(edge=1001, covered=-1000, lm=100000),
 'cov_max_cr': W(edge=1001, covered=-1000, crowded=10),
 'lm_cov': W(edge=1001, covered=-1000, lm=1e6, crowded=10),
 'crowd_min': W(edge=1, crowded=1000),
 'crowd_min_lm': W(edge=1, crowded=1000, lm=1e6),
 'lm_crowd': W(edge=1, crowded=1000, lm=1e6 * 0 + 30),
}
names = list(hyps)
Wm = torch.stack([hyps[n] for n in names])
for b in [0, 1, 2, 3, 4, 6, 99]:
    bud = torch.full((len(names),), b, device='cuda')
    Q = c.solve(P, Wm, legged=legged, budget=bud, lstar=ls)
    best = Q.min(-1, keepdim=True).values
    tie = Q <= best + 1e-6
    cover = tie.gather(2, y.view(1, -1, 1).expand(len(names), -1, 1)).squeeze(2).cpu().numpy()
    rank = tabs[:, P['head']]
    sc = torch.where(tie.unsqueeze(1), rank.unsqueeze(0).double(), torch.full((1,1,1,1), 99.0, device='cuda', dtype=torch.float64))
    hit = (sc.argmin(-1) == y.view(1, 1, -1)).cpu().numpy()  # [K,O,B]
    for k, n in enumerate(names):
        a_d = hit[k][:, ~night].mean(1).max(); a_n = hit[k][:, night].mean(1).max()
        print(f'b={b} {n:14s} cover day={cover[k][~night].mean():.3f} night={cover[k][night].mean():.3f} | acc day={a_d:.3f} night={a_n:.3f}')
