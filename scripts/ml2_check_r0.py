import sys, time
sys.path.insert(0, 'scripts')
import numpy as np, torch
import ml2_core as c
d = c.cache('train')
P = c.pack(d['scenes'])
rain, urg = d['conds'][:, 0], d['conds'][:, 1]
def w_r0(rain):
    w = np.zeros(c.F); w[c.FI['edge']] = 1e6; w[c.FI['lm']] = 1e4
    if rain:
        w[c.FI['edge']] += 100; w[c.FI['covered']] = -100; w[c.FI['crowded']] = 1
    else:
        w[c.FI['crowded']] = 100
    return w
W = torch.tensor(np.stack([w_r0(False), w_r0(True)]), device='cuda', dtype=torch.float64)
t = time.time(); Q = c.solve(P, W); torch.cuda.synchronize(); print('solve', time.time() - t)
SRLU, RSLU, LSRU = (0, 1, 3, 2), (1, 0, 3, 2), (3, 0, 1, 2)
tabs = {k: torch.tensor(c.rel_rank_table(o), device='cuda') for k, o in [('s', SRLU), ('r', RSLU), ('l', LSRU)]}
pred = np.zeros(2000, int)
for wi in (0, 1):
    for k in 'srl':
        p = c.pick(Q[wi:wi+1], P['head'], tabs[k])[0].cpu().numpy()
        sel = (rain == bool(wi)) & ((~urg & (k == 's')) | (urg & ~rain & (k == 'r')) | (urg & rain & (k == 'l')))
        pred[sel] = p[sel]
print('R0 acc', (pred == d['Y'][:, 0]).mean())
for K in (8, 32):
    W = torch.tensor(np.random.randint(0, 50, (K, c.F)), device='cuda', dtype=torch.float32)
    W[:, 0] = 100
    t = time.time(); Q = c.solve(P, W); torch.cuda.synchronize(); print('K', K, time.time() - t)
print('U', P['book'].shape)
