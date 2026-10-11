import sys, json
sys.path.insert(0, 'scripts')
import numpy as np, torch
import ml2_core as c, ml2_inverse as inv
tr = c.cache('train')
S, _ = c.load_split('train')
r = int(sys.argv[1])
res = json.loads((c.ROOT / 'work/ml2/inverse_cell.json').read_text())
tabs = inv.order_tabs('cuda')
import itertools
rows = []
for cell in itertools.product([False, True], repeat=3):
    rr = res[f'R{r}|{cell}']
    idx = np.where((tr['conds'] == np.array(cell)).all(1))[0]
    P = c.pack([tr['scenes'][i] for i in idx])
    acc, cov, Q = inv.score(P, inv.wvec(rr['w'], 'cuda').unsqueeze(0), tr['Y'][idx, r], r == 4, tabs)
    Q = Q[0].cpu().numpy()
    for j, i in enumerate(idx):
        q = Q[j]; y = tr['Y'][i, r]
        ok = q[y] <= q.min() + 1e-6
        s = S[i]; ms = s['mission']
        ngoal = len(tr['scenes'][i]['goals']); nvia = len(tr['scenes'][i]['vias'])
        rows.append((ok, nvia > 0, ngoal > 1, nvia > 1, ms['goal_ref'] is not None, q[y] >= 1e8))
rows = np.array(rows)
names = ['has_via', 'multi_goal', 'multi_via', 'goal_ref', 'label_unreachable']
for k, n in enumerate(names):
    m = rows[:, k + 1]
    print(f'{n}: n={m.sum()} cover={rows[m,0].mean():.3f} | without n={(~m).sum()} cover={rows[~m,0].mean():.3f}')
