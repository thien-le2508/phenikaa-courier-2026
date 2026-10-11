import sys, json
sys.path.insert(0, 'scripts')
import numpy as np, torch
import ml2_core as c, ml2_inverse as inv
import policy_lab as pl
r = int(sys.argv[1]); cell = tuple(x == '1' for x in sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
w = json.loads(sys.argv[4]) if len(sys.argv) > 4 else json.loads((c.ROOT / 'work/ml2/inverse_cell.json').read_text())[f'R{r}|{cell}']['w']
tr = c.cache('train'); L = pl.load('train')
idx = np.where((tr['conds'] == np.array(cell)).all(1))[0]
P = c.pack([tr['scenes'][i] for i in idx])
acc, cov, Q = inv.score(P, inv.wvec(w, 'cuda').unsqueeze(0), tr['Y'][idx, r], r == 4, inv.order_tabs('cuda'))
Q = Q[0].cpu().numpy(); shown = 0
print('w', w, 'cover', float(cov))
for j, i in enumerate(idx):
    y = tr['Y'][i, r]; q = Q[j]
    if q[y] <= q.min() + 1e-6: continue
    sc = L[i]
    sc.goals = tr['scenes'][i]['goals']; sc.vias = tr['scenes'][i]['vias']
    print(pl.describe(sc) if hasattr(sc, 'urgent') else sc.id, '| label', 'UDLR'[y], 'Q', {('UDLR'[d]): q[d] for d in range(4) if q[d] < 1e8})
    print(pl.render(sc)); shown += 1
    if shown >= n: break
