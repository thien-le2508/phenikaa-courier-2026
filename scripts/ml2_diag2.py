import sys, json, itertools, collections
sys.path.insert(0, 'scripts')
import numpy as np, torch
import ml2_core as c, ml2_inverse as inv
tr = c.cache('train'); S, _ = c.load_split('train')
r = int(sys.argv[1])
res = json.loads((c.ROOT / sys.argv[2]).read_text())
tabs = inv.order_tabs('cuda')
ok = np.zeros(2000, bool)
for cell in itertools.product([False, True], repeat=3):
    rr = res[f'R{r}|{cell}']
    idx = np.where((tr['conds'] == np.array(cell)).all(1))[0]
    P = c.pack([tr['scenes'][i] for i in idx])
    acc, cov, Q = inv.score(P, inv.wvec(rr['w'], 'cuda').unsqueeze(0), tr['Y'][idx, r], r == 4, tabs)
    Q = Q[0].cpu().numpy()
    for j, i in enumerate(idx):
        ok[i] = Q[j, tr['Y'][i, r]] <= Q[j].min() + 1e-6
def rep(name, keys):
    g = collections.defaultdict(list)
    for k, o in zip(keys, ok): g[k].append(o)
    print(name, {k: (len(v), round(float(np.mean(v)), 3)) for k, v in sorted(g.items(), key=lambda t: str(t[0]))})
rep('style', [s['style'] for s in S])
rep('swap', [tuple(sorted((k, v) for k, v in s['road_look'].items() if k != v)) for s in S])
rep('rows', [s['grid']['rows'] for s in S])
rep('heading', [s['robot']['heading'] for s in S])
rep('nlm', [len(s['landmarks']) for s in S])
rep('goal', [s['mission']['goal'] for s in S])
