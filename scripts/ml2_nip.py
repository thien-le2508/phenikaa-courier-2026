"""Neural inverse planner (differentiable MMP / LEARCH-style) per robot.

cost(transition | scene group g) = softplus(W[g] . feat) + eps    (feat = ml2_core transition features)
Q[d] = cost(first step d) + Bellman-Ford distance-to-go       (hard min; gradient = path features)
logits = -Q / tau + first-move bias[g, rel(d)] * small
loss = cross-entropy(label). W[g] = W_shared + W_group[g] (L2-shrunk toward shared).
Groups = (rain, urgent, fragile, night) -> 16.

    python scripts/ml2_nip.py <robots> [epochs]
Outputs work/ml2/nip.json and nip_probs.pkl (OOF 5-fold on train + train-only on validation).
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml2_core as c  # noqa: E402
from ml2_inv2 import group_ids  # noqa: E402

OUT = c.ROOT / "work" / "ml2"
KEYS = ["rain", "urgent", "fragile", "night"]
G = 16
BIG = 1e4
REL = torch.tensor([[c.relative_turn(h, d) for d in range(4)] for h in range(4)])


def sub(P, idx):
    idx_t = torch.as_tensor(idx, device=P["succ"].device)
    Q = {k: (v[idx_t] if torch.is_tensor(v) and k != "book" else v) for k, v in P.items()}
    Q["B"] = len(idx)
    return Q


class NIP(torch.nn.Module):
    def __init__(self, legged):
        super().__init__()
        F = c.F
        w0 = torch.zeros(F)
        w0[c.FI["edge"]] = 0.54  # softplus(0.54) ~ 1 edge
        self.ws = torch.nn.Parameter(w0)
        self.wg = torch.nn.Parameter(torch.zeros(G, F))
        self.fb = torch.nn.Parameter(torch.zeros(G, 4))
        self.ab = torch.nn.Parameter(torch.zeros(G, 4))
        self.logtau = torch.nn.Parameter(torch.tensor(-1.0))
        self.legged = legged

    def weights(self):
        raw = self.ws.unsqueeze(0) + self.wg  # [G,F]
        e = torch.nn.functional.softplus(raw[:, c.FI["edge"]]) + 0.05  # edge cost > 0
        w = torch.nn.functional.softplus(raw)  # other costs >= 0
        cov = torch.tanh(raw[:, c.FI["covered"]]) * 0.9 * e  # covered may be negative but > -edge
        cols = [e if j == c.FI["edge"] else cov if j == c.FI["covered"] else w[:, j] for j in range(c.F)]
        return torch.stack(cols, 1)

    def forward(self, P, g, iters=200):
        book = P["book"].float()
        bad = book[:, c.F] > 0
        if not self.legged:
            bad = bad | (book[:, c.FI["stairs"]] > 0)
        W = self.weights()  # [G,F]
        B, S = P["B"], P["S"]
        U = book.shape[0]
        code = P["code"]
        term = P["term"]
        succ = P["succ"].view(B, S * 4)
        with torch.no_grad():
            cb = (W @ book[:, :c.F].T).masked_fill(bad.unsqueeze(0), BIG)  # [G,U]
            cost = cb.reshape(-1)[(g.view(B, 1, 1) * U + code).view(-1)].view(B, S, 4)
            dist = torch.full((B, S + 1), BIG, device=cost.device).masked_fill(term, 0.0)
            for it in range(iters):
                nxt = torch.gather(dist, 1, succ).view(B, S, 4)
                new = torch.cat([(cost + nxt).min(-1).values.clamp(max=BIG), dist[:, S:]], 1)
                new = new.masked_fill(term, 0.0)
                nd = torch.minimum(dist, new)
                if torch.equal(nd, dist):
                    break
                dist = nd
            nxt = torch.gather(dist, 1, succ).view(B, S, 4)
            act = (cost + nxt).argmin(-1)  # [B,S]
            nstate = torch.gather(P["succ"], 2, act.unsqueeze(-1)).squeeze(-1)  # [B,S]
            fa = book[torch.gather(code, 2, act.unsqueeze(-1)).squeeze(-1), :c.F]  # [B,S,F]
            live = (~term[:, :S]) & (dist[:, :S] < BIG / 2)
            fa = fa * live.unsqueeze(-1)
            phi = torch.zeros(B, S + 1, c.F, device=cost.device)
            for it in range(iters):
                nphi = torch.gather(phi, 1, nstate.unsqueeze(-1).expand(B, S, c.F))
                new = torch.cat([fa + nphi * live.unsqueeze(-1), phi[:, S:]], 1)
                if torch.equal(new, phi):
                    break
                phi = new
            st = P["start"]
            sn = torch.gather(P["succ"], 1, st.view(B, 1, 1).expand(B, 1, 4)).squeeze(1)  # [B,4]
            c0code = torch.gather(code, 1, st.view(B, 1, 1).expand(B, 1, 4)).squeeze(1)
            f0 = book[c0code, :c.F]  # [B,4,F]
            reach = (torch.gather(dist, 1, sn) < BIG / 2) & ~bad[c0code]
            pf = f0 + torch.gather(phi, 1, sn.unsqueeze(-1).expand(B, 4, c.F))  # [B,4,F]
        Q = (pf * W[g].unsqueeze(1)).sum(-1)  # differentiable, equals path cost
        rel = REL.to(Q.device)[P["head"]]
        bias = torch.gather(self.fb[g], 1, rel) + self.ab[g]
        logits = -Q / self.logtau.exp() + 0.1 * bias
        logits = logits.masked_fill(~reach, -1e9)
        return logits, Q


def path_grad_forward(model, P, g):
    return model(P, g)


def train_model(Ptr, ytr, gtr, legged, epochs=40, bs=250, lr=0.1, seed=0):
    torch.manual_seed(seed)
    dev = Ptr["succ"].device
    model = NIP(legged).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Ptr["B"]
    y = torch.as_tensor(ytr, device=dev)
    g = torch.as_tensor(gtr, device=dev)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            P = sub(Ptr, idx)
            logits, _ = model(P, g[idx])
            loss = torch.nn.functional.cross_entropy(logits, y[idx]) + 1e-3 * (model.wg ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, P, g, bs=500):
    out = []
    for i in range(0, P["B"], bs):
        idx = np.arange(i, min(P["B"], i + bs))
        logits, _ = model(sub(P, idx), torch.as_tensor(g[idx], device=P["succ"].device))
        out.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(out)


def main():
    robots = [int(x) for x in sys.argv[1].split(",")]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    dev = "cuda"
    tr, va = c.cache("train"), c.cache("validation")
    Ptr, Pva = c.pack(tr["scenes"], dev), c.pack(va["scenes"], dev)
    gtr, gva = group_ids(tr, KEYS), group_ids(va, KEYS)
    n = len(gtr)
    kf = list(KFold(5, shuffle=True, random_state=2026).split(np.arange(n)))
    rp, pp = OUT / "nip.json", OUT / "nip_probs.pkl"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    probs = pickle.loads(pp.read_bytes()) if pp.exists() else {}
    for r in robots:
        t0 = time.time()
        legged = r == 4
        yt, yv = tr["Y"][:, r], va["Y"][:, r]
        oof = np.zeros((n, 4))
        for a, b in kf:
            m = train_model(sub(Ptr, a), yt[a], gtr[a], legged, epochs)
            oof[b] = predict(m, sub(Ptr, b), gtr[b])
        m = train_model(Ptr, yt, gtr, legged, epochs)
        pv = predict(m, Pva, gva)
        res[str(r)] = dict(oof=float(np.mean(oof.argmax(1) == yt)), val=float(np.mean(pv.argmax(1) == yv)))
        probs[r] = dict(oof=oof, val=pv)
        torch.save(m.state_dict(), OUT / f"nip_R{r}.pt")
        print(f"nip R{r} OOF={res[str(r)]['oof']:.4f} val={res[str(r)]['val']:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        rp.write_text(json.dumps(res, indent=1))
        pp.write_bytes(pickle.dumps(probs))


if __name__ == "__main__":
    main()
