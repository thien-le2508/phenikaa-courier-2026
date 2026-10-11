# ML2 notes — learned policy fallback + inverse optimisation (2026-10-08)

Owner: ML-POLICY agent. Files: `courier/policy_ml2.py`, `scripts/ml2_*.py`, `work/ml2/`.
All numbers use the ground-truth graph and parsed mission from scenes.json. No test data was used.

## 1. Literature: ideas that apply here

- **Inverse shortest path / inverse optimisation** (Ahuja & Orlin 2001, "Inverse Optimization", Oper. Res. 49(5)): find
  edge costs so that the given paths are shortest. Under L1/L∞ this is an LP; for a fixed path it is solved with
  shortest-path or min-cost-flow duals. Here we only see the *first move*, so the demonstrated path is latent.
  The constraint "best path through the label move ≤ best path through every other move" is non-convex,
  so we use search or a latent perceptron in place of an LP.
- **Maximum Margin Planning** (Ratliff, Bagnell & Zinkevich, ICML 2006) and **LEARCH** (Ratliff, Silver & Bagnell
  2009): learn a linear or boosted cost map so that A*/Dijkstra reproduces demonstrations. The loss is a structured hinge,
  `w·Φ(demo) − min_path [w·Φ(path) − loss(path)]`. Its subgradient is `Φ(demo) − Φ(loss-augmented planner path)`, so each
  step needs only one call to the planner. With a latent demonstration (only the first move is known), use
  `Φ(best path through the label move)`. This is the same as a latent structured perceptron.
- **Structured perceptron** (Collins 2002) with Dijkstra as the decoder. It has the same update and is exact for
  separable data. With lexicographic costs (like R0) the separating weights have very different magnitudes, so
  integer coordinate search with accuracy as the objective is more practical (section 3).
- **Differentiable solvers / Fenchel–Young losses** (Vlastelica et al. 2020, "Differentiation of blackbox combinatorial
  solvers"; Berthet et al. 2020, "Learning with differentiable perturbed optimizers"): these give gradients through
  Dijkstra to learn cost maps. They are useful when costs come from rich features, such as images. Our cost features
  are tiny and discrete, so exact search is better.
- **Neural algorithmic reasoning** (Xu et al. 2020, "What can neural networks reason about?"; Veličković et al. CLRS):
  a message-passing GNN aligns with Bellman-Ford (min-aggregation is the relaxation step). The practical version used
  here is a GPU *batched Bellman-Ford* over (node, heading, via-stage) states with learnable linear edge costs.
  It is effectively a GNN with min-aggregation and hard-wired message functions. It evaluates about 20 weight
  hypotheses per second over all 2000 train scenes.
- **Learning to rank for action selection**: score each legal first move with features of the best routes behind
  it under many cost hypotheses (pointwise or listwise ranking). This is the current HGB ranker; listwise softmax over
  the ≤4 moves is the natural loss.

Sources: Ratliff et al. MMP (https://publications.ri.cmu.edu/maximum-margin-planning), LEARCH
(https://www.ri.cmu.edu/pub_files/2009/7/learch.pdf), Ahuja & Orlin inverse optimisation
(https://ideas.repec.org/a/inm/oropre/v49y2001i5p771-783.html), Xu et al. 2019 (https://arxiv.org/pdf/1905.13211).
