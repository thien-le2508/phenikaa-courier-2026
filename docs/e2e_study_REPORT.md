# Oracle-policy vs end-to-end study (2026-10-09)

## Scope and leakage controls

- Read only `delivery_public/train` and `delivery_public/validation`. No
  `test_private` file was opened.
- CV was evaluated from images, not from `scenes.json`; `scenes.json` was used
  only after inference to score the result.
- NLP uses `nlp_v3_holdout.pt`, which was not trained on validation text.
- Policy rankers and the style classifier were refit on **train only**. The
  production `policy_best.pkl` was deliberately not used because it was refit
  on train+validation and would make this decomposition optimistic.
- No production/final pipeline file was modified.

## Main result

| Input to the same train-only policy | Macro accuracy | Decisions changed vs oracle | Hurt / helped |
|---|---:|---:|---:|
| Oracle graph + oracle mission + oracle style | 90.433% | 0 | 0 / 0 |
| Predicted CV only | 90.467% | 1 | 0 / 1 |
| Predicted NLP only | 90.433% | 0 | 0 / 0 |
| Predicted style only | 90.433% | 0 | 0 / 0 |
| Full end-to-end | 90.467% | 1 | 0 / 1 |

The apparent +0.033 point end-to-end gain is not real model quality. It is one
accidental edge-order effect described below. Semantically canonicalized CV and
oracle graphs produce identical decisions on all 3,000 robot examples.

The oracle-policy error count is 287/3,000. Errors by robot are:

| Robot | Accuracy | Errors / 300 |
|---|---:|---:|
| R0 | 100.00% | 0 |
| R1 | 94.67% | 16 |
| R2 | 94.00% | 18 |
| R3 | 88.33% | 35 |
| R4 | 89.67% | 31 |
| R5 | 95.00% | 15 |
| R6 | 86.33% | 41 |
| R7 | 85.33% | 44 |
| R8 | 88.33% | 35 |
| R9 | 82.67% | 52 |

Therefore, on public validation, nearly all remaining error is in policy
identification, especially R9, R7 and R6. It is not caused by image reading,
goal/via parsing, urgency/fragility parsing or style recognition.

## CV findings

The safe two-model ensemble

```text
heat_v3 + mean(crops_v3, crops_v3b)
```

reconstructed all 300/300 validation graphs exactly. This includes nodes,
edges, status, stairs, one-way direction, landmarks, robot position/heading,
weather and legend mapping.

A useful individual case is scene 249. `crops_v3b` alone called the landmark at
`(5, 1)` a clinic, while `crops_v3` called it the correct dorm. Averaging the
two models recovered the correct class. This supports the ensemble as a safe
robustness improvement, but it is only a public-validation result and does not
prove perfect private-test CV.

## NLP and mission fields

`nlp_v3_holdout` scored 100% on all 300 validation missions:

- goal: 300/300
- goal reference: 300/300
- via: 300/300
- via reference: 300/300
- urgent: 300/300
- fragile: 300/300
- exact mission: 300/300
- resolved target: 300/300

Replacing only the oracle mission with the held-out NLP mission changed zero
of 3,000 robot decisions. Thus goal/via/flags have zero measured contribution
to the current validation gap. This does not remove the generalization risk of
new private-test phrasing.

The train-only image-style classifier also scored 300/300 and changed no policy
decision.

## Important policy invariance bug

Although raw and oracle graphs were semantically identical, raw CV changed one
R7 decision in scene 138:

```text
raw CV order -> direction 1 (label 1)
canonical order -> direction 3
```

The cause is in structured route summaries. When several full paths have equal
objective cost, `_route_summary` retains the first popped path. Adjacency order
comes from input edge order, so equal-cost paths can produce different path
counters even though the graph is unchanged. For direction 3 in scene 138, 18
route-summary features changed. The R7 probability then changed enough to swap
the selected first move.

Shuffling only the edge list 40 times, with no semantic change, gave:

```text
direction 1: 19 times
direction 3: 21 times
```

Sorting nodes, edges and landmarks into row-major canonical order before
feature extraction made predicted-CV and oracle decisions identical on all
3,000 validation cases. Since training features were generated from row-major
`scenes.json`, canonical ordering is the principled representation, even though
the raw unordered reader happened to fix one validation label by chance.

A stronger future fix is to make `_route_summary`'s equal-cost handling itself
order-invariant: either use a deterministic secondary key over the complete
path-stat tuple or aggregate the Pareto/equal-cost summaries. Canonicalization
is the smaller safe guard.

## Conditional error profile

Full end-to-end macro accuracy by condition:

| Slice | Accuracy |
|---|---:|
| dry | 91.51% |
| rain | 89.06% |
| not urgent | 90.77% |
| urgent | 90.00% |
| not fragile | 89.57% |
| fragile | 91.51% |
| no via | 89.93% |
| via | 90.96% |

Rain is the clearest weak slice (about 2.45 points below dry). Since CV weather
and NLP flags are exact here, this gap points to policy interactions with road
status/weather, not perception. Via and fragile cases are not the current
bottleneck in aggregate.

## Recommendation

1. Keep the two-crop ensemble as the studied CV candidate.
2. Canonicalize graph container order, or make equal-cost route summaries
   intrinsically order-invariant, before relying on R7/v2 features.
3. Spend further modelling effort on policy rules for R9, R7 and R6, with rain
   as the first conditional slice to inspect.
4. Retain NLP/CV stress tests for private-shift robustness, but do not spend
   public-validation optimization budget there: both are saturated on this
   split.

## Artifacts

- `scripts/e2e_study_eval.py`: reproducible 8-way decomposition.
- `scripts/e2e_study_canonical.py`: graph-order invariance check.
- `work/e2e_study_results.json`: full metrics and condition breakdown.
- `work/e2e_study_canonical.json`: the single raw-order discrepancy.
- `work/e2e_study_permutation.json`: edge-permutation stress result.
- `work/e2e_study_cv_graphs.pkl`: validation-only predicted-graph cache.
- `work/e2e_study_policy_trainonly.pkl`: train-only research policy bundle.

