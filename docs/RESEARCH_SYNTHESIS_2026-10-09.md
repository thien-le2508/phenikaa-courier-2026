# Policy research synthesis (2026-10-09)

## Leakage and evaluation discipline

- Every study was restricted to public `train` and `validation`; no `test_private` file was read.
- Candidate/configuration selection used scene-level train OOF. Validation was a report-only sanity check.
- End-to-end decomposition refit policy/style on train only and used the held-out NLP model.
- No research candidate was silently installed into the final prediction pipeline.

## Executive conclusion

The public-validation bottleneck is policy inference, not perception:

- CV ensemble reconstructs 300/300 validation graphs exactly.
- Held-out NLP resolves all 300 missions exactly.
- Train-only style prediction is 300/300.
- Canonical end-to-end inputs reproduce oracle-policy decisions 3000/3000.
- The train-only oracle policy still makes 287/3000 errors (90.433%).

Errors concentrate in R9 (52), R7 (44), R6 (41), then R3/R8 (35 each). Rain is the weakest slice
(89.06% versus 91.51% for dry scenes).

## Findings ranked by confidence

### 1. Promote for implementation review: R9 conditional structured utility

R9 shared + rain/urgent/fragile/via gates + style gates, with hierarchical L2 shrinkage:

| Model | Train OOF | Validation |
|---|---:|---:|
| Existing v1 HGB | 85.70% | 78.33% |
| Conditional structured R9 | **89.90%** | **91.67%** |

The gain survives every validation style and binary-condition slice. A probability blend lowers validation to 89.33%,
so the structured model should remain standalone. This is the strongest policy discovery in the study.

### 2. Promote for implementation review: deterministic graph ordering

Semantically identical graphs can currently yield different policy decisions because equal-cost route summaries retain
the first path encountered. Shuffling edge order for R7 validation scene 138 produced direction 1 nineteen times and
direction 3 twenty-one times. Canonical row-major ordering makes predicted-CV and oracle decisions identical on all
3000 validation robot cases.

The minimal guard is canonical sorting of nodes/edges/landmarks before feature extraction. The principled fix is an
order-independent secondary key or aggregation over equal-cost/Pareto summaries. This should be fixed and regression
tested before trusting v2 features on new graph serializations.

### 3. Keep as calibrated candidates: R5/R6 ensembles

Nested (fold-honest) probability blending finds:

- R5: 96.85% nested OOF / 95.33% validation.
- R6: 89.05% nested OOF / 89.33% validation.

The new conditional component receives zero weight here; gains come from calibrated v1/v2/NIP combinations. These are
ensemble improvements, not evidence for a complex condition rule.

### 4. Keep as auxiliary evidence, not hard rules

- R6 tie behavior: non-night `RSLU`, night `LSRU`, indifference band about 0.2 edges; stable across all five folds.
- R3 route interaction: rain increases the value of covered routes and crowd avoidance.
- R5/R6: U-turn aversion within a limited detour budget is a useful route counter.
- R8: via changes utility; no-via favors avoiding U-turns, via favors avoiding crowded edges.
- R4: stairs are feasible, but data does not support always preferring or avoiding them.

These structured experts are weaker than HGB alone. They should enter a ranker as regularized features, not replace it.

### 5. Reject or defer

- Independent policies for all 16 rain/urgent/fragile/night cells: large train-validation gaps, especially R8
  (84.35% train versus 59.33% validation in full-data inverse fitting).
- Unrestricted pairwise/high-order interactions: consistently worse OOF than main effects.
- Conditional models for R7/R8: train-to-validation distribution shift; via prevalence is 32.7% in train and 52.0%
  in validation.
- Hard near-tie rules for R1/R7: apparent rules often put U-turn first, suggesting missing utility features rather
  than a true tie preference.
- Separate goal-ambiguity policy: only 10 validation scenes have unresolved duplicate goals.

## Robust model-selection audit

Paired bootstrap against v1 shows positive train-OOF intervals for standard ensembles on R1, R4, R5, R6, R7 and R8.
R3's ensemble interval crosses zero, so its small gain is uncertain. Standard R9 ensembles are also uncertain and are
superseded by the independently validated conditional model.

OOF-selected per-robot candidates improve mean R1--R9 OOF from 90.38% for v1 alone to about 91.85%, but nested blend
estimates must be preferred wherever weights were tuned on the same OOF predictions.

## Recommended implementation order

1. Add graph-order invariance regression tests, then canonicalize route-feature input (or make equal-cost summaries
   intrinsically invariant).
2. Package the train-full R9 conditional model for inference and validate its feature construction against research
   OOF/validation artifacts.
3. Package calibrated R5/R6 blends if inference compatibility tests pass.
4. Add selected route counters as strongly regularized features and rerun fixed-fold OOF; do not use hard rules.
5. Re-run end-to-end validation after every integration. Keep test-private completely sealed until producing the final
   submission predictions.

## Reports and machine-readable artifacts

- Conditional/hierarchical: `work/conditional_RESEARCH_2026-10-09.md`, `work/ml2/conditional_*.{json,pkl}`
- Tie/near-tie: `work/TIE_STUDY_2026-10-09.md`, `work/ml2/tie_study.json`
- Structured routes: `work/ROUTE_STUDY_2026-10-09.md`, `work/ml2/route_study_*.{json,pkl}`
- End-to-end: `work/e2e_study_REPORT.md`, `work/e2e_study_*.{json,pkl}`
- Initial interactions: `work/INTERACTION_RESEARCH_2026-10-09.md`, `work/ml2/interaction_study.json`
- Robust selection: `work/ml2/robust_selection.json`

All associated scripts under `scripts/` passed Python bytecode compilation after the studies completed.
