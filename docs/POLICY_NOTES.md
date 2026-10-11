# Policy notes — round v3 (INTERIM, 2026-10-08)

Status: interim, best-so-far `courier/policy.py`, so a first test submission can go in. Only R0 is fully
reverse-engineered. R1..R9 still use stopgaps (rules from the last round, sometimes with a small extra term).

## Scores (GT graph + GT mission without GT rc; `scripts/eval_policy.py`, `scripts/policy_eval.py`)

| split | macro | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 0.7802 | 1.000 | 0.722 | 0.815 | 0.747 | 0.813 | 0.796 | 0.683 | 0.811 | 0.742 | 0.676 |
| validation | 0.6840 | 1.000 | 0.650 | 0.707 | 0.580 | 0.767 | 0.707 | 0.523 | 0.720 | 0.520 | 0.667 |

The scores are the same when the goal type is dropped for map-described goals (`mission_from_scene(s, drop_map_goal_type=True)`).

## Resolving refs (exact: 0 mismatches on train and validation, for goal_ref and via_ref)
- `north/south/west/east`: the copy of the goal type that lies furthest that way (north = smallest row).
- `near/far`: the copy of the goal type with the smallest/largest Euclidean (row, col) distance to the anchor. Manhattan gives the same answer on the data.
- `north_most/south_most/west_most/east_most`: the extreme landmark among **ALL landmarks on the map**. It is
  always unique in the data, and it always has the goal type. **The goal type is NOT needed.** If several landmarks tie (only possible after a misread), the resolver prefers the goal type if one is known, otherwise the smallest rc.
- `anchor_near`: the landmark (of any type except the anchor's type) nearest the anchor by **Manhattan** distance. The anchor always has a single copy.
  Manhattan always gives a unique answer that matches GT. Euclidean is wrong in 6 train and 3 val cases, so do not use it. **The goal type is NOT needed.**
  Ties (only after a misread) fall back to Euclidean, then to the goal type.
- So for map-described goals the NLP only has to output `goal_ref = {kind, anchor}`. `goal` may be None.
- A via of the same type as the goal is dropped only when it has no ref (same as the last round).

## Robot rules
Every path search runs reverse Dijkstra over the state (node, heading, via-visited) with integer lexicographic costs.
An "avoid landmark" step means stepping onto a landmark cell that is neither the chosen goal nor the chosen via.

| R | status | rule |
|---|---|---|
| 0 | **solved (100/100)** | key 1: fewest edges. key 2: fewest non-target landmarks passed through. key 3: rain: fewest non-covered edges, then fewest crowded edges; dry: fewest crowded edges. Tie-break (relative to heading): not urgent: straight > right > left > U-turn; urgent and dry: right > straight > left > U; urgent and rain: left > straight > right > U (rain+urgent is a thin sample of 13 ties) |
| 1 | stopgap | old rule (crowded = 6) + secondary key: avoid landmarks. Labels show a strong landmark-avoidance (more than 2 edges for 1 landmark) |
| 2 | stopgap | uses R0's rule (fits better than the old rule) |
| 3 | stopgap | old rule + avoid landmarks |
| 4 | stopgap | R0's rule, legged (can use stairs) |
| 5 | stopgap | old rule (turn = 3, U-turn = 30) + avoid landmarks |
| 6 | stopgap | old rule |
| 7 | stopgap | old rule + avoid landmarks |
| 8 | stopgap | old rule + avoid landmarks |
| 9 | stopgap | old greedy Manhattan |

Findings so far, for the next step:
- With unit cost, R0's label is always inside the set of shortest first moves (2000/2000).
- Landmark nodes that are not targets are avoided. For R0 this is only a secondary key; for R1 it looks heavily weighted.
- Weather and urgency change even R0's secondary keys and tie-break. Expect the other robots to depend on rain, urgent and fragile as well.

## Graph / mission fields
No new fields. The graph needs nodes, edges (status, stairs, oneway_to), landmarks (type, rc) for ALL landmarks, robot,
heading and rain. **Landmark positions now matter for every robot**: path costs depend on passing through landmarks, and
`*_most`/`anchor_near` use the full landmark list. So landmark detection accuracy matters more than before.

## Files
- `courier/policy.py`: the policy.
- `scripts/policy_eval.py`: eval plus a resolver check plus a check with the goal type dropped.
- `scripts/policy_lab.py`: analysis helpers (ASCII render, generic Dijkstra).
