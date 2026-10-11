"""R9 Tham Lam (greedy, one-step look-ahead) — reverse-engineered rule (EXACT-B). 100% train / 100% validation.

Target: the via copies while the via is not visited, else the goal copies (refs resolved by courier.policy.targets).
If several copies remain, ONE copy is fixed from the start position: smallest squared Euclidean distance (row, col)
to the start (ties: smallest rc).
Score every legal neighbour v of the start (no closed / wrong-way one-way / stairs edge), lexicographic, smaller better:
  k1 = Manhattan(v, T) + 100*[edge crowded and mission fragile] + 3*[night style and v is a landmark other than T]
  k2 = squared Euclidean(v, T)       only when NOT urgent (0 when urgent)
  k3 = rain: -[edge covered]  (prefer covered) ; dry: 0
  k4 = [v is a landmark other than T]
  k5 = [edge crowded]
Tie-break, absolute directions: night style: RIGHT, LEFT, UP, DOWN ; other styles: UP, RIGHT, DOWN, LEFT.
"style" comes from the image (scenes.json `style`; CV-predicted at test time) — read from m.style or the `style` arg.
"""
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
CR_FRAG = 100      # crowded first edge when fragile: effectively forbidden unless every move is crowded
LM_NIGHT = 3       # stepping on a landmark at night: worth more than one Manhattan detour (2), less than two (4)
LM_DEF = "nontarget"
NIGHT_LM_DEF = "nontarget"
TARGET_FIX = True
TARGET_KEY = lambda s, t: ((s[0] - t[0]) ** 2 + (s[1] - t[1]) ** 2,)  # noqa: E731
ORDER_DAY = (UP, RIGHT, DOWN, LEFT)
ORDER_NIGHT = (RIGHT, LEFT, UP, DOWN)


def _man(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def move(m, mission, goals, vias, style=None):
    style = style if style is not None else getattr(m, "style", None)
    night = style == "night"
    urgent, fragile = bool(mission.get("urgent")), bool(mission.get("fragile"))
    vias_s = set(vias) if vias else set()
    tg = list(vias) if (vias and m.start not in vias_s) else list(goals or [])
    if not tg:
        return None
    if TARGET_FIX and len(tg) > 1:
        tg = [min(sorted(tg), key=lambda t: TARGET_KEY(m.start, t))]
    avoid = m.lm_cells - set(goals or []) - vias_s
    nontg = m.lm_cells - set(tg)
    best = {}
    for d, v, status, st in m.adj[m.start]:
        if st:
            continue
        man = min(_man(v, t) for t in tg)
        euc = min((v[0] - t[0]) ** 2 + (v[1] - t[1]) ** 2 for t in tg)
        lm = int(v in (avoid if LM_DEF == "avoid" else nontg))
        lmn = int(v in (avoid if NIGHT_LM_DEF == "avoid" else nontg))
        cr = int(status == "crowded")
        best[d] = (man + CR_FRAG * cr * fragile + LM_NIGHT * lmn * night,
                   0 if urgent else euc,
                   -int(status == "covered") if m.rain else 0,
                   lm, cr)
    if not best:
        return None
    b = min(best.values())
    order = ORDER_NIGHT if night else ORDER_DAY
    return next(d for d in order if best.get(d) == b)
