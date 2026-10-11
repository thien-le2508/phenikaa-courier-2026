"""Sinh câu yêu cầu tổng hợp có nhãn cho vòng v3.

    python -m courier.nlp.generate --n 150000 --out work/nlp/synth_holdout.json          # không dùng cụm của val v3
    python -m courier.nlp.generate --n 150000 --out work/nlp/synth_full.json --use-val   # thêm cụm của val v3

Nhãn: goal (None nếu nơi giao chỉ được mô tả qua bản đồ và câu không lộ loại địa điểm), goal_ref {kind, anchor},
via, via_ref, urgent, fragile.
Từ điển: lexicon.py (vòng trước: nhóm train/new/val cũ đều được dùng, vì đó là dữ liệu vòng trước, không phải
val/test v3) + lexicon3.py (nhóm train/new; nhóm "val" = cụm chỉ có ở val v3, chỉ dùng khi --use-val).
"""
import argparse
import json
import random
import unicodedata

from courier.data import LANDMARKS
from courier.nlp import lexicon as L
from courier.nlp import lexicon3 as L3

MAP_KINDS = ["north_most", "south_most", "west_most", "east_most", "anchor_near"]
DIR_KINDS = ["north", "south", "west", "east"]


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


KEYBOARD = "qwertyuiopasdfghjklzxcvbnm"


def typo(text, rng, n):
    chars = list(text)
    for _ in range(n):
        idx = [i for i, c in enumerate(chars) if c.isalpha()]
        if len(idx) < 2:
            break
        i = rng.choice(idx)
        op = rng.random()
        if op < 0.3 and i + 1 < len(chars) and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op < 0.55:
            del chars[i]
        elif op < 0.7:
            chars.insert(i, chars[i])
        elif op < 0.85:
            chars[i] = rng.choice(KEYBOARD)
        else:
            chars[i] = strip_accents(chars[i])
    return "".join(chars)


ABBR = [("không", "ko"), ("không", "k"), ("được", "dc"), ("được", "đc"), ("với", "vs"), ("mình", "mk"),
        ("nhé", "nha"), ("rồi", "r"), ("gì", "j")]


def add_noise(text, rng, p_lower=0.22, p_strip=0.32, p_partial=0.08, p_typo=0.4, p_extra=0.15):
    if rng.random() < p_extra:  # nhiễu bổ sung: viết tắt, bỏ dấu câu, khoảng trắng thừa
        r = rng.random()
        if r < 0.4:
            a, b = rng.choice(ABBR)
            text = text.replace(" " + a + " ", " " + b + " ", 1)
        elif r < 0.7:
            text = text.replace(".", "", rng.randint(1, 3))
        else:
            text = text.replace(" ", "  ", rng.randint(1, 3))
    if rng.random() < p_lower:
        text = text.lower()
    r = rng.random()
    if r < p_strip:
        text = strip_accents(text)
    elif r < p_strip + p_partial:
        words = text.split(" ")
        frac = rng.uniform(0.2, 0.7)
        text = " ".join(strip_accents(w) if rng.random() < frac else w for w in words)
    if rng.random() < p_typo:
        text = typo(text, rng, rng.choice([1, 1, 2, 2, 3, 4, 5]))
    return text


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


class Generator:
    def __init__(self, use_val=False, seed=0):
        self.rng = random.Random(seed)
        self.use_val = use_val
        P = lambda e: L.pool(e, True)  # lexicon cũ: dùng hết (kể cả nhóm val của vòng trước)
        Q = lambda e: list(e.get("train", [])) + list(e.get("new", [])) + (list(e.get("val", [])) if use_val else [])
        self.aliases = {t: P(L.ALIASES[t]) + L3.ALIASES_EXTRA[t] for t in LANDMARKS}
        self.roles = {t: P(L.ROLES[t]) + L3.ROLES_EXTRA[t] + (L3.ROLES_VAL3.get(t, []) if use_val else []) for t in LANDMARKS}
        self.refs = {k: P(v) + L.SPATIAL_EXTRA[k] + L3.REFS_EXTRA[k] for k, v in L.REFS.items()}
        self.verbs, self.to, self.please = P(L.VERBS), P(L.TO), P(L.PLEASE)
        self.goal_frames = P(L.GOAL_FRAMES) + Q(L3.GOAL_FRAMES_EXTRA)
        self.role_only = [f for f in self.goal_frames if f in L.ROLE_ONLY]
        self.role_g = Q(L3.ROLE_G_FRAMES) + [f for f in self.goal_frames if "{ROLE}" in f and "{G}" in f]
        self.g_frames = [f for f in self.goal_frames if "{G}" in f and "{ROLE}" not in f]
        self.via_goal_frames = P(L.VIA_GOAL_FRAMES) + Q(L3.VIA_GOAL_FRAMES_EXTRA)
        self.via_frames = P(L.VIA_FRAMES) + Q(L3.VIA_FRAMES_EXTRA)
        self.dis_frames = P(L.DISTRACTOR_FRAMES) + Q(L3.DISTRACTOR_EXTRA)
        self.urgent = {b: P(L.URGENT[b]) for b in (True, False)}
        self.fragile = {b: P(L.FRAGILE[b]) for b in (True, False)}
        self.greetings = P(L.GREETINGS) + Q(L3.GREETINGS_EXTRA)
        self.closings = P(L.CLOSINGS) + Q(L3.CLOSINGS_EXTRA)
        self.func_acts = {t: P(dict(train=[], **L.FUNC_ACTS[t])) for t in LANDMARKS}
        self.words = {k: P(dict(train=[], **v)) for k, v in L.ATTR_WORDS.items()}
        self.generic = Q(L3.GENERIC_PLACE)
        self.most = {k: Q(v) for k, v in L3.MOST.items()}
        self.anchor_near = Q(L3.ANCHOR_NEAR)
        self.corr_pre, self.corr_suf = Q(L3.CORRECTION_PREFIX), Q(L3.CORRECTION_SUFFIX)
        self.contrast = Q(L3.CONTRAST)
        self.train_most = {k: v["train"] for k, v in L3.MOST.items()}

    # ------------------------------------------------------------------ mảnh câu
    def alias(self, t):
        if self.rng.random() < 0.2:
            return self.rng.choice(L.FUNC_PREFIX) + " " + self.rng.choice(self.func_acts[t])
        return self.rng.choice(self.aliases[t])

    def attr(self, kind, value):
        rng = self.rng
        if rng.random() < 0.5:
            return rng.choice((self.urgent if kind == "urgent" else self.fragile)[value])
        frame = rng.choice((L.URGENT_GRAMMAR if kind == "urgent" else L.FRAGILE_GRAMMAR)[value])
        w = {k: rng.choice(v) for k, v in self.words.items()}
        rep = dict(w)
        rep.update({k + "_cap": _cap(v) for k, v in w.items()})
        rep.update({k + "_low": v[:1].lower() + v[1:] for k, v in w.items()})
        rep["n"] = str(rng.choice([3, 5, 10, 15, 20, 30]))
        for k, v in rep.items():
            frame = frame.replace("{" + k + "}", v)
        return _cap(frame)

    def role(self, t):
        return self.rng.choice(self.roles[t]).replace("{ALIAS}", self.alias(t))

    def item(self, t, generic=False):
        pool = L.ITEMS[t] if (not generic and self.rng.random() < 0.55) else L.ITEMS["generic"]
        return self.rng.choice(pool)

    def place(self, t, ref):
        """Tên địa điểm có loại, kèm tham chiếu kiểu cũ (north/.../near/far)."""
        s = self.alias(t)
        if ref:
            phrase = self.rng.choice(self.refs[ref["kind"]])
            if "{A}" in phrase:
                phrase = phrase.replace("{A}", self.alias(ref["anchor"]))
            if self.rng.random() < 0.15:
                s += ","
            s += " " + phrase
        return s

    def map_place(self, ref):
        """Nơi giao chỉ mô tả qua bản đồ (không có loại)."""
        rng = self.rng
        k = ref["kind"]
        pool = self.anchor_near if k == "anchor_near" else self.most[k]
        phrase = rng.choice(pool)
        p = rng.choice(self.generic)
        phrase = phrase.replace("{P}", p)
        if "{A}" in phrase:
            phrase = phrase.replace("{A}", self.alias(ref["anchor"]))
        return phrase

    def fill(self, frame, **kw):
        rng = self.rng
        verb = rng.choice(self.verbs)
        rep = {"{v}": verb, "{V}": _cap(verb), "{TO}": rng.choice(self.to), "{PL}": rng.choice(self.please)}
        rep.update(kw)
        for k, v in rep.items():
            frame = frame.replace(k, v)
        return _cap(frame)

    def sample_ref(self, exclude, maybe_anchor=None):
        kind = self.rng.choice(DIR_KINDS + ["near", "near", "far", "far"])
        if kind in ("near", "far"):
            if maybe_anchor and self.rng.random() < 0.08:
                return {"kind": kind, "anchor": maybe_anchor}
            return {"kind": kind, "anchor": self.rng.choice([t for t in LANDMARKS if t not in exclude])}
        return {"kind": kind, "anchor": None}

    # ------------------------------------------------------------------ một yêu cầu
    def sample(self):
        rng = self.rng
        goal = rng.choice(LANDMARKS)
        via = rng.choice([t for t in LANDMARKS if t != goal]) if rng.random() < 0.48 else None
        r = rng.random()
        if r < 0.36:
            goal_ref = None
        elif r < 0.60:
            anc = via if (via and rng.random() < 0.12) else rng.choice([t for t in LANDMARKS if t not in (goal, via)])
            goal_ref = {"kind": "anchor_near", "anchor": anc}  # mốc có thể chính là điểm ghé
        elif r < 0.80:
            goal_ref = {"kind": rng.choice(MAP_KINDS[:4]), "anchor": None}
        else:
            goal_ref = self.sample_ref({goal, via}, maybe_anchor=via)
        via_ref = self.sample_ref({via, goal}, maybe_anchor=goal) if via and rng.random() < 0.25 else None
        is_map = goal_ref is not None and goal_ref["kind"] in MAP_KINDS
        revealed = True

        used = {goal, via}
        if goal_ref and goal_ref.get("anchor"):
            pass  # mốc có thể xuất hiện cả ở câu gây nhiễu (giống train)
        others = [t for t in LANDMARKS if t not in used]
        wrong = rng.choice(others)  # nơi sai trong đính chính / đối chiếu

        # ---- cụm nơi giao G
        contrast = False
        if is_map:
            G = self.map_place(goal_ref)
        else:
            G = self.place(goal, goal_ref)
        if rng.random() < (0.05 if is_map else 0.2):
            contrast = True
            frame = rng.choice(self.contrast)
            G = frame.replace("{G}", G).replace("{X}", self.alias(wrong))

        I, J = self.item(goal, generic=is_map), self.item(via) if via else ""
        core = []
        if via and rng.random() < 0.6:
            frame = rng.choice(self.via_goal_frames)
            core.append(self.fill(frame, **{"{G}": G, "{B}": self.place(via, via_ref), "{I}": I, "{J}": J}))
            if is_map:
                revealed = False
        else:
            if is_map:
                if rng.random() < 0.3:
                    frame = rng.choice(self.role_g)
                else:
                    frame = rng.choice(self.g_frames)
                    revealed = False
            elif goal_ref or contrast:
                frame = rng.choice([f for f in self.goal_frames if f not in self.role_only])
            else:
                frame = rng.choice(self.goal_frames)
            core.append(self.fill(frame, **{"{G}": G, "{I}": I, "{ROLE}": self.role(goal)}))
            if via:
                vs = self.fill(rng.choice(self.via_frames), **{"{B}": self.place(via, via_ref), "{J}": J})
                core.append(vs) if rng.random() < 0.8 else core.insert(0, vs)

        # ---- đính chính
        extra = []
        r = rng.random()
        if r < 0.22:  # tiền tố: "Lúc nãy nhắn nhầm là X. Đúng ra: <câu chính>"
            frame = rng.choice(self.corr_pre)
            X = self.alias(wrong if not contrast else rng.choice(others))
            pre, post = frame.split("{CORE}")
            pre = pre.replace("{X}", X)
            core[0] = pre + (core[0] if pre.endswith((": ", ". ")) or pre.endswith(":") else core[0][:1].lower() + core[0][1:])
            core[0] = _cap(core[0])
        elif r < 0.36:
            extra.append(self.rng.choice(self.corr_suf).replace("{X}", self.alias(rng.choice(others))))

        # ---- gấp / dễ vỡ
        urgent = fragile = False
        attrs = []
        if rng.random() < 0.8:
            urgent = rng.random() < 0.5
            attrs.append(self.attr("urgent", urgent))
        if rng.random() < 0.8:
            fragile = rng.random() < 0.5
            attrs.append(self.attr("fragile", fragile))
        if len(attrs) == 2 and rng.random() < 0.15:
            a, b = attrs if rng.random() < 0.5 else attrs[::-1]
            attrs = [a.rstrip(".!") + ", " + b[0].lower() + b[1:]]
        extra += attrs

        # ---- gây nhiễu: không trùng loại nơi giao / điểm ghé; có thể trùng mốc
        cand = [t for t in LANDMARKS if t not in used]
        for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 3])):
            t = rng.choice(cand)
            extra.append(self.fill(rng.choice(self.dis_frames), **{"{X}": self.alias(t)}))
        if rng.random() < 0.08:
            extra.append(rng.choice(L3.FILLERS))
        rng.shuffle(extra)
        sents = list(core)
        for e in extra:
            sents.insert(rng.randrange(len(sents) + 1), e)
        if rng.random() < 0.4:
            sents.append(rng.choice(self.closings))

        text = " ".join(sents)
        prefix = []
        if rng.random() < 0.5:
            prefix.append(rng.choice(self.greetings))
        if rng.random() < 0.15:
            prefix.insert(rng.randrange(len(prefix) + 1), f"[Đơn #{rng.randint(1, 9999)}]")
        if prefix:
            if prefix[-1].endswith(","):
                text = text[0].lower() + text[1:]
            text = " ".join(prefix) + " " + text
        text = add_noise(text, rng)
        return {"text": text, "goal": goal if (revealed or not is_map) else None, "goal_true": goal,
                "goal_ref": goal_ref, "via": via, "via_ref": via_ref, "urgent": urgent, "fragile": fragile}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-val", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    g = Generator(use_val=a.use_val, seed=a.seed)
    data = [g.sample() for _ in range(a.n)]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    for d in data[:12]:
        print(json.dumps(d, ensure_ascii=False))


if __name__ == "__main__":
    main()
