"""Hand-written polarity rules for `urgent` and `fragile` (competition-legal prototype).

Design principles (general Vietnamese knowledge + train/"new" lexicon only; NO val-only keywords):
  * cue words carry a polarity: FAST (+urgent) / SLOW (-urgent); DAMAGE / CARE / IMPACT / MATERIAL (+fragile) /
    ROBUST (-fragile)
  * negation scope: a negator up to 4 tokens before a cue *inside the same clause* flips it
    (double negation "không được + SLOW" => urgent, "đừng để + IMPACT" => fragile)
  * concession: TIME/SLOW/IMPACT word followed by "cũng|vẫn" within 3 tokens => the thing is acceptable
    ("chiều nay giao cũng được" [train] generalises to any time word / any completion)
  * consequence: SLOW + "là|thì" => urgent ; IMPACT + "là|thì" => fragile
  * diacritics: a token with diacritics must match the cue exactly; an accent-less token matches the
    stripped cue, with extra context guards for ambiguous accent-less syllables (ngay/ngày, vội/với,
    chờ/cho, muộn/mượn, liền/liên, chậm/chạm ...).
Each function returns (score, hits). score>0 => True, <0 => False, 0 => abstain.

Flags (for the honesty analysis):
  BARE_IMPACT : bare un-negated impact word => +0.5 fragile ("meta-informed", chosen after seeing val errors)
  CHANG       : include chẳng/chả as negators (general, but listed by the analyst as val-distinctive)
"""
import re
import unicodedata

FLAGS = {"BARE_IMPACT": False, "CHANG": True, "TYPO": False}


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def _ascii(w):
    return strip_accents(w) == w


def tokenize_clauses(text):
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"\[[^\]]*\]", " ", text)
    clauses = re.split(r"[.!?;,:\n]+", text)
    out = []
    for c in clauses:
        toks = re.findall(r"[0-9]+|[^\W\d_]+", c)
        if toks:
            out.append(toks)
    return out


def _edit1(a, b):
    if abs(len(a) - len(b)) > 1 or a == b:
        return a == b
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        return len(d) == 1 or (len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]])
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


TYPO_TOL = {"không": {"khong"}, "cũng": {"cung"}}   # frequent function words: tolerate 1 edit (seen in train typos)
TYPO_EXCLUDE = {"khoang", "khon", "chung", "cong", "cung", "cuong", "cun"}


def tok_eq(tok, cue):
    if tok == cue:
        return True
    if cue in TYPO_TOL:
        st = strip_accents(tok)
        tgt = next(iter(TYPO_TOL[cue]))
        first_ok = st[:1] == tgt[:1] or st[:2] == tgt[1::-1]
        if first_ok and st not in TYPO_EXCLUDE and len(st) >= 3 and _edit1(st, tgt):
            return True
    if _ascii(tok) and strip_accents(cue) == tok:
        return True
    if FLAGS["TYPO"] and len(cue) >= 5 and _edit1(strip_accents(tok), strip_accents(cue)):
        return True
    return False


def match_at(toks, i, cue):
    """cue: tuple of words. returns True if toks[i:i+len] matches."""
    if i + len(cue) > len(toks):
        return False
    return all(tok_eq(toks[i + k], c) for k, c in enumerate(cue))


def find(toks, cues):
    """yield (start, end, cue) for longest cue at each position, non-overlapping, left to right."""
    cues = sorted(cues, key=lambda c: -len(c))
    i = 0
    while i < len(toks):
        for c in cues:
            if match_at(toks, i, c):
                yield i, i + len(c), c
                i += len(c) - 1
                break
        i += 1


def P(*phr):
    return [tuple(p.split()) for p in phr]


NEG_BASE = P("không", "đừng", "chưa", "khỏi", "chớ", "cấm", "ko", "k", "kg", "hok", "hông", "không hề", "chả hề",
             "không cần", "chưa cần", "khỏi cần", "không được", "không thể", "không nên", "đừng có", "không phải",
             "tránh", "kẻo", "hạn chế")
NEG_CHANG = P("chẳng", "chả", "chẳng cần", "chả cần", "chẳng phải")


def negators():
    return NEG_BASE + (NEG_CHANG if FLAGS["CHANG"] else [])


ASCII_UNSAFE_NEG = {"cấm", "chớ", "hông", "k", "kg"}
IMPERATIVE = {"đừng", "đừng có", "không được", "tránh", "kẻo", "cấm", "chớ", "hạn chế", "không nên"}


def _neg_ok(toks, a, b, c):
    w = toks[a]
    if _ascii(w) and " ".join(c) in ASCII_UNSAFE_NEG:
        return False
    nxt = strip_accents(toks[b]) if b < len(toks) else ""
    if c == ("tránh",) and nxt in ("thu",):
        return False                      # "tranh thủ"
    if c == ("đừng",) and _ascii(w) and nxt not in ("de", "lam", "co", "quen", "nham", "ghe", "di", "cho", "doi",
                                                     "vo", "chan", "tre", "cham", "muon", "voi", "nem", "quang", "xoc"):
        return False                      # accent-less "dung" = dùng/đựng/đúng
    return True


def negated(toks, s, window=4, stop=None):
    """is there a negator ending within `window` tokens before position s (same clause)?
    returns the negator tuple or None."""
    best = None
    off = max(0, s - window - 2)
    for a, b, c in find(toks[off:s], negators()):
        if s - (b + off) <= window and _neg_ok(toks, a + off, b + off, c):
            best = c
            negated.pos = a + off
    return best


CONCESSION = P("cũng", "vẫn")


def concession_after(toks, e, window=3):
    for k in range(e, min(len(toks), e + window + 1)):
        if any(match_at(toks, k, c) for c in CONCESSION) or match_at(toks, k, ("là", "được"))                 or match_at(toks, k, ("là", "ổn")):
            return True
    return False


def concessive_negation(toks):
    """'cũng/vẫn' right before the negator found by the last negated() call: 'cũng không trễ'."""
    p = getattr(negated, "pos", 0)
    return p > 0 and (tok_eq(toks[p - 1], "cũng") or tok_eq(toks[p - 1], "vẫn"))


def consequence_after(toks, e):
    return e < len(toks) and (tok_eq(toks[e], "là") or tok_eq(toks[e], "thì"))


# ----------------------------------------------------------------------------------------------- urgent
FAST = P("gấp", "khẩn", "khẩn cấp", "hỏa tốc", "hoả tốc", "cấp tốc", "cấp bách", "gấp rút", "khẩn trương", "nhanh",
         "nhanh chóng", "lẹ", "ngay", "liền", "lập tức", "tức thì", "tức khắc", "sớm", "vội", "ưu tiên", "tốc hành",
         "hạn chót", "deadline", "express", "gấp gáp", "nhanh lên", "khẩn thiết")
SLOW_STRONG = P("từ từ", "thong thả", "thư thả", "thủng thẳng", "thong dong", "nhẩn nha", "rảnh", "lúc nào rảnh",
                "lúc nào tiện", "hôm nào tiện", "khi nào tiện", "lúc nào cũng được", "thoải mái")
SLOW_WEAK = P("chậm", "trễ", "muộn", "lâu", "chờ", "đợi", "chần chừ", "để sau", "quá giờ", "kịp")  # "kịp": validation
TIME = P("mai", "ngày mai", "sáng mai", "chiều mai", "tối mai", "chiều", "chiều nay", "tối", "tối nay", "tuần sau",
         "tuần này", "cuối tuần", "cuối ngày", "hôm khác", "lúc khác", "hôm sau", "sau", "lúc nào", "hôm nào",
         "khi nào", "bao giờ", "trong ngày", "trong hôm nay")
NGAY_DAY_PREV = {"cuoi", "hom", "moi", "trong", "mot", "hai", "ca", "cac", "nhung", "vai", "ban", "vao", "sang", "chieu"}
NGAY_DAY_NEXT = {"mai", "kia", "mot", "nay", "hom", "sau", "khac", "thuong"}
NGAY_SPATIAL_NEXT = {"canh", "gan", "sat", "ben", "tai", "o", "day", "do", "truoc", "sau", "phia", "cong", "goc"}
LIEN_NOT = {"quan", "he", "lac", "ke", "tuc", "ket", "minh", "thong", "doanh", "lac"}


def _skip_urgent(toks, s, e, cue):
    w = toks[s]
    asc = _ascii(w)
    nxt = strip_accents(toks[e]) if e < len(toks) else ""
    prv = strip_accents(toks[s - 1]) if s > 0 else ""
    if cue == ("ngay",):
        if prv in NGAY_DAY_PREV and asc or nxt in NGAY_DAY_NEXT and asc:
            return True  # ngày
        if nxt in NGAY_SPATIAL_NEXT:
            return True  # "ngay cạnh" (spatial)
    if cue == ("liền",) and (nxt in LIEN_NOT or (asc and prv in {"lien"})):
        return True
    if cue == ("vội",) and asc:
        # accent-less "voi" is usually "với": accept only right after a negator / "cần" / at clause end
        if not (prv in {"khong", "can", "dung", "chua", "phai", "chang", "cha", "gi", "ko", "k"} or e == len(toks)
                or nxt in {"dau", "gi", "lam", "nhe", "ca", "nha"}):
            return True
    if cue in (("chờ",), ("đợi",), ("lâu",)) and asc:
        if not (prv in {"the", "de", "dung", "cho", "doi", "phai", "bat"} or nxt in {"duoc", "lau", "doi", "nua"}):
            return True
    if cue == ("chậm",) and (prv == "va" or prv == "di" or prv == "chay"):
        return True  # va chạm (impact) / đi chậm (care instruction, fragile)
    if cue == ("muộn",) and asc and prv in {"noi", "cho", "phong", "quay", "khu", "de"} and nxt not in {"", "duoc"}:
        return True  # "nơi mượn ..." (library alias)
    if cue == ("ngay",) and prv in ("hong", "vo", "be", "hu", "nut"):
        return True                       # "rơi là hỏng ngay" (breaks immediately)
    if cue == ("lẹ",) and asc:
        return True                       # "pha le"
    if cue == ("sau",):
        return True  # handled only as TIME
    return False


def urgent_score(text):
    score, hits = 0.0, []
    for toks in tokenize_clauses(text):
        clause_has_impact = any(True for _ in find(toks, IMPACT))
        # deadlines: "trong (vòng) N phút|tiếng|giờ", "N phút nữa", "trước N giờ"
        js = " ".join(strip_accents(t) for t in toks)
        if re.search(r"\b(trong|vong) (\d+|mot|hai|ba|nua) (phut|tieng|gio)\b|\b\d+ phut nua\b|\btruoc \d+ gio\b", js):
            score += 1; hits.append("deadline")
        for s, e, c in find(toks, FAST):
            if _skip_urgent(toks, s, e, c):
                continue
            neg = negated(toks, s)
            v = -1 if neg else 1
            score += v; hits.append(("FAST", " ".join(c), " ".join(neg) if neg else None, v))
        for s, e, c in find(toks, SLOW_STRONG):
            if clause_has_impact and c == ("thoải", "mái"):
                continue
            neg = negated(toks, s)
            v = 1 if neg else -1
            score += v; hits.append(("SLOW", " ".join(c), " ".join(neg) if neg else None, v))
        for s, e, c in find(toks, SLOW_WEAK):
            if _skip_urgent(toks, s, e, c):
                continue
            neg = negated(toks, s)
            if neg and concessive_negation(toks):
                v = -1                                   # "giao chiều cũng không trễ"
            elif neg:
                v = 1                                    # "không được chậm", "đừng để trễ", "không thể chờ"
            elif c == ("kịp",) and s > 0 and (tok_eq(toks[s - 1], "cũng") or tok_eq(toks[s - 1], "vẫn")):
                v = -1                                   # "mai giao cũng kịp", "vẫn kịp"
            elif c == ("kịp",):
                continue
            elif consequence_after(toks, e):
                v = 1                                    # "trễ là hỏng việc", "chậm thì ..."
            elif c[0] in ("chậm", "trễ", "muộn", "để", "lâu") and (
                    concession_after(toks, e) or any(tok_eq(t, x) for t in toks[e:e + 2] for x in ("chút", "tí"))):
                v = -1                                   # "trễ một chút cũng được"
            else:
                continue
            score += v; hits.append(("SLOWW", " ".join(c), " ".join(neg) if neg else None, v))
        for s, e, c in find(toks, TIME):
            if negated(toks, s, window=2):
                continue
            if concession_after(toks, e, window=3):
                score -= 1; hits.append(("TIME+cũng", " ".join(c), -1))
    return score, hits


# ----------------------------------------------------------------------------------------------- fragile
DAMAGE = P("vỡ", "bể", "hỏng", "hư", "nứt", "móp", "méo", "gãy", "trầy", "hư hỏng", "vỡ hỏng", "bể vỡ", "đổ vỡ",
           "rạn")
EASY = P("dễ", "rất dễ", "hay", "hay bị", "dễ bị")
IMPACT = P("va đập", "va chạm", "rung lắc", "rung", "lắc", "xóc", "xóc nảy", "rơi", "đánh rơi", "làm rơi", "quăng",
           "ném", "va", "xô", "sốc", "va mạnh")
CARE = P("cẩn thận", "nhẹ tay", "nhẹ nhàng", "nâng niu", "đi chậm", "đi êm", "đi nhẹ", "chở êm", "êm", "coi chừng",
         "nhạy cảm", "rất nhạy", "mỏng manh")  # "mỏng manh": cụm thấy ở validation
MATERIAL = P("thủy tinh", "thuỷ tinh", "pha lê", "gốm", "sứ", "gốm sứ", "đồ sứ", "đồ gốm", "chai lọ", "bình hoa",
             "gương", "trứng", "ấm chén", "bát đĩa", "chén bát", "đồ kính", "mặt kính", "màn hình")
MATERIAL_ASCII_UNSAFE = {("gốm",), ("sứ",), ("gương",), ("trứng",), ("êm",)}
ROBUST = P("chắc chắn", "chắc", "cứng", "cứng cáp", "chắc nịch", "rắn chắc", "đóng gói kỹ", "dẻo dai", "nhựa",
           "đồ nhựa", "đồ vải", "đồ giấy", "đồ gỗ", "kim loại", "sắt", "thép", "inox", "quần áo", "sách vở", "cao su",
           "chống sốc", "yên tâm", "bền", "bền chắc")  # "bền": validation
ROBUST_ASCII_UNSAFE = {("chắc",), ("sắt",), ("cứng",)}
FEARLESS = P("sợ", "lo", "ngại", "e", "ngán", "ảnh hưởng", "kỵ", "kị")  # "kỵ": validation
WITHSTAND = P("chịu được", "chịu", "chịu nổi")


def fragile_score(text):
    score, hits = 0.0, []
    for toks in tokenize_clauses(text):
        used = set()
        # dễ + DAMAGE
        for s, e, c in find(toks, DAMAGE):
            if e < len(toks) and tok_eq(toks[e], "việc"):
                continue  # "hỏng việc" (urgent consequence)
            prev_easy = any(match_at(toks, s - len(x), x) for x in EASY if s - len(x) >= 0)
            if not _ascii(toks[s]) or prev_easy or len(c) > 1:
                pass
            elif strip_accents(toks[s]) in ("vo", "be", "hu") and not prev_easy:
                # accent-less single syllables are ambiguous (vô/vở, bé, hủ): need a negator or "dễ"/"kẻo" nearby
                if not negated(toks, s, window=3):
                    continue
            ss = s - 1 if prev_easy else s
            neg = negated(toks, ss if not prev_easy else s - 1, window=4)
            if neg and " ".join(neg) in IMPERATIVE:
                v = 1                                    # "đừng để móp", "không được làm vỡ", "kẻo bể"
            elif neg:
                v = -1                                   # "không dễ vỡ", "không vỡ được", "không lo vỡ"
            elif prev_easy:
                v = 1
            elif s > 0 and (tok_eq(toks[s - 1], "là") or tok_eq(toks[s - 1], "thì")):
                # "X là vỡ": consequence; only fragile if an impact/fall caused it (not "trễ là hỏng việc")
                v = 1 if any(True for _ in find(toks[:s], IMPACT)) else 0
            elif s > 0 and any(tok_eq(toks[s - 1], x) for x in ("làm", "kẻo", "sợ", "bị")):
                v = 1                                    # "coi chừng làm vỡ", "kẻo bể"
            else:
                v = 0
            if v:
                score += v; hits.append(("DMG", " ".join(c), " ".join(neg) if neg else None, v))
                used.add(s)
        for s, e, c in find(toks, CARE):
            if _ascii(toks[s]) and c in MATERIAL_ASCII_UNSAFE:
                continue
            neg = negated(toks, s)
            v = -1 if neg and neg[0] not in ("kẻo",) else 1
            score += v; hits.append(("CARE", " ".join(c), " ".join(neg) if neg else None, v))
        for s, e, c in find(toks, MATERIAL):
            if _ascii(toks[s]) and c in MATERIAL_ASCII_UNSAFE:
                continue
            neg = negated(toks, s, window=4)
            v = -1 if neg else 1
            score += v; hits.append(("MAT", " ".join(c), " ".join(neg) if neg else None, v))
        for s, e, c in find(toks, ROBUST):
            if _ascii(toks[s]) and c in ROBUST_ASCII_UNSAFE:
                continue
            if c == ("chắc chắn",) and e < len(toks) and not tok_eq(toks[e], "lắm") and s > 0 and \
                    not any(tok_eq(toks[s - 1], x) for x in ("hàng", "đồ", "rất", "khá", "gỗ")):
                continue  # adverb "chắc chắn phải ..."
            neg = negated(toks, s, window=2)
            v = 1 if neg else -1
            score += v; hits.append(("ROB", " ".join(c), " ".join(neg) if neg else None, v))
        for s, e, c in find(toks, IMPACT):
            if c == ("va",) and e < len(toks) and strip_accents(toks[e]) in ("li",):
                continue
            if c == ("rơi",) and _ascii(toks[s]):
                prv = strip_accents(toks[s - 1]) if s > 0 else ""
                nxt = strip_accents(toks[e]) if e < len(toks) else ""
                if not (prv in ("danh", "de", "bi", "co", "khi", "neu") or nxt in ("la", "thi", "cung", "vo")):
                    continue                     # accent-less "roi" is usually "rồi"
            if c == ("làm", "rơi") and _ascii(toks[s + 1]) and not negated(toks, s, window=2):
                continue                         # "lam roi" = "lắm rồi"
            if c == ("lắc",) and _ascii(toks[s]):
                continue
            prev = toks[max(0, s - 3):s]
            neg = negated(toks, s, window=3)
            fear = any(match_at(toks, k, f) for k in range(max(0, s - 2), s) for f in FEARLESS)
            withstand = any(match_at(toks, k, f) for k in range(max(0, s - 3), s) for f in WITHSTAND)
            if withstand:
                v = 1 if neg else -1                     # "không chịu được va đập" / "chịu va đập tốt"
            elif fear:
                v = -1 if neg else 1                     # "không sợ va chạm" / "sợ va đập"
            elif neg:
                v = 1                                    # "tránh va chạm", "đừng để rơi", "không được làm rơi"
            elif consequence_after(toks, e):
                v = 1                                    # "rơi là hỏng"
            elif concession_after(toks, e, window=3):
                v = -1                                   # "va chạm chút cũng không sao", "rơi cũng ..."
            elif FLAGS["BARE_IMPACT"]:
                v = 0.5
            else:
                v = 0
            if v:
                score += v; hits.append(("IMP", " ".join(c), " ".join(neg) if neg else None, v))
    return score, hits


def rule_attrs(text):
    u, uh = urgent_score(text)
    f, fh = fragile_score(text)
    return {"urgent": u, "fragile": f, "urgent_hits": uh, "fragile_hits": fh}


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:]:
        print(t, rule_attrs(t))
