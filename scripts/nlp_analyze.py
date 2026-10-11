"""Phân tích câu yêu cầu: tách câu, thay tên gọi bằng chỗ trống, đếm khung câu.

    python scripts/nlp_analyze.py --split train
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.data import LANDMARKS, load_split  # noqa: E402
from courier.nlp import lexicon as L  # noqa: E402

VI = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


def all_phr(entry):
    return [x for g in entry.values() for x in g]


def build_subs():
    subs = []
    for t in LANDMARKS:
        for a in all_phr(L.ALIASES[t]):
            subs.append((a.lower(), "<" + t[:3].upper() + ">"))
        for r in all_phr(L.ROLES[t]):
            if "{ALIAS}" not in r:
                subs.append((r.lower(), "<R-" + t[:3].upper() + ">"))
    for t, items in L.ITEMS.items():
        for i in items:
            subs.append((i.lower(), "<I>"))
    subs.sort(key=lambda x: -len(x[0]))
    return subs


def delex(s, subs):
    s = s.lower()
    for a, rep in subs:
        s = re.sub(r"(?<![\w<])" + re.escape(a) + r"(?![\w>])", rep, s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--top", type=int, default=400)
    a = ap.parse_args()
    _, _, scenes = load_split(a.split)
    subs = build_subs()
    cnt = collections.Counter()
    for s in scenes:
        t = s["mission"]["text"]
        if not VI.search(t):
            continue
        for sent in re.split(r"(?<=[.!?:)])\s+", t):
            cnt[delex(sent, subs)] += 1
    for k, v in cnt.most_common(a.top):
        print(v, k)
    print("unique", len(cnt))


if __name__ == "__main__":
    main()
