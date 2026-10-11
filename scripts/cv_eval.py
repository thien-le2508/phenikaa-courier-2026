"""Graph-accuracy evaluation of the CV reader against scenes.json GT (no policy/NLP dependency).

    python scripts/cv_eval.py --heat work/heat_v3.pt --crops work/crops_v3.pt [--split validation] [--n 300]
        [--dump work/cv_eval_val.json] [--opts '{"edge_rev": false}']

Reports % scenes whose graph is perfect, per-component % and an error breakdown
(missing/extra nodes, wrong edge status/stairs/oneway, missing/extra/wrong landmarks, robot, heading, weather, legend).
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from courier.cv.reader import MapReader  # noqa: E402
from courier.data import DATA, load_split  # noqa: E402


def gold_graph(s):
    return {
        "nodes": [tuple(n["rc"]) for n in s["nodes"]],
        "edges": s["edges"],
        "landmarks": [(lm["type"], tuple(lm["rc"])) for lm in s["landmarks"]],
        "robot": tuple(s["robot"]["rc"]),
        "heading": s["robot"]["heading"],
        "rain": s["weather"] == "rain",
    }


def edge_key(e):
    a, b = tuple(e["a"]), tuple(e["b"])
    ow = tuple(e["oneway_to"]) if e.get("oneway_to") else None
    return (min(a, b), max(a, b)), (e["status"], bool(e["stairs"]), ow)


def compare(g, gold, s=None, aux=None):
    """Returns (per-component ok dict, list of error strings)."""
    errs = []
    pn, gn = set(map(tuple, g["nodes"])), set(gold["nodes"])
    for rc in sorted(gn - pn):
        errs.append(f"node_missing {rc}")
    for rc in sorted(pn - gn):
        errs.append(f"node_extra {rc}")
    ge = dict(edge_key(e) for e in gold["edges"])
    pe = dict(edge_key(e) for e in g["edges"])
    for k in sorted(set(ge) | set(pe)):
        a, b = ge.get(k), pe.get(k)
        if a == b:
            continue
        if a is None:
            errs.append(f"edge_extra {k} {b}")
        elif b is None:
            errs.append(f"edge_missing {k} {a}")
        else:
            if a[0] != b[0]:
                errs.append(f"edge_status {k} {a[0]}->{b[0]}")
            if a[1] != b[1]:
                errs.append(f"edge_stairs {k} {a[1]}->{b[1]}")
            if a[2] != b[2]:
                errs.append(f"edge_oneway {k} {a[2]}->{b[2]}")
    pl, gl = Counter((t, tuple(rc)) for t, rc in g["landmarks"]), Counter(gold["landmarks"])
    gpos = {rc: t for t, rc in gold["landmarks"]}
    ppos = {tuple(rc): t for t, rc in g["landmarks"]}
    for rc in sorted(set(gpos) | set(ppos)):
        a, b = gpos.get(rc), ppos.get(rc)
        if a == b:
            continue
        if a is None:
            errs.append(f"lm_extra {rc} {b}")
        elif b is None:
            errs.append(f"lm_missing {rc} {a}")
        else:
            errs.append(f"lm_type {rc} {a}->{b}")
    if tuple(g["robot"]) != gold["robot"]:
        errs.append(f"robot_rc {gold['robot']}->{tuple(g['robot'])}")
    if g["heading"] != gold["heading"]:
        errs.append(f"robot_heading {gold['heading']}->{g['heading']}")
    if g["rain"] != gold["rain"]:
        errs.append(f"weather {gold['rain']}->{g['rain']}")
    ok = {
        "nodes": pn == gn,
        "edges": ge == pe,
        "edge_status": all(ge.get(k, (None,))[0] == pe.get(k, (None,))[0] for k in set(ge) | set(pe)),
        "edge_stairs": all((ge.get(k) or (0, None))[1] == (pe.get(k) or (0, None))[1] for k in set(ge) | set(pe)),
        "edge_oneway": all((ge.get(k) or (0, 0, None))[2] == (pe.get(k) or (0, 0, None))[2] for k in set(ge) | set(pe)),
        "landmarks": pl == gl,
        "robot": tuple(g["robot"]) == gold["robot"] and g["heading"] == gold["heading"],
        "weather": g["rain"] == gold["rain"],
    }
    if s is not None and aux is not None and "look_to_status" in aux:
        legend_ok = all(aux["look_to_status"].get(s["road_look"].get(st, st)) == st
                        for st in ("normal", "crowded", "covered"))
        ok["legend_map"] = legend_ok
        if not legend_ok:
            errs.append("legend_map")
        gk = sorted(l["kind"].split(":")[-1] for l in s["legend"])
        pk = sorted(k for k in aux.get("legend_kinds", []) if k)
        ok["legend_rows"] = gk == pk  # informational (not part of "graph")
    ok["graph"] = ok["nodes"] and ok["edges"] and ok["landmarks"] and ok["robot"] and ok["weather"]
    return ok, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heat", required=True)
    ap.add_argument("--crops", required=True, nargs="+")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=0, help="first n scenes only (0 = all)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--dump")
    ap.add_argument("--opts", default="{}")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--device", default="cuda", help="cpu works too (~0.7 s/image), no GPU lock needed")
    a = ap.parse_args()
    reader = MapReader(a.heat, a.crops if len(a.crops) > 1 else a.crops[0], opts=json.loads(a.opts), device=a.device)
    _, _, scenes = load_split(a.split)
    idx = list(range(a.start, len(scenes)))
    if a.n:
        idx = idx[:a.n]
    ok = Counter()
    err_types = Counter()
    by_style, by_style_n = Counter(), Counter()
    bad = []
    t0 = time.time()
    for i in idx:
        s = scenes[i]
        g, aux = reader.read(str(DATA / a.split / s["image"]))
        res, errs = compare(g, gold_graph(s), s, aux)
        for k, v in res.items():
            ok[k] += v
        by_style_n[s["style"]] += 1
        by_style[s["style"]] += res["graph"]
        for e in errs:
            err_types[e.split()[0]] += 1
        if not res["graph"]:
            bad.append({"i": i, "style": s["style"], "deg": s["degradation"], "errs": errs})
            if not a.quiet:
                print(i, s["style"], s["degradation"], errs[:6], flush=True)
    n = len(idx)
    print(f"\n== {a.split} n={n} heat={a.heat} crops={a.crops} ({time.time() - t0:.0f}s)")
    print(f"GRAPH PERFECT: {ok['graph'] / n:.4f} ({ok['graph']}/{n})")
    print("per component:", {k: round(v / n, 4) for k, v in ok.items()})
    print("by style:", {st: f"{by_style[st]}/{by_style_n[st]}" for st in sorted(by_style_n)})
    print("error counts:", dict(err_types.most_common()))
    if a.dump:
        json.dump({"n": n, "ok": dict(ok), "errors": dict(err_types), "bad": bad}, open(a.dump, "w"), indent=1,
                  default=str)


if __name__ == "__main__":
    main()
