"""Classify the fine weather icon (sun / sun+cloud / cloud / rain) from the GT weather_box crop (analysis only).
Writes work/policy_weather_<split>.json : {scene_id: label}."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def feats(im, box):
    x0, y0, x1, y1 = box
    a = np.asarray(im.crop((int(x0), int(y0), int(x1), int(y1))).convert('RGB')).astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1); sat = (mx - mn) / (mx + 1e-6)
    yellow = ((r > 180) & (g > 110) & (b < 120) & (sat > 0.35)).mean()
    gray = ((sat < 0.18) & (mx > 120) & (mx < 215)).mean()
    blue = ((b > r + 30) & (b > 120) & (sat > 0.3)).mean()
    return yellow, gray, blue


if __name__ == '__main__':
    split = sys.argv[1]
    S = json.loads((ROOT / 'delivery_public' / split / 'scenes.json').read_text(encoding='utf-8'))
    out = {}
    for s in S:
        im = Image.open(ROOT / 'delivery_public' / split / s['image'])
        out[s['scene_id']] = [s['weather'], *[round(x, 4) for x in feats(im, s['weather_box'])],
                              [l['text'] for l in s['legend'] if l['kind'] == 'weather']]
    (ROOT / 'work' / f'policy_weather_{split}.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
