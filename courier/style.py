"""Small drawing-style classifier used by the learned policy ranker."""
import numpy as np
from PIL import Image


def style_features(image_or_path):
    im = Image.open(image_or_path).convert("RGB") if not isinstance(image_or_path, Image.Image) else image_or_path.convert("RGB")
    small = np.asarray(im.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    flat = small.reshape(-1, 3)
    stats = []
    for c in range(3):
        v = flat[:, c]
        stats.extend((v.mean(), v.std(), *np.quantile(v, (.05, .25, .5, .75, .95))))
        stats.extend(np.histogram(v, bins=16, range=(0, 1), density=True)[0] / 16.0)
    mx, mn = flat.max(1), flat.min(1)
    sat = (mx - mn) / (mx + 1e-6)
    gray = flat.mean(1)
    stats.extend((sat.mean(), sat.std(), gray.mean(), gray.std()))
    # Low-resolution pixels capture palette and line texture while remaining tiny.
    return np.concatenate((small[::2, ::2].reshape(-1), np.asarray(stats, dtype=np.float32)))


def predict_style(model, image_path):
    return str(model.predict(style_features(image_path)[None])[0])
