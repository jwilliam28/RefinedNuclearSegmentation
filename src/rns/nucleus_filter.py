"""Object-level nucleus filter: is a refined object a real nucleus, or a fold, strand, speck, cytoplasm edge or Candida?

H&E-trained detectors run at a low threshold (to catch pale nuclei packed into groups) also outline non-nuclear objects,
most of all on slides with few or pale nuclei. Three kinds of evidence separate them, combined in one logistic model:
  A  detector agreement   was the object found by StarDist (fine pass), StarDist (coarse pass) and InstanSeg?
  B  the object itself    contrast against a 1-3 um ring, chromatin texture, stain colour, shape, edge focus, and size
                          (log size, and how far BELOW a typical nucleus it is, so specks and yeast score low without
                          penalising large dysplastic nuclei)
  C  slide signal         the image's median nuclear OD and the object's darkness relative to it; still computed, but
                          the default model leaves them out (they over-fitted to individual slides)
The default model (models/nucleus_filter_pap_v1.json) was trained on objects graded by a cytopathologist (Pap tests and
urine, two scanners); retrain on your own grades with scripts/train_nucleus_filter.py.
"""
import json
import math
import os
from importlib import resources

import cv2
import numpy as np

FEATURES = ['area_um2', 'log_area', 'log_small2', 'solidity', 'circularity', 'elongation', 'od_in', 'od_ring', 'contrast',
            'contrast_ratio', 'chrom_r', 'chrom_g', 'chrom_b', 'lab_a', 'lab_b', 'lab_L', 'tex_std', 'tex_dog', 'focus_rel',
            'sd_fine', 'sd_coarse', 'instanseg', 'n_detectors']
TYPICAL_AREA_UM2 = 28.0


def with_derived(f):
    """Adds the size features (from area_um2) if a stored feature row lacks them: log size, and the squared log distance
    below a typical nucleus (zero for anything at or above it)."""
    if f is None or 'log_small2' in f or 'area_um2' not in f:
        return f
    la = math.log(max(float(f['area_um2']), 1e-3))
    return dict(f, log_area=la, log_small2=min(0.0, la - math.log(TYPICAL_AREA_UM2)) ** 2)


def _inside(poly, x, y):
    p = np.asarray(poly, float)
    if not (p[:, 0].min() <= x <= p[:, 0].max() and p[:, 1].min() <= y <= p[:, 1].max()):
        return False
    return cv2.pointPolygonTest(p.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def object_features(poly, rgb, origin, um, dets):
    """Per-object features (all but the image-level ones). poly in full-image pixels; rgb the region; dets =
    {'stardist': [...], 'stardist_coarse': [...], 'instanseg': [...]} raw detector outlines of the region."""
    q = np.asarray(poly, float) - np.asarray(origin, float)
    H, W = rgb.shape[:2]
    r1, r3 = max(1, int(round(1 / um))), max(2, int(round(3 / um)))
    x0, y0 = max(0, int(np.floor(q[:, 0].min())) - r3 - 2), max(0, int(np.floor(q[:, 1].min())) - r3 - 2)
    x1, y1 = min(W, int(np.ceil(q[:, 0].max())) + r3 + 3), min(H, int(np.ceil(q[:, 1].max())) + r3 + 3)
    sub = rgb[y0:y1, x0:x1]
    qq = q - [x0, y0]
    m = np.zeros(sub.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(qq).astype(np.int32)], 1)
    mb = m.astype(bool)
    if mb.sum() < 5:
        return None
    ring = cv2.dilate(m, np.ones((2 * r3 + 1,) * 2, np.uint8)).astype(bool) & ~cv2.dilate(m, np.ones((2 * r1 + 1,) * 2, np.uint8)).astype(bool)
    o = -np.log(np.clip(sub.astype(np.float32), 1, 255) / 255.0)
    odm = np.minimum(o[..., 0], o[..., 1])
    s = o.sum(2) + 1e-3
    lb = cv2.cvtColor(np.ascontiguousarray(sub.astype(np.uint8)), cv2.COLOR_RGB2LAB).astype(np.float32)
    sig = 0.5 / um
    dog = cv2.GaussianBlur(odm, (0, 0), sig * 0.5) - cv2.GaussianBlur(odm, (0, 0), sig)
    cnt = np.round(qq).astype(np.int32).reshape(-1, 1, 2)
    area_px = float(mb.sum())
    per = cv2.arcLength(cnt, True)
    el = 1.0
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        el = max(MA, ma) / max(min(MA, ma), 1e-6)
    cx, cy = np.asarray(poly, float).mean(0)
    agree = {k: int(any(_inside(p, cx, cy) for p in dets.get(k, []))) for k in ('stardist', 'stardist_coarse', 'instanseg')}
    od_in = float(odm[mb].mean())
    od_ring = float(odm[ring].mean()) if ring.any() else float('nan')
    return with_derived(dict(area_um2=area_px * um * um, solidity=area_px / max(cv2.contourArea(cv2.convexHull(cnt)), 1e-6),
                circularity=4 * math.pi * area_px / max(per * per, 1e-6), elongation=el, od_in=od_in, od_ring=od_ring,
                contrast=od_in - od_ring, contrast_ratio=od_in / max(od_ring, 1e-3),
                chrom_r=float((o[..., 0] / s)[mb].mean()), chrom_g=float((o[..., 1] / s)[mb].mean()), chrom_b=float((o[..., 2] / s)[mb].mean()),
                lab_a=float(lb[..., 1][mb].mean() - 128), lab_b=float(lb[..., 2][mb].mean() - 128), lab_L=float(lb[..., 0][mb].mean()),
                tex_std=float(odm[mb].std()), tex_dog=float(dog[mb].std()),
                sd_fine=agree['stardist'], sd_coarse=agree['stardist_coarse'], instanseg=agree['instanseg'], n_detectors=sum(agree.values())))


def add_image_features(feature_rows):
    """Slide signal (C): the image's median od_in over its objects, and each object's od_in relative to it."""
    vals = [f['od_in'] for f in feature_rows if f is not None]
    med = float(np.median(vals)) if vals else float('nan')
    for f in feature_rows:
        if f is not None:
            f['image_median_od'] = med
            f['hyper_rel'] = f['od_in'] / med if med and np.isfinite(med) else float('nan')
    return med


class NucleusFilter:
    """Standardised logistic model over FEATURES; NaN features are replaced by the training medians. With
    split_area_um2 set, objects at least that large are scored by a second logistic model (`large`): large dysplastic
    nuclei and large non-nuclear objects (folds, clumps) are told apart by different evidence than small ones."""

    def __init__(self, coef, intercept, mean, scale, fill, threshold, info=None, features=None, large=None, split_area_um2=None):
        self.features = list(features or FEATURES)
        self.coef, self.intercept = np.asarray(coef, float), float(intercept)
        self.mean, self.scale, self.fill = np.asarray(mean, float), np.asarray(scale, float), np.asarray(fill, float)
        self.threshold = float(threshold)
        self.info = info or {}
        self.large = {k: (np.asarray(v, float) if k != 'intercept' else float(v)) for k, v in large.items()} if large else None
        self.split_area_um2 = float(split_area_um2) if split_area_um2 else None

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = str(resources.files('rns').joinpath('models', 'nucleus_filter_pap_v1.json'))
        d = json.load(open(path))
        return cls(d['coef'], d['intercept'], d['mean'], d['scale'], d['fill'], d['threshold'], d.get('info'), d.get('features'),
                   d.get('large'), d.get('split_area_um2'))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = dict(features=self.features, coef=self.coef.tolist(), intercept=self.intercept, mean=self.mean.tolist(),
                 scale=self.scale.tolist(), fill=self.fill.tolist(), threshold=self.threshold, info=self.info)
        if self.large:
            d.update(split_area_um2=self.split_area_um2,
                     large={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.large.items()})
        json.dump(d, open(path, 'w'), indent=1)

    def matrix(self, feature_rows):
        rows = [with_derived(f) for f in feature_rows]
        return np.array([[f.get(k, float('nan')) if f is not None else float('nan') for k in self.features] for f in rows], float)

    @staticmethod
    def _lin(X, coef, intercept, mean, scale, fill):
        Z = (np.where(np.isfinite(X), X, fill) - mean) / scale
        return 1.0 / (1.0 + np.exp(-(Z @ coef + intercept)))

    def proba(self, feature_rows):
        if not len(feature_rows):
            return np.zeros(0)
        X = self.matrix(feature_rows)
        p = self._lin(X, self.coef, self.intercept, self.mean, self.scale, self.fill)
        if self.large:
            a = X[:, self.features.index('area_um2')]
            big = np.isfinite(a) & (a >= self.split_area_um2)
            if big.any():
                L = self.large
                p[big] = self._lin(X[big], L['coef'], L['intercept'], L['mean'], L['scale'], L['fill'])
        return p

    @staticmethod
    def _fit(X, y, C):
        from sklearn.linear_model import LogisticRegression
        fill = np.nanmedian(X, axis=0)
        X = np.where(np.isfinite(X), X, fill)
        mean, scale = X.mean(0), X.std(0) + 1e-9
        m = LogisticRegression(C=C, max_iter=5000, class_weight='balanced').fit((X - mean) / scale, np.asarray(y, int))
        return dict(coef=m.coef_[0], intercept=float(m.intercept_[0]), mean=mean, scale=scale, fill=fill)

    @classmethod
    def train(cls, feature_rows, labels, threshold=0.5, C=0.5, info=None, features=None, split_area_um2=None):
        features = list(features or FEATURES)
        X = np.array([[f.get(k, float('nan')) for k in features] for f in map(with_derived, feature_rows)], float)
        y = np.asarray(labels, int)
        if not split_area_um2:
            s = cls._fit(X, y, C)
            return cls(s['coef'], s['intercept'], s['mean'], s['scale'], s['fill'], threshold, info, features)
        big = X[:, features.index('area_um2')] >= split_area_um2
        s, L = cls._fit(X[~big], y[~big], C), cls._fit(X[big], y[big], C)
        return cls(s['coef'], s['intercept'], s['mean'], s['scale'], s['fill'], threshold, info, features, L, split_area_um2)
