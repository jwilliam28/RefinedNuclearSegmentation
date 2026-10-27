"""Refinement of StarDist / InstanSeg nuclei on Papanicolaou-stained (and other brightfield) images.

H&E-trained detectors find Pap nuclei well but draw their outlines loosely: the cyan cytoplasm rim is often included,
pale parts of a nucleus are cut off, one nucleus is sometimes split into fragments and defocused parts of thick groups
come out as large smooth blobs. The steps here, applied per image region, are

1. union      StarDist (fine pass) plus InstanSeg nuclei that StarDist missed;
2. consolidate fragments that one compact coarse-pass StarDist nucleus covers;
3. classify   pixels as nucleus vs cytoplasm-or-glass with a per-image logistic regression on stain colour and texture,
              trained on the detections themselves (no manual labels);
4. boundary   per nucleus, the search area is the detector outline plus a margin (optionally grown along the nucleus
              probability, Params.grow_um, off by default; never into another nucleus); the darkness outline
              (nuclear optical density (OD) above an Otsu threshold estimated on the nucleus plus a surround ring,
              glass left out) is used first; where it fails, 128 rays coming inward from the edge stop at the first
              sustained nucleus probability > 0.5 and snap to the steepest rise of OD (Params.darkness_preference);
5. clean up   debris filters, non-overlapping ownership (optional merging of touching pieces, off by default);
6. smooth     Fourier smoothing with a fixed physical cutoff (2 um), the same on every scanner;
7. focus      edge gradient of glass-normalised luminance per nucleus, divided by the same field's sharp-nucleus
              reference: sharp >= 0.75, intermediate, blurred < 0.50. A flag, not a filter: the gradient also
              depends on edge contrast, so nuclei on dense cytoplasm or deep in crowded groups can read as softer.

Nuclear OD is min(OD_red, OD_green): nuclei absorb red and green, cyan cytoplasm absorbs red only, pink cytoplasm
green only, so the minimum is high only in nuclei.
"""
import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label as sklabel, regionprops

from .params import Params


# ---------------------------------------------------------------- pixel features

def optical_density(rgb):
    f = np.clip(rgb.astype(np.float32), 1, 255) / 255.0
    return -np.log(f)


def nuclear_od(rgb, um):
    """min(OD_R, OD_G), lightly smoothed (sigma 0.3 um)."""
    o = optical_density(rgb)
    return cv2.GaussianBlur(np.minimum(o[..., 0], o[..., 1]), (0, 0), max(0.7, 0.3 / um))


def features(rgb, um):
    o = optical_density(rgb)
    s = o.sum(2) + 1e-3
    lb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    odm = np.minimum(o[..., 0], o[..., 1])
    sig = max(1.0, 0.5 / um)
    mu = cv2.GaussianBlur(odm, (0, 0), sig)
    sd = np.sqrt(np.maximum(cv2.GaussianBlur(odm * odm, (0, 0), sig) - mu * mu, 0))
    return np.dstack([o[..., 0], o[..., 1], o[..., 2], o[..., 0] / s, o[..., 1] / s, o[..., 2] / s, s,
                      lb[..., 1] - 128, lb[..., 2] - 128, odm, sd])


def poly_mask(poly, shape, origin):
    q = np.asarray(poly, float) - np.asarray(origin, float)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    return m.astype(bool)


def poly_mask_local(poly, shape, origin, pad=0):
    """The pixels poly_mask sets, as (mask, y0, x0): a window of the frame around the polygon's bounding box grown by
    `pad` pixels (clipped to the frame). Same rasterisation as poly_mask (integer shift), without a full-frame array."""
    q = np.round(np.asarray(poly, float) - np.asarray(origin, float)).astype(np.int32)
    H, W = shape
    x0, y0 = max(0, int(q[:, 0].min()) - pad), max(0, int(q[:, 1].min()) - pad)
    x1, y1 = min(W, int(q[:, 0].max()) + pad + 1), min(H, int(q[:, 1].max()) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), bool), 0, 0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [q - np.array([x0, y0], np.int32)], 1)
    return m.astype(bool), y0, x0


def _sl(lm):
    m, y0, x0 = lm
    return slice(y0, y0 + m.shape[0]), slice(x0, x0 + m.shape[1])


def _crop(lm, wy, wx):
    """A local mask restricted to the frame window (wy, wx)."""
    m, y0, x0 = lm
    out = np.zeros((wy.stop - wy.start, wx.stop - wx.start), bool)
    ya, yb = max(wy.start, y0), min(wy.stop, y0 + m.shape[0])
    xa, xb = max(wx.start, x0), min(wx.stop, x0 + m.shape[1])
    if ya < yb and xa < xb:
        out[ya - wy.start:yb - wy.start, xa - wx.start:xb - wx.start] = m[ya - y0:yb - y0, xa - x0:xb - x0]
    return out


def _overlap(a, b):
    """Number of pixels two local masks share."""
    (ma, ay, ax), (mb, by, bx) = a, b
    ya, yb = max(ay, by), min(ay + ma.shape[0], by + mb.shape[0])
    xa, xb = max(ax, bx), min(ax + ma.shape[1], bx + mb.shape[1])
    if ya >= yb or xa >= xb:
        return 0
    return int((ma[ya - ay:yb - ay, xa - ax:xb - ax] & mb[ya - by:yb - by, xa - bx:xb - bx]).sum())


def _local_window(lm, pad, shape):
    """What _window(full mask, pad) returns, from a local mask whose window already extends >= pad past the polygon:
    (wy, wx, mask in that window)."""
    m, y0, x0 = lm
    ys, xs = np.nonzero(m)
    wy = slice(max(0, y0 + ys.min() - pad), min(shape[0], y0 + ys.max() + pad + 1))
    wx = slice(max(0, x0 + xs.min() - pad), min(shape[1], x0 + xs.max() + pad + 1))
    return wy, wx, m[wy.start - y0:wy.stop - y0, wx.start - x0:wx.stop - x0]


def _iters(um_dist, um, minimum=1):
    return max(minimum, int(round(um_dist / um)))


# ---------------------------------------------------------------- classifier

class NucleusClassifier:
    """Per-image two-class logistic regression: nucleus cores (1) vs surrounding cytoplasm and glass pooled (0)."""

    def __init__(self, params=None):
        self.p = params or Params()
        self.clf = self.mu = self.sd = None
        self.train_accuracy = None

    def fit(self, regions, um):
        """regions: list of (rgb, polys, origin) with polys in full-image pixel coordinates."""
        p = self.p
        X, y = [], []
        rng = np.random.default_rng(0)
        for rgb, polys, origin in regions:
            F = features(rgb, um)
            shp = rgb.shape[:2]
            allm = np.zeros(shp, bool)
            core = np.zeros(shp, bool)
            for q in polys:
                lm = poly_mask_local(q, shp, origin, 1)
                m, sl = lm[0], _sl(lm)
                allm[sl] |= m
                if m.sum() > 20:                       # inner half of every instance: pale nuclei included, rims never
                    dt = ndi.distance_transform_edt(m)
                    core[sl] |= dt >= 0.5 * dt.max()
            d_in = ndi.binary_dilation(allm, iterations=_iters(p.cyto_ring_um[0], um))
            d_out = ndi.binary_dilation(allm, iterations=_iters(p.cyto_ring_um[1], um, 2))
            luma = rgb.mean(2)
            glass = luma > np.percentile(luma, 98) * p.glass_luma_frac
            cyto = d_out & ~d_in & ~glass
            for lab, msk in ((1, core), (0, cyto), (0, glass)):
                idx = np.flatnonzero(msk.ravel())
                if idx.size == 0:
                    continue
                idx = rng.choice(idx, min(idx.size, p.samples_per_class), replace=False)
                X.append(F.reshape(-1, F.shape[2])[idx])
                y.append(np.full(idx.size, lab))
        if not X or len(set(np.concatenate(y).tolist())) < 2:
            raise ValueError('not enough nuclei or background to train the colour classifier')
        X = np.vstack(X)
        y = np.concatenate(y)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.clf = LogisticRegression(max_iter=500, class_weight='balanced').fit((X - self.mu) / self.sd, y)
        self.train_accuracy = float((self.clf.predict((X - self.mu) / self.sd) == y).mean())
        return self

    def prob(self, rgb, um):
        F = features(rgb, um)
        pr = self.clf.predict_proba((F.reshape(-1, F.shape[2]) - self.mu) / self.sd)[:, 1]
        return pr.reshape(rgb.shape[:2])


# ---------------------------------------------------------------- per-nucleus outlines

def _window(m, pad):
    ys, xs = np.nonzero(m)
    return (slice(max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1)),
            slice(max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1)))


def reach_mask(m0, others, ps, od, um, p):
    """Where the boundary search may look. Detectors trained on round H&E nuclei often stop short of a large or
    irregular Pap nucleus, so the search area is the detector outline + ray_margin_um, extended along the nucleus
    probability (ps > 0.5, connected to the detector outline) up to grow_um outside it - never into another detected
    nucleus, and only through pixels at least 0.6 x as dark (nuclear OD) as the detector outline's core, so dense
    cytoplasm that the colour classifier also calls nuclear (about 0.3-0.4 x) is not taken. The growth is kept only if the new part is closed off by cytoplasm / glass (p <= 0.5) along most of its
    new border: in dense groups the dark material between nuclei also has p > 0.5 and the growth runs into the
    neighbours or out to the grow limit, so there the detector outline + margin is kept.
    Returns (reach, grown nucleus-probability component or the detector mask)."""
    base = ndi.binary_dilation(m0, iterations=_iters(p.ray_margin_um, um))
    comp = m0.copy()
    if p.grow_um > 0:
        full = ndi.binary_dilation(m0, iterations=_iters(p.grow_um, um))
        dt = ndi.distance_transform_edt(m0)
        dark = od >= p.grow_min_od_frac * float(np.median(od[dt >= 0.5 * dt.max()]))
        lab = sklabel(((ps > 0.5) & dark & full & ~others) | m0)
        ids = np.unique(lab[m0])
        grown = ndi.binary_fill_holes(np.isin(lab, ids[ids > 0])) & ~others
        ring = ndi.binary_dilation(grown) & ~grown & ~base       # the border of the part beyond the normal margin
        if ring.any():
            leak = int((ring & ~full).sum())
            neighbour = int((ring & others).sum())
            if ring.sum() - leak - neighbour >= p.grow_min_closed * ring.sum():
                comp = grown
    reach = base | ndi.binary_dilation(comp, iterations=_iters(1.0, um))
    return reach & ~others, comp


def radial_outline(ps, m0, reach, odm_s, um, p):
    """Outside-in radial boundary: 128 rays from the outer edge of `reach` move inward, stop at the first sustained
    nucleus probability > 0.5 and snap to the steepest rise of nuclear OD; circular median + Gaussian.
    All arrays share one (window) frame; returns (x, y) points in that frame."""
    shp = ps.shape
    cy, cx = ndi.center_of_mass(m0)
    K = p.rays
    step = p.ray_step_um / um
    sustain = int(math.ceil(p.ray_sustain_um / p.ray_step_um))
    w = int(math.ceil(p.ray_snap_um / p.ray_step_um))
    radii = np.full(K, np.nan)
    for k in range(K):
        th = 2 * math.pi * k / K
        dx, dy = math.cos(th), math.sin(th)
        r = 0.0
        while True:                                     # outer limit of the search area along this ray
            xx, yy = int(round(cx + dx * (r + step))), int(round(cy + dy * (r + step)))
            if not (0 <= xx < shp[1] and 0 <= yy < shp[0]) or not reach[yy, xx]:
                break
            r += step
        rs = np.arange(r, 0, -step)
        if rs.size < 2:
            radii[k] = r
            continue
        xs = np.clip(np.round(cx + dx * rs).astype(int), 0, shp[1] - 1)
        ys = np.clip(np.round(cy + dy * rs).astype(int), 0, shp[0] - 1)
        above = ps[ys, xs] > 0.5
        hit = next((i for i in range(len(above) - sustain) if above[i:i + sustain].all()), None)
        if hit is None:
            continue                                     # filled from neighbouring rays below
        lo, hi = max(0, hit - w), min(len(rs) - 1, hit + w)
        g = np.gradient(odm_s[ys, xs])[lo:hi + 1]       # coming inward, nuclear OD rises at the membrane
        radii[k] = rs[lo + int(np.argmax(g))]
    ok = np.isfinite(radii) & (radii > 0)
    if ok.sum() < K // 2:
        return None
    idx = np.arange(K)
    radii = np.interp(idx, idx[ok], radii[ok], period=K)
    rr = np.r_[radii[-2:], radii, radii[:2]]
    med = np.array([np.median(rr[i:i + 5]) for i in range(K)])
    med = ndi.gaussian_filter1d(med, 1.5, mode='wrap')
    th = 2 * math.pi * idx / K
    return np.c_[cx + np.cos(th) * med, cy + np.sin(th) * med]


def darkness_outline(m0, comp, others, odm_s, um, p=None):
    """Darkness outline: pixels of (detector outline + grown component) dilated 1 um whose nuclear OD is above an Otsu
    threshold; the component overlapping the detector outline most, closed, filled, smoothed. Returns a mask in the
    window frame. The threshold is estimated on a WIDER sample - the same region plus a surround ring of
    max(1 um, darkness_ring_frac x the detector outline's equivalent radius), glass (OD <= darkness_glass_od) left out -
    so that it falls between nucleus and cytoplasm: estimated on the 1 um region alone, a large nucleus fills most of
    the sample and the threshold split it into its darker and paler chromatin (large LSIL nuclei lost up to half
    their area, 2026-10-26)."""
    p = p or Params()
    reg = ndi.binary_dilation(m0 | comp, iterations=_iters(1.0, um)) & ~others
    req = math.sqrt(m0.sum() / math.pi) * um
    wide = ndi.binary_dilation(m0 | comp, iterations=_iters(max(1.0, p.darkness_ring_frac * req), um)) & ~others
    v = odm_s[wide & (odm_s > p.darkness_glass_od)]
    if v.size < 30:
        return None
    try:
        t = threshold_otsu(v)
    except ValueError:
        return None
    cl = sklabel(reg & (odm_s > t))
    if cl.max() == 0:
        return None
    ov = np.bincount(cl[m0].ravel(), minlength=cl.max() + 1)[1:]
    if ov.max() == 0:
        return None
    m = cl == (np.argmax(ov) + 1)
    m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=_iters(0.5, um)))
    return ndi.gaussian_filter(m.astype(float), max(1.0, 0.5 / um)) > 0.5


def mask_contour(m, origin):
    cs = find_contours(np.pad(m, 1).astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    return np.c_[c[:, 1] - 1, c[:, 0] - 1] + np.asarray(origin, float)


def edge_gradient(odm_s):
    gy, gx = np.gradient(odm_s)
    return np.hypot(gx, gy)


def edge_score(poly, odm_s, origin, g=None):
    g = edge_gradient(odm_s) if g is None else g
    q = np.round(np.asarray(poly, float) - np.asarray(origin, float)).astype(int)
    q[:, 0] = np.clip(q[:, 0], 0, g.shape[1] - 1)
    q[:, 1] = np.clip(q[:, 1], 0, g.shape[0] - 1)
    return float(g[q[:, 1], q[:, 0]].mean())


def seed_outline(poly, occ, ps, odm_s, origin, um, p, g=None):
    """Refined outline of one detected nucleus (full-image coordinates), or None. occ counts how many seeds cover
    each pixel of the region (used to keep the search out of neighbouring nuclei); g = edge_gradient(odm_s), passed
    in so that it is computed once per region."""
    pad = _iters(p.grow_um + p.ray_margin_um + 2.0, um)
    lm = poly_mask_local(poly, ps.shape, origin, pad)
    if lm[0].sum() < 10:
        return None
    wy, wx, m0 = _local_window(lm, pad, ps.shape)
    others = ((occ[wy, wx] - m0.astype(np.int32)) > 0) & ~m0
    ps_w, od_w = ps[wy, wx], odm_s[wy, wx]
    reach, comp = reach_mask(m0, others, ps_w, od_w, um, p)
    o = (origin[0] + wx.start, origin[1] + wy.start)
    a = radial_outline(ps_w, m0, reach, od_w, um, p)
    a = a + np.asarray(o, float) if a is not None and len(a) > 5 else None
    bm = darkness_outline(m0, comp, others, od_w, um, p)
    b = mask_contour(bm, o) if bm is not None else None
    b = b if b is not None and len(b) > 5 else None
    if a is not None and b is not None:          # the smoother radial outline unless the darkness edge is clearly stronger
        g = edge_gradient(odm_s) if g is None else g
        return b if edge_score(b, odm_s, origin, g) >= p.darkness_preference * edge_score(a, odm_s, origin, g) else a
    return a if a is not None else b


# ---------------------------------------------------------------- union, consolidation, clean-up

def agreed_seeds(stardist, instanseg):
    """The StarDist outlines whose centre lies inside an InstanSeg outline: objects both detectors find (on Pap, objects
    only one detector finds were mostly folds, cell edges and debris - cytopathologist-graded random objects)."""
    if not instanseg:
        return []
    boxes = [(np.asarray(q, float).min(0), np.asarray(q, float).max(0)) for q in instanseg]
    out = []
    for q in stardist:
        c = np.asarray(q, float).mean(0)
        for (lo, hi), qi in zip(boxes, instanseg):
            if lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1] and                     cv2.pointPolygonTest(np.asarray(qi, np.float32).reshape(-1, 1, 2), (float(c[0]), float(c[1])), False) >= 0:
                out.append(q)
                break
    return out


def union_detections(stardist, instanseg, shape, origin, p):
    union = list(stardist)
    cover = np.zeros(shape, bool)
    for q in stardist:
        lm = poly_mask_local(q, shape, origin)
        cover[_sl(lm)] |= lm[0]
    added = 0
    for q in instanseg:
        lm = poly_mask_local(q, shape, origin)
        mi, sl = lm[0], _sl(lm)
        if mi.sum() and (mi & cover[sl]).sum() / mi.sum() < p.instanseg_add_max_cover:
            union.append(q)
            cover[sl] |= mi
            added += 1
    return union, added


def _no_gap(pieces, region, odm_s, ps, um, p):
    """True if the material between the pieces (pixels of `region` within 1 um of two different pieces) is as
    nuclear as the pieces themselves: nuclear OD >= coarse_no_gap x the paler piece's mean OD and mean nucleus
    probability > 0.5. Two separate nuclei meet along a paler line; one nucleus cut by a detector (often along a
    fold or groove, which is darker) does not."""
    d = _iters(1.0, um)
    near = np.zeros(region.shape, np.int32)
    for m in pieces:
        near += ndi.binary_dilation(m, iterations=d)
    cut = region & (near >= 2)
    if cut.sum() < 5:
        return False
    paler = min(float(odm_s[m].mean()) for m in pieces)
    return float(odm_s[cut].mean()) >= p.coarse_no_gap * paler and float(ps[cut].mean()) > 0.5


def consolidate(union, coarse, shape, origin, p, odm_s=None, ps=None, um=None):
    """Replace >= 2 fine pieces by one coarse StarDist instance when it is compact (solidity >= 0.90) and covers each
    piece, and either one piece dominates (a big piece + fragments) or the coarse outline is not elongated
    (major/minor <= 1.4) and there is no pale gap between the pieces (a nucleus cut along a fold). Similar pieces
    separated by a paler line, or side by side in an elongated outline, are touching nuclei and stay separate."""
    fm = [poly_mask_local(q, shape, origin) for q in union]           # local masks: only overlapping windows are compared
    fsum = [int(lm[0].sum()) for lm in fm]
    used, out = set(), []
    for cp in coarse:
        clm = poly_mask_local(cp, shape, origin)
        cm = clm[0]
        csum = int(cm.sum())
        if csum < 20:
            continue
        rp = regionprops(cm.astype(np.uint8))
        if not rp or rp[0].solidity < p.coarse_min_solidity:
            continue
        inside = [i for i in range(len(fm)) if i not in used and fsum[i] and _overlap(fm[i], clm) / fsum[i] >= p.coarse_min_inside]
        if len(inside) < 2:
            continue
        dominant = max(fsum[i] for i in inside) >= p.coarse_dominant_frac * csum
        # the no-gap path also needs an ordinary shape: two nuclei side by side make an elongated union (1.7 in testing),
        # one nucleus cut along a fold does not (1.1-1.2); overlapping nuclei are darker where they overlap, so the
        # no-gap test alone cannot tell them from a fold
        elong = rp[0].axis_major_length / max(rp[0].axis_minor_length, 1e-6)
        continuous = False
        if p.coarse_no_gap > 0 and odm_s is not None and ps is not None and elong <= p.coarse_no_gap_max_elong:
            # one window holding the coarse outline and its pieces, + the 1 um dilation of _no_gap and a margin
            d = _iters(1.0, um) + 1
            sls = [_sl(clm)] + [_sl(fm[i]) for i in inside]
            wy = slice(max(0, min(a.start for a, _ in sls) - d), min(shape[0], max(a.stop for a, _ in sls) + d))
            wx = slice(max(0, min(b.start for _, b in sls) - d), min(shape[1], max(b.stop for _, b in sls) + d))
            continuous = _no_gap([_crop(fm[i], wy, wx) for i in inside], _crop(clm, wy, wx), odm_s[wy, wx], ps[wy, wx], um, p)
        if dominant or continuous:
            used.update(inside)
            out.append(cp)
    return out + [q for i, q in enumerate(union) if i not in used]


def _bbox(m):
    ys, xs = np.nonzero(m)
    return ys.min(), ys.max(), xs.min(), xs.max()


def _bbox_local(lm):
    m, y0, x0 = lm
    ys, xs = np.nonzero(m)
    return y0 + ys.min(), y0 + ys.max(), x0 + xs.min(), x0 + xs.max()


def _merge_pieces(keep, odm_s, p):
    """Two touching pieces become one nucleus if (halves) the union is compact, not elongated and of ordinary size,
    or (fragment) one piece dominates, the union is compact and chromatin continues across the cut.
    Only pairs whose bounding boxes lie within 2 px are tested, each in a small window (same result as testing every
    pair on the full frame, at a fraction of the cost in crowded fields). keep: local masks (mask, y0, x0)."""
    if len(keep) < 2:
        return keep
    med = float(np.median([lm[0].sum() for lm in keep]))
    H, W = odm_s.shape
    boxes = [_bbox_local(lm) for lm in keep]
    changed = True
    while changed:
        changed = False
        for i in range(len(keep)):
            bi = boxes[i]
            for j in range(i + 1, len(keep)):
                bj = boxes[j]
                if bi[0] > bj[1] + 2 or bj[0] > bi[1] + 2 or bi[2] > bj[3] + 2 or bj[2] > bi[3] + 2:
                    continue
                win = (slice(max(0, min(bi[0], bj[0]) - 3), min(H, max(bi[1], bj[1]) + 4)),
                       slice(max(0, min(bi[2], bj[2]) - 3), min(W, max(bi[3], bj[3]) + 4)))
                a, b = _crop(keep[i], *win), _crop(keep[j], *win)
                od = odm_s[win]
                border = ndi.binary_dilation(a) & ndi.binary_dilation(b)
                if border.sum() < 0.3 * 2 * math.sqrt(min(a.sum(), b.sum()) / math.pi):
                    continue
                u = ndi.binary_fill_holes(a | b | border)
                rp = regionprops(u.astype(np.uint8))
                if not rp:
                    continue
                r0 = rp[0]
                elong = r0.axis_major_length / max(r0.axis_minor_length, 1e-6)
                halves = (r0.solidity >= p.halves_min_solidity and elong <= p.halves_max_elongation
                          and u.sum() <= p.halves_max_area_vs_median * med)
                dom = max(a.sum(), b.sum()) / max(u.sum(), 1)
                no_gap = od[border].mean() >= p.fragment_no_gap * min(od[a].mean(), od[b].mean())
                fragment = dom >= p.fragment_dominant_frac and r0.solidity >= p.fragment_min_solidity and no_gap
                if halves or fragment:
                    keep[i] = (u, win[0].start, win[1].start)
                    boxes[i] = _bbox_local(keep[i])
                    keep.pop(j)
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return keep


def resolve(polys, pmap, origin, um, odm_s, p):
    """Debris filters, merging of pieces, non-overlapping ownership (nearest centroid); sub-pixel contours."""
    shp = pmap.shape
    r_in, r_out = _iters(1.0, um), _iters(2.5, um, 2)
    allm = [poly_mask_local(q, shp, origin, r_out + 2) for q in polys]
    occ = np.zeros(shp, np.int32)
    for lm in allm:
        occ[_sl(lm)] += lm[0]
    keep = []
    for lm in allm:
        m = lm[0]
        if m.sum() * um * um < p.min_area_um2 or pmap[_sl(lm)][m].mean() < p.min_mean_prob:
            continue
        rp = regionprops(m.astype(np.uint8))
        if not rp or rp[0].solidity < p.min_solidity:
            continue
        wy, wx, mw = _local_window(lm, r_out + 2, shp)       # the surround test in a window around the nucleus
        ow = odm_s[wy, wx]
        ring = ndi.binary_dilation(mw, iterations=r_out) & ~ndi.binary_dilation(mw, iterations=r_in)
        ring &= ow > 0.08                                     # ignore glass in the surround
        ring &= (occ[wy, wx] - mw.astype(np.int32)) == 0      # ignore neighbouring nuclei
        if ring.sum() > max(10, 0.3 * mw.sum()) and ow[mw].mean() < p.min_contrast * ow[ring].mean():
            continue
        keep.append(lm)
    if p.merge_pieces:
        keep = _merge_pieces(keep, odm_s, p)
    if not keep:
        return []
    # ownership: pixels covered once go to that nucleus, overlaps to the nearest centroid (ties: the lower index)
    cent = [np.asarray(ndi.center_of_mass(lm[0])) + [lm[1], lm[2]] for lm in keep]
    cover = np.zeros(shp, np.int32)
    for lm in keep:
        cover[_sl(lm)] += lm[0]
    lab = np.zeros(shp, np.int32)
    best = None
    for i, lm in enumerate(keep, 1):
        m, y0, x0 = lm
        sl = _sl(lm)
        cw, lw = cover[sl], lab[sl]
        lw[m & (cw == 1)] = i
        yy, xx = np.nonzero(m & (cw >= 2))
        if yy.size:
            best = np.full(shp, np.inf) if best is None else best
            d = np.hypot(cent[i - 1][0] - (yy + y0), cent[i - 1][1] - (xx + x0))
            bw = best[sl]
            upd = d < bw[yy, xx]
            bw[yy[upd], xx[upd]] = d[upd]
            lw[yy[upd], xx[upd]] = i
    out = []
    for i, lm in enumerate(keep, 1):
        sl = _sl(lm)
        m = lab[sl] == i
        if m.sum() * um * um < p.min_area_um2:
            continue
        cl = sklabel(m)
        if cl.max() > 1:
            m = cl == (np.argmax(np.bincount(cl.ravel())[1:]) + 1)
        c = mask_contour(m, (origin[0] + sl[1].start, origin[1] + sl[0].start))
        if c is not None:
            out.append(c)
    return out


# ---------------------------------------------------------------- smoothing and focus

def fourier_smooth(poly, um, wavelength_um=2.0, n=256):
    """Resample the closed contour by arc length and damp its Fourier harmonics with a Gaussian taper that halves the
    harmonic whose wavelength is `wavelength_um`; folds and notches wider than about 3 um survive."""
    q = np.asarray(poly, float)
    if len(q) < 8:
        return q
    qq = np.r_[q, q[:1]]
    d = np.hypot(*np.diff(qq, axis=0).T)
    per = d.sum()
    if per <= 0:
        return q
    s = np.r_[0, np.cumsum(d)]
    t = np.linspace(0, per, n, endpoint=False)
    Z = np.fft.fft(np.interp(t, s, qq[:, 0]) + 1j * np.interp(t, s, qq[:, 1]))
    from scipy.spatial import ConvexHull
    try:                                                # cutoff from the hull perimeter: jagged edges inflate the arc length
        per_ref = ConvexHull(q).area
    except Exception:
        per_ref = per
    kh = max(4.0, per_ref * um / wavelength_um)
    k = np.fft.fftfreq(n, 1.0 / n)
    Z *= np.exp(-math.log(2) * (k / kh) ** 2)
    z = np.fft.ifft(Z)
    return np.c_[z.real, z.imag]


def focus_measure(polys, rgb, origin, um, p):
    """Per-nucleus focus measure (FM): 90th percentile of the gradient magnitude (per um) of glass-normalised
    luminance, pre-smoothed with a 0.5 um Gaussian (a scale both 20x and 40x optics resolve), in a +/- 1 um band
    around the outline. Sharpening multiplies gradients by a roughly constant gain, so FM is only compared with
    the same image's own nuclei (see focus_reference)."""
    L = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    L /= max(float(np.percentile(L, 99)), 1.0)
    Ls = cv2.GaussianBlur(L, (0, 0), p.focus_smooth_um / um)
    g = np.hypot(cv2.Scharr(Ls, cv2.CV_32F, 1, 0), cv2.Scharr(Ls, cv2.CV_32F, 0, 1)) / 32.0 / um
    e = _iters(p.focus_band_um, um)
    out = []
    for q in polys:
        lm = poly_mask_local(q, rgb.shape[:2], origin, e + 2)
        if lm[0].sum() < 10:
            out.append(0.0)
            continue
        wy, wx, mw = _local_window(lm, e + 2, rgb.shape[:2])
        band = ndi.binary_dilation(mw, iterations=e) & ~ndi.binary_erosion(mw, iterations=e)
        out.append(float(np.percentile(g[wy, wx][band], 90)))
    return out


def focus_reference(fm_by_region, p):
    """Reference FM per region (field): the 75th percentile of its nuclei's FM when it has >= 5 nuclei, otherwise the
    image-level value; floored at 0.60 x the image's median field reference so that a field whose whole plane is out
    of focus is not normalised away."""
    allv = [f for fm in fm_by_region for f in fm]
    img_ref = float(np.percentile(allv, p.focus_ref_percentile)) if allv else 0.0
    field = [float(np.percentile(fm, p.focus_ref_percentile)) if len(fm) >= p.focus_min_field_nuclei else None
             for fm in fm_by_region]
    typical = float(np.median([f for f in field if f is not None])) if any(f is not None for f in field) else img_ref
    return [max(f if f is not None else img_ref, p.focus_floor * typical) for f in field], img_ref


def focus_states(fm, areas, ref, p):
    """'sharp' (FM/ref >= 0.75), 'intermediate' (0.50-0.75) or 'blurred' (< 0.50); optionally 'blob' (blurred and
    larger than focus_blob_area_vs_median x the median sharp area; off by default)."""
    rel = [f / ref if ref > 0 else 0.0 for f in fm]
    sharp_area = [a for a, r in zip(areas, rel) if r >= p.focus_sharp_rel]
    med = float(np.median(sharp_area)) if sharp_area else (float(np.median(areas)) if areas else 0.0)
    states = []
    for a, r in zip(areas, rel):
        if r >= p.focus_sharp_rel:
            states.append('sharp')
        elif r >= p.focus_blur_rel:
            states.append('intermediate')
        elif p.focus_blob_area_vs_median > 0 and a > p.focus_blob_area_vs_median * med:
            states.append('blob')
        else:
            states.append('blurred')
    return rel, states


# ---------------------------------------------------------------- measurements

def polygon_area(poly):
    q = np.asarray(poly, float)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_perimeter(poly):
    q = np.asarray(poly, float)
    return float(np.hypot(*np.diff(np.r_[q, q[:1]], axis=0).T).sum())


def shape_measurements(poly, um):
    from scipy.spatial import ConvexHull
    q = np.asarray(poly, float)
    a = polygon_area(q) * um * um
    per = polygon_perimeter(q) * um
    try:
        hull = ConvexHull(q)
        ha, hp = hull.volume * um * um, hull.area * um
    except Exception:
        ha, hp = a, per
    return dict(area_um2=a, perimeter_um=per, circularity=4 * math.pi * a / per ** 2 if per else 0.0,
                solidity=a / ha if ha else 0.0, convexity=hp / per if per else 0.0)


# ---------------------------------------------------------------- one image

def refine_image(regions, um, params=None):
    """regions: list of dicts with rgb (H,W,3 uint8), origin (x0, y0) in full-image pixels, and lists of polygons
    (N,2 arrays, full-image pixels) 'stardist', 'stardist_coarse', 'instanseg'. All regions come from one image
    (the colour classifier and the focus reference are per image).
    Returns one list of nuclei per region (dicts: poly, state, focus_fm, focus_rel, measurements) and run info."""
    p = params or Params()
    if p.work_pixel_um > 0 and um < p.work_pixel_um / 1.1 and regions:
        return _refine_resampled(regions, um, p)
    if p.seed_mode not in ('union', 'coarse_agree'):
        raise ValueError(f'unknown seed_mode {p.seed_mode!r}')
    agree = [agreed_seeds(r.get('stardist_coarse', []), r.get('instanseg', [])) if p.seed_mode == 'coarse_agree' else None
             for r in regions]
    clf = NucleusClassifier(p).fit([(r['rgb'], r['stardist'] if a is None else a, r['origin']) for r, a in zip(regions, agree)], um)
    per_region = []
    for r, a in zip(regions, agree):
        rgb, origin = r['rgb'], r['origin']
        shp = rgb.shape[:2]
        pm = clf.prob(rgb, um)
        ps = cv2.GaussianBlur(pm.astype(np.float32), (0, 0), max(0.7, 0.3 / um))
        odm_s = nuclear_od(rgb, um)
        if a is None:
            union, added = union_detections(r['stardist'], r.get('instanseg', []), shp, origin, p)
            seeds = consolidate(union, r.get('stardist_coarse', []), shp, origin, p, odm_s, ps, um)
        else:                                    # objects both detectors find (coarse StarDist kept where InstanSeg agrees)
            added = 0
            seeds = consolidate(list(a), [], shp, origin, p, odm_s, ps, um)
        occ = np.zeros(shp, np.int32)
        for q in seeds:
            lm = poly_mask_local(q, shp, origin)
            occ[_sl(lm)] += lm[0]
        g = edge_gradient(odm_s)
        cands = [c for c in (seed_outline(q, occ, ps, odm_s, origin, um, p, g) for q in seeds) if c is not None]
        polys = [fourier_smooth(q, um, p.smooth_wavelength_um) for q in resolve(cands, pm, origin, um, odm_s, p)]
        per_region.append(dict(polys=polys, fm=focus_measure(polys, rgb, origin, um, p), odm_s=odm_s, added=added,
                               n_seeds=len(seeds)))
    n_removed = 0
    if p.nucleus_filter:
        # object-level nucleus filter (nucleus_filter.py): detector agreement + object features (+ slide signal if the model
        # uses it). Its focus
        # feature uses the pre-filter field references; the final focus states below are computed on the kept objects only.
        from .nucleus_filter import NucleusFilter, object_features, add_image_features
        refs0, _ = focus_reference([pr['fm'] for pr in per_region], p)
        feats = []
        for r, pr, ref in zip(regions, per_region, refs0):
            dets = {'stardist': r['stardist'], 'stardist_coarse': r.get('stardist_coarse', []), 'instanseg': r.get('instanseg', [])}
            pr['feats'] = []
            for q, fmv in zip(pr['polys'], pr['fm']):
                f = object_features(q, r['rgb'], r['origin'], um, dets)
                if f is not None:
                    f['focus_rel'] = fmv / ref if ref > 0 else float('nan')
                pr['feats'].append(f)
                feats.append(f)
        add_image_features(feats)
        model = NucleusFilter.load(p.nucleus_filter_model or None)
        th = p.nucleus_filter_threshold if p.nucleus_filter_threshold > 0 else model.threshold
        filter_table = []
        for ri, (r, pr) in enumerate(zip(regions, per_region)):
            prob = model.proba([f if f is not None else {} for f in pr['feats']])
            keep = [f is not None and pv >= th for f, pv in zip(pr['feats'], prob)]
            for q, f, pv, k in zip(pr['polys'], pr['feats'], prob, keep):
                c = np.asarray(q, float).mean(0)
                filter_table.append(dict(region=ri, centroid_x=float(c[0]), centroid_y=float(c[1]), p_nucleus=float(pv), kept=bool(k),
                                         **(f or {})))
            n_removed += len(keep) - sum(keep)
            pr['p_nucleus'] = [float(pv) for pv, k in zip(prob, keep) if k]
            pr['polys'] = [q for q, k in zip(pr['polys'], keep) if k]
            pr['fm'] = [fmv for fmv, k in zip(pr['fm'], keep) if k]
    refs, img_ref = focus_reference([pr['fm'] for pr in per_region], p)
    out = []
    for r, pr, ref in zip(regions, per_region, refs):
        areas = [polygon_area(q) * um * um for q in pr['polys']]
        rel, states = focus_states(pr['fm'], areas, ref, p)
        nuclei = []
        pn = pr.get('p_nucleus', [None] * len(pr['polys']))
        for q, f, rl, st, pnv in zip(pr['polys'], pr['fm'], rel, states, pn):
            lm = poly_mask_local(q, r['rgb'].shape[:2], r['origin'])
            meas = shape_measurements(q, um)
            meas['nuclear_od_mean'] = float(pr['odm_s'][_sl(lm)][lm[0]].mean()) if lm[0].any() else 0.0
            nd = dict(poly=q, state=st, focus_fm=f, focus_rel=rl, **meas)
            if pnv is not None:
                nd['p_nucleus'] = pnv
            nuclei.append(nd)
        out.append(nuclei)
    n_all = sum(len(pr['fm']) for pr in per_region)
    info = dict(classifier_train_accuracy=clf.train_accuracy, focus_ref_image=img_ref, focus_ref_fields=refs,
                n_nuclei=n_all, focus_ref_reliable=n_all >= p.focus_min_ref_nuclei, nucleus_filter_removed=n_removed,
                nucleus_filter_table=filter_table if p.nucleus_filter else None,
                instanseg_added=[pr['added'] for pr in per_region], seeds=[pr['n_seeds'] for pr in per_region])
    return out, info


def _refine_resampled(regions, um, p):
    """Images much finer than p.work_pixel_um (over-sampled scans) are refined at that pixel size: every region is
    resampled (area averaging), the detector outlines are scaled, and the outputs are mapped back to the original
    full-image pixels. All settings are in micrometres, so only the pixel grid changes."""
    from dataclasses import replace
    regs, maps = [], []
    for r in regions:
        h, w = r['rgb'].shape[:2]
        nh, nw = max(1, int(round(h * um / p.work_pixel_um))), max(1, int(round(w * um / p.work_pixel_um)))
        sx, sy = nw / w, nh / h
        ox, oy = r['origin']
        fwd = (lambda q, ox=ox, oy=oy, sx=sx, sy=sy: np.c_[(np.asarray(q, float)[:, 0] - ox + 0.5) * sx - 0.5,
                                                         (np.asarray(q, float)[:, 1] - oy + 0.5) * sy - 0.5])
        maps.append((ox, oy, sx, sy))
        regs.append(dict(r, rgb=cv2.resize(r['rgb'], (nw, nh), interpolation=cv2.INTER_AREA), origin=(0, 0),
                         **{k: [fwd(q) for q in r.get(k, [])] for k in ('stardist', 'stardist_coarse', 'instanseg')}))
    um2 = um / float(np.mean([(m[2] + m[3]) / 2 for m in maps]))
    out, info = refine_image(regs, um2, replace(p, work_pixel_um=0.0))

    def back(q, m):
        ox, oy, sx, sy = m
        q = np.asarray(q, float)
        return np.c_[(q[:, 0] + 0.5) / sx - 0.5 + ox, (q[:, 1] + 0.5) / sy - 0.5 + oy]
    for nuclei, m in zip(out, maps):
        for d in nuclei:
            d['poly'] = back(d['poly'], m)
    for t in info.get('nucleus_filter_table') or []:
        c = back([[t['centroid_x'], t['centroid_y']]], maps[t['region']])[0]
        t['centroid_x'], t['centroid_y'] = float(c[0]), float(c[1])
    info.update(resampled_from_um=um, work_pixel_um=um2)
    return out, info
