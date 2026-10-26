"""Synthetic Pap-like images: blue-purple nuclei on cyan cytoplasm on white glass; one nucleus defocused."""
import math

import cv2
import numpy as np
import pytest

from rns import Params, refine_image
from rns.refine import (focus_measure, fourier_smooth, nuclear_od, poly_mask, polygon_area, resolve)

UM = 0.25
GLASS = (245, 245, 242)
CYTO = (150, 215, 215)       # cyan: absorbs red only
NUC = (95, 80, 150)          # absorbs red and green


def circle(cx, cy, r, n=64):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def scene(blur_last=True, size=400, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32) + GLASS
    sharp_layer = img.copy()
    nuclei = [(100, 100, 24), (260, 110, 28), (110, 270, 22), (280, 280, 30)]
    for cx, cy, r in nuclei:
        cv2.circle(sharp_layer, (cx, cy), int(r * 2.4), CYTO, -1)
    for cx, cy, r in nuclei:
        cv2.circle(sharp_layer, (cx, cy), r, NUC, -1)
    sharp_layer = cv2.GaussianBlur(sharp_layer, (0, 0), 1.0)
    img = sharp_layer
    if blur_last:                                         # defocus the last cell
        cx, cy, r = nuclei[-1]
        m = np.zeros((size, size), np.float32)
        cv2.circle(m, (cx, cy), int(r * 2.6), 1, -1)
        m = cv2.GaussianBlur(m, (0, 0), 6)[..., None]
        img = img * (1 - m) + cv2.GaussianBlur(sharp_layer, (0, 0), 9.0) * m
    img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    return img, nuclei


def detector_polys(nuclei, grow=1.25):
    """Loose H&E-style outlines that include part of the cytoplasm rim."""
    return [circle(cx, cy, r * grow) for cx, cy, r in nuclei]


def test_fourier_smooth_keeps_area_and_removes_wiggles():
    t = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    r = 40 + 1.5 * np.sin(60 * t)                         # 60 wiggles on a 40 px radius (wavelength ~1 um at 0.25 um/px)
    q = np.c_[r * np.cos(t), r * np.sin(t)]
    s = fourier_smooth(q, UM, 2.0)
    rs = np.hypot(s[:, 0], s[:, 1])
    assert abs(polygon_area(s) / (math.pi * 40 ** 2) - 1) < 0.03
    assert rs.std() < 0.3 * (1.5 / math.sqrt(2))


def test_fourier_smooth_keeps_large_notch():
    t = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    r = 40 - 12 * np.exp(-((t - math.pi) / 0.25) ** 2)   # a notch ~10 px deep, ~20 px (5 um) wide
    q = np.c_[r * np.cos(t), r * np.sin(t)]
    s = fourier_smooth(q, UM, 2.0)
    assert np.hypot(s[:, 0], s[:, 1]).min() < 32


def test_focus_measure_separates_sharp_and_defocused():
    img, nuclei = scene()
    polys = [circle(cx, cy, r) for cx, cy, r in nuclei]
    fm = focus_measure(polys, img, (0, 0), UM, Params())
    assert fm[3] < 0.5 * min(fm[:3])


def test_short_detector_outline_grows_to_the_nucleus():
    """A detector outline covering only the left 2/3 of a nucleus (as StarDist does on large irregular Pap nuclei)."""
    img, nuclei = scene(blur_last=False)
    cx, cy, r = nuclei[1]
    short = circle(cx - 0.3 * r, cy, 0.75 * r)                 # misses the right third
    others = [circle(x, y, q * 1.1) for i, (x, y, q) in enumerate(nuclei) if i != 1]
    region = dict(rgb=img, origin=(0, 0), stardist=others + [short], stardist_coarse=[], instanseg=[])
    out, _ = refine_image([region], UM, Params(focus_min_ref_nuclei=1))
    n = min(out[0], key=lambda d: np.hypot(*(np.asarray(d['poly']).mean(0) - (cx, cy))))
    assert n['area_um2'] == pytest.approx(math.pi * r * r * UM * UM, rel=0.15)
    assert np.asarray(n['poly'])[:, 0].max() > cx + 0.85 * r


def test_resolve_outputs_do_not_overlap():
    img, nuclei = scene(blur_last=False)
    odm = nuclear_od(img, UM)
    pm = (odm > 0.3).astype(float)
    polys = [circle(150, 150, 30), circle(190, 150, 30)]
    img2 = img.copy()
    out = resolve(polys, pm + 0.5, (0, 0), UM, odm + 0.5, Params(min_contrast=0.0, halves_min_solidity=1.1,
                                                                   fragment_min_solidity=1.1))
    masks = [poly_mask(q, img2.shape[:2], (0, 0)) for q in out]
    assert len(masks) == 2
    assert (masks[0] & masks[1]).sum() < 0.02 * masks[0].sum()


def test_refine_image_tightens_outlines_and_flags_defocus():
    img, nuclei = scene()
    region = dict(rgb=img, origin=(0, 0), stardist=detector_polys(nuclei), stardist_coarse=[], instanseg=[])
    out, info = refine_image([region], UM, Params(focus_min_ref_nuclei=1))
    nn = out[0]
    assert len(nn) == 4
    assert info['classifier_train_accuracy'] > 0.9
    by_pos = {min(range(4), key=lambda i: np.hypot(*(np.asarray(n['poly']).mean(0) - nuclei[i][:2]))): n for n in nn}
    for i in range(3):                                   # outlines pulled in from the 1.25x detector outline
        true_area = math.pi * nuclei[i][2] ** 2 * UM * UM
        assert by_pos[i]['area_um2'] == pytest.approx(true_area, rel=0.2)
        assert by_pos[i]['state'] == 'sharp'
    assert by_pos[3]['state'] in ('blurred', 'blob', 'intermediate')
    assert by_pos[3]['focus_rel'] < min(by_pos[i]['focus_rel'] for i in range(3))
