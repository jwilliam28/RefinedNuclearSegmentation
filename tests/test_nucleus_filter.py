import math

import numpy as np

from rns import Params, refine_image
from rns.nucleus_filter import FEATURES, NucleusFilter, with_derived
from test_refine import UM, detector_polys, scene


def _typical(model):
    """A feature row at the model's training medians (a typical object)."""
    return dict(zip(model.features, model.fill.tolist()))


def test_default_model_loads_and_uses_the_size_band():
    m = NucleusFilter.load()
    assert m.features == FEATURES
    assert 'log_small2' in m.features and 'image_median_od' not in m.features
    assert 0.0 < m.threshold < 1.0


def test_size_band_derived_from_area_when_missing():
    m = NucleusFilter.load()
    f = _typical(m)
    f.pop('log_area')
    f.pop('log_small2')
    g = with_derived(f)
    assert g['log_area'] == math.log(f['area_um2'])
    assert np.allclose(m.proba([f]), m.proba([g]))


def test_specks_score_below_a_typical_nucleus():
    m = NucleusFilter.load()
    base = _typical(m)
    base.pop('log_area')
    base.pop('log_small2')
    p = {a: float(m.proba([dict(base, area_um2=a)])[0]) for a in (8.0, 28.0)}
    assert p[8.0] < p[28.0]
    assert with_derived(dict(area_um2=120.0))['log_small2'] == 0.0     # large nuclei are not pushed down by size


def test_refine_image_reports_the_filter_table():
    img, nuclei = scene()
    region = dict(rgb=img, origin=(0, 0), stardist=detector_polys(nuclei), stardist_coarse=[], instanseg=[])
    out, info = refine_image([region], UM, Params(focus_min_ref_nuclei=1))
    table = info['nucleus_filter_table']
    assert len(table) >= len(out[0])
    assert sum(t['kept'] for t in table) == len(out[0])
    assert all('p_nucleus' in t and 'log_small2' in t for t in table)


def test_split_model_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    rows, y = [], []
    for i in range(200):
        big = i % 2 == 0
        f = {k: float(rng.normal()) for k in FEATURES}
        f['area_um2'] = float(rng.uniform(45, 150) if big else rng.uniform(10, 35))
        f.pop('log_area')
        f.pop('log_small2')
        rows.append(f)
        y.append(int(f['contrast'] > 0) if big else int(f['tex_dog'] > 0))
    m = NucleusFilter.train(rows, y, threshold=0.6, split_area_um2=40.0)
    assert m.large is not None
    m.save(str(tmp_path / 'f.json'))
    m2 = NucleusFilter.load(str(tmp_path / 'f.json'))
    assert m2.split_area_um2 == 40.0
    assert np.allclose(m.proba(rows), m2.proba(rows))
    p = m.proba(rows)
    big = np.array([r['area_um2'] >= 40 for r in rows])
    yy = np.array(y)
    assert ((p[big] > 0.5) == (yy[big] == 1)).mean() > 0.8        # large objects follow their own rule (contrast)
    assert ((p[~big] > 0.5) == (yy[~big] == 1)).mean() > 0.8      # small objects theirs (texture)
