"""A synthetic detect-run folder goes through load_run -> refine_image -> GeoJSON/CSV and back."""
import json
import math

import numpy as np
from PIL import Image

from rns import Params, refine_image
from rns import io as rio
from test_refine import UM, detector_polys, scene


def feature(poly, box_id):
    ring = np.asarray(poly, float).tolist()
    ring.append(ring[0])
    return dict(type='Feature', geometry=dict(type='Polygon', coordinates=[ring]),
                properties=dict(objectType='detection', measurements={'box_id': float(box_id)}))


def make_run(tmp_path):
    img, nuclei = scene()
    x0, y0 = 1000, 2000                                   # the box sits somewhere inside a large image
    shifted = [(cx + x0, cy + y0, r) for cx, cy, r in nuclei]
    (tmp_path / 'regions').mkdir()
    Image.fromarray(img).save(tmp_path / 'regions' / 'box_0001.png')
    json.dump(dict(pixel_size_um=UM, boxes=[dict(id=1, x0=x0, y0=y0, w=img.shape[1], h=img.shape[0],
                                                  cx=x0 + 200, cy=y0 + 200, box_um=100)]),
              open(tmp_path / 'boxes.json', 'w'))
    json.dump([feature(p, 1) for p in detector_polys(shifted)], open(tmp_path / 'stardist_fine.geojson', 'w'))
    json.dump([], open(tmp_path / 'stardist_coarse.geojson', 'w'))
    return shifted


def test_round_trip(tmp_path):
    shifted = make_run(tmp_path)
    um, regions = rio.load_run(str(tmp_path))
    assert um == UM and len(regions) == 1 and len(regions[0]['stardist']) == 4
    assert regions[0]['instanseg'] == []
    nuclei, info = refine_image(regions, um, Params(focus_min_ref_nuclei=1, nucleus_filter=False))
    n = rio.write_geojson(str(tmp_path / 'nuclei.geojson'), nuclei, regions)
    rio.write_csv(str(tmp_path / 'nuclei.csv'), nuclei, regions)
    back = rio.load_qupath_geojson(str(tmp_path / 'nuclei.geojson'))
    assert len(back) == n >= 3
    for d in back:                                         # outlines stay in full-image coordinates
        c = d['poly'].mean(0)
        assert min(math.hypot(c[0] - cx, c[1] - cy) for cx, cy, _ in shifted) < 6
    rows = open(tmp_path / 'nuclei.csv').read().strip().splitlines()
    assert rows[0].startswith('box_id,nucleus,state') and len(rows) == sum(len(x) for x in nuclei) + 1


def test_stale_instanseg_is_ignored(tmp_path):
    """An instanseg.geojson left by an earlier run must not be used when this run skipped or failed InstanSeg."""
    shifted = make_run(tmp_path)
    json.dump([feature(np.asarray([[1100.0, 2100.0], [1180.0, 2100.0], [1180.0, 2180.0], [1100.0, 2180.0]]), 1)],
              open(tmp_path / 'instanseg.geojson', 'w'))
    meta = json.load(open(tmp_path / 'boxes.json'))
    for status, expected in ((None, 0), ('skipped', 0), ('failed', 0), ('ok', 1)):
        if status is None:
            meta.pop('instanseg', None)
        else:
            meta['instanseg'] = status
        json.dump(meta, open(tmp_path / 'boxes.json', 'w'))
        _, regions = rio.load_run(str(tmp_path))
        assert len(regions[0]['instanseg']) == expected, status
    assert shifted
