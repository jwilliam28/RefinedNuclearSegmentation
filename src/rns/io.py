"""Reading QuPath detections and region images; writing refined nuclei as QuPath-importable GeoJSON and CSV."""
import csv
import json
import os

import numpy as np
from PIL import Image


def load_qupath_geojson(path):
    """QuPath detection GeoJSON (a list of features or a FeatureCollection) -> list of dict(poly, box_id)."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        d = json.load(f)
    feats = d if isinstance(d, list) else d.get('features', [])
    out = []
    for ft in feats:
        g = ft.get('geometry') or {}
        if g.get('type') == 'Polygon':
            ring = g['coordinates'][0]
        elif g.get('type') == 'MultiPolygon':
            ring = max(g['coordinates'], key=lambda c: len(c[0]))[0]
        else:
            continue
        ms = (ft.get('properties') or {}).get('measurements') or {}
        if isinstance(ms, list):                       # older QuPath: [{"name": ..., "value": ...}]
            ms = {m.get('name'): m.get('value') for m in ms}
        bid = ms.get('box_id')
        out.append(dict(poly=np.asarray(ring, float), box_id=int(round(bid)) if bid is not None else None))
    return out


def load_run(run_dir):
    """Read the output of detect_boxes.groovy: boxes.json, region PNGs and the three detection files."""
    with open(os.path.join(run_dir, 'boxes.json')) as f:
        meta = json.load(f)
    dets = {k: load_qupath_geojson(os.path.join(run_dir, f'{k}.geojson')) for k in ('stardist_fine', 'stardist_coarse')}
    # InstanSeg only if this run produced it (a file left from an earlier run must not leak in)
    dets['instanseg'] = load_qupath_geojson(os.path.join(run_dir, 'instanseg.geojson')) if meta.get('instanseg') == 'ok' else []
    regions = []
    for b in meta['boxes']:
        bid = int(b['id'])
        rgb = np.asarray(Image.open(os.path.join(run_dir, 'regions', f'box_{bid:04d}.png')).convert('RGB'))
        x0, y0 = int(b['x0']), int(b['y0'])
        h, w = rgb.shape[:2]

        def near(d):                                   # detections of this box (or, without box ids, inside it)
            if d['box_id'] is not None:
                return d['box_id'] == bid
            c = d['poly'].mean(0)
            return x0 <= c[0] < x0 + w and y0 <= c[1] < y0 + h
        regions.append(dict(id=bid, rgb=rgb, origin=(x0, y0), box=b,
                            stardist=[d['poly'] for d in dets['stardist_fine'] if near(d)],
                            stardist_coarse=[d['poly'] for d in dets['stardist_coarse'] if near(d)],
                            instanseg=[d['poly'] for d in dets['instanseg'] if near(d)]))
    return float(meta['pixel_size_um']), regions


COLOURS = {'sharp': [255, 200, 0], 'intermediate': [255, 120, 0], 'blurred': [0, 170, 255], 'blob': [160, 160, 160]}


def write_geojson(path, nuclei_by_region, regions, include_blobs=False):
    feats = []
    for r, nuclei in zip(regions, nuclei_by_region):
        for i, n in enumerate(nuclei):
            if n['state'] == 'blob' and not include_blobs:
                continue
            ring = np.asarray(n['poly'], float).tolist()
            ring.append(ring[0])
            meas = {k: float(v) for k, v in n.items() if k not in ('poly', 'state') and v is not None and np.isfinite(v)}
            meas['box_id'] = float(r['id'])
            meas['nucleus'] = float(i + 1)                 # same numbering as nuclei.csv
            feats.append(dict(type='Feature', geometry=dict(type='Polygon', coordinates=[ring]),
                              properties=dict(objectType='detection',
                                              classification=dict(name=f"Nucleus: {n['state']}", color=COLOURS[n['state']]),
                                              measurements=meas)))
    with open(path, 'w') as f:
        json.dump(dict(type='FeatureCollection', features=feats), f, allow_nan=False)
    return len(feats)


def write_csv(path, nuclei_by_region, regions, include_blobs=False):
    cols = ['box_id', 'nucleus', 'state', 'centroid_x', 'centroid_y', 'area_um2', 'perimeter_um', 'circularity',
            'solidity', 'convexity', 'nuclear_od_mean', 'focus_fm', 'focus_rel', 'p_nucleus']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r, nuclei in zip(regions, nuclei_by_region):
            for i, n in enumerate(nuclei):
                if n['state'] == 'blob' and not include_blobs:
                    continue
                c = np.asarray(n['poly'], float).mean(0)
                row = dict(n, box_id=r['id'], nucleus=i + 1, centroid_x=c[0], centroid_y=c[1])
                w.writerow([_fmt(k, row.get(k)) for k in cols])


def _fmt(k, v):
    if v is None:
        return ''
    if isinstance(v, (float, np.floating)):
        return f'{v:.2f}' if k in ('centroid_x', 'centroid_y') else f'{v:.4g}'
    return v


def write_overlay(path, rgb, origin, nuclei, size=None):
    from PIL import ImageDraw
    im = Image.fromarray(rgb)
    k = 1.0
    if size:
        k = size / rgb.shape[1]
        im = im.resize((size, int(round(rgb.shape[0] * k))), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for n in nuclei:
        if n['state'] == 'blob':
            continue
        q = [((x - origin[0]) * k, (y - origin[1]) * k) for x, y in np.asarray(n['poly'], float)]
        d.line(q + [q[0]], fill=tuple(COLOURS[n['state']]), width=1)
    im.save(path)


def write_filter_table(path, table, regions):
    """Every object the nucleus filter saw (kept or removed), with its features: add a 'label' column (1 = nucleus,
    0 = not) to some rows and retrain with scripts/train_nucleus_filter.py."""
    if not table:
        return 0
    cols = ['box_id', 'centroid_x', 'centroid_y', 'p_nucleus', 'kept'] + [k for k in table[0] if k not in ('region', 'centroid_x', 'centroid_y', 'p_nucleus', 'kept')]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in table:
            row = dict(t, box_id=regions[t['region']]['id'])
            w.writerow([_fmt(k, row.get(k)) for k in cols])
    return len(table)
