"""Retrain the nucleus filter on your own grades.

  python scripts/train_nucleus_filter.py graded.csv [more.csv ...] --out models/my_filter.json [--precision 0.94] [--group-col study]
                                         [--split-um2 40]

Input: one or more nucleus_filter_table.csv files (written by `rns refine`) with an added column 'label' (1 = nucleus,
0 = not a nucleus; rows without a label are ignored). The threshold is the smallest one whose cross-validated precision
(leave-one-group-out when --group-col names a column such as the slide, else 5-fold) reaches --precision.
Use it with: rns refine --params p.json, where p.json is {"nucleus_filter_model": "models/my_filter.json"}.
"""
import argparse
import csv
import sys

import numpy as np

sys.path.insert(0, 'src')
from rns.nucleus_filter import FEATURES, NucleusFilter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--precision', type=float, default=0.94)
    ap.add_argument('--group-col', default='')
    ap.add_argument('--split-um2', type=float, default=40.0, help='objects at least this large get their own model (0 = one model)')
    a = ap.parse_args()
    rows = [r for f in a.csv for r in csv.DictReader(open(f)) if r.get('label', '').strip() in ('0', '1')]
    if len(rows) < 30:
        raise SystemExit(f'only {len(rows)} labelled rows; grade at least ~100 objects')
    # columns missing from older tables (the size-band features) are derived from area_um2 by the filter
    feats = [{k: float(r[k]) if r[k] not in ('', 'nan') else float('nan') for k in FEATURES if k in r} for r in rows]
    y = np.array([int(r['label']) for r in rows])
    groups = np.array([r.get(a.group_col, '') for r in rows]) if a.group_col else np.arange(len(rows)) % 5
    p = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(set(y[tr])) < 2:
            continue
        m = NucleusFilter.train([feats[i] for i in np.flatnonzero(tr)], y[tr], split_area_um2=a.split_um2 or None)
        p[te] = m.proba([feats[i] for i in np.flatnonzero(te)])
    best = None
    for th in np.round(np.arange(0.2, 0.91, 0.05), 2):
        keep = p >= th
        prec = (keep & (y == 1)).sum() / max(keep.sum(), 1)
        rec = (keep & (y == 1)).sum() / max(y.sum(), 1)
        print(f'threshold {th:.2f}: precision {prec:.3f}, recall {rec:.3f}')
        if best is None and prec >= a.precision:
            best = float(th)
    best = best if best is not None else 0.7
    model = NucleusFilter.train(feats, y, threshold=best, info=dict(trained_on=f'{len(y)} graded objects from {", ".join(a.csv)}'),
                                split_area_um2=a.split_um2 or None)
    model.save(a.out)
    print('saved', a.out, 'threshold', best)


if __name__ == '__main__':
    main()
