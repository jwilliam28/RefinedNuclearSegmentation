"""Command line: rns setup-qupath | detect | refine | run."""
import argparse
import glob
import json
import os
import sys

from . import __version__
from . import io as rio
from . import qupath as qp
from .params import Params
from .refine import refine_image


def _detect(a):
    q = qp.find_qupath(a.qupath)
    cmd = qp.detect_command(q, os.path.abspath(a.image), os.path.abspath(a.boxes), os.path.abspath(a.out),
                            os.path.abspath(a.stardist_model),
                            os.path.abspath(a.instanseg_model) if a.instanseg_model else None,
                            a.pixel_size, a.fine_thr, a.fine_px, a.coarse_thr, a.coarse_px, not a.no_papmap)
    if a.print_only:
        print(qp.shell_command(cmd, a.java_mem))
        return
    os.makedirs(a.out, exist_ok=True)
    stale = [os.path.join(a.out, n) for n in ('stardist_fine.geojson', 'stardist_coarse.geojson', 'instanseg.geojson', 'boxes.json')]
    stale += glob.glob(os.path.join(a.out, 'regions', 'box_*.png')) + glob.glob(os.path.join(a.out, 'refined', '*'))
    for f in stale:                               # a missing file must always mean "not produced by this run"
        if os.path.isfile(f):
            os.remove(f)
    lines = qp.run(cmd, java_mem=a.java_mem)
    if a.instanseg_model and any(ln.startswith('RNS_INSTANSEG_SKIPPED') for ln in lines):
        print('warning: InstanSeg was requested but did not run; continuing with StarDist only', file=sys.stderr)


def _refine(a):
    um, regions = rio.load_run(a.out)
    if a.pixel_size:
        um = a.pixel_size
    p = Params()
    if a.params:
        with open(a.params) as f:
            for k, v in json.load(f).items():
                if not hasattr(p, k):
                    raise SystemExit(f'unknown parameter {k}')
                setattr(p, k, v)
    nuclei, info = refine_image(regions, um, p)
    od = os.path.join(a.out, 'refined')
    os.makedirs(od, exist_ok=True)
    n = rio.write_geojson(os.path.join(od, 'nuclei.geojson'), nuclei, regions)
    rio.write_csv(os.path.join(od, 'nuclei.csv'), nuclei, regions)
    rio.write_filter_table(os.path.join(od, 'nucleus_filter_table.csv'), info.pop('nucleus_filter_table', None), regions)
    if not a.no_overlays:
        for r, nn in zip(regions, nuclei):
            rio.write_overlay(os.path.join(od, f"overlay_box_{r['id']:04d}.png"), r['rgb'], r['origin'], nn)
    counts = {s: sum(1 for nn in nuclei for x in nn if x['state'] == s) for s in ('sharp', 'intermediate', 'blurred', 'blob')}
    info.update(pixel_size_um=um, nuclei_written=n, counts=counts, params=p.to_dict(), version=__version__)
    with open(os.path.join(od, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=1)
    print(f"refined: {counts['sharp']} sharp, {counts['intermediate']} intermediate, {counts['blurred']} blurred, "
          f"{counts['blob']} blurred blobs removed (classifier accuracy {info['classifier_train_accuracy']:.3f})")
    if not info['focus_ref_reliable']:
        print(f"warning: only {info['n_nuclei']} nuclei in this image; the focus reference is rough", file=sys.stderr)


def _setup(a):
    q = qp.find_qupath(a.qupath)
    cmd = [q, 'script', '-a', a.user_dir, qp.GROOVY['set_user_dir.groovy']]
    if a.print_only:
        print(qp.shell_command(cmd))
        return
    qp.run(cmd)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='rns', description='Refined nuclear segmentation (StarDist + InstanSeg + colour/ray refinement)')
    ap.add_argument('--version', action='version', version=__version__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('setup-qupath', help='one-time: tell headless QuPath where its user directory (extensions) is')
    s.add_argument('--user-dir', required=True)
    s.add_argument('--qupath')
    s.add_argument('--print-only', action='store_true')
    s.set_defaults(fn=_setup)

    def det_args(x):
        x.add_argument('--image', required=True, help='any image QuPath can open')
        x.add_argument('--boxes', required=True, help='CSV with header x,y,box_um (box centres, full-resolution pixels)')
        x.add_argument('--stardist-model', required=True, help='StarDist .pb, e.g. he_heavy_augment.pb')
        x.add_argument('--instanseg-model', help='InstanSeg model directory (optional cross-detector)')
        x.add_argument('--pixel-size', type=float, default=0.0, help='um per pixel; 0 = image metadata')
        x.add_argument('--fine-thr', type=float, default=0.35)
        x.add_argument('--fine-px', type=float, default=0.35)
        x.add_argument('--coarse-thr', type=float, default=0.5)
        x.add_argument('--coarse-px', type=float, default=0.5)
        x.add_argument('--no-papmap', action='store_true', help='feed raw RGB (for H&E); default remaps Pap colours')
        x.add_argument('--qupath')
        x.add_argument('--java-mem', default='8g')
        x.add_argument('--print-only', action='store_true', help='print the QuPath command instead of running it')

    def ref_args(x):
        x.add_argument('--params', help='JSON file overriding Params fields')
        x.add_argument('--no-overlays', action='store_true')

    s = sub.add_parser('detect', help='run StarDist (fine + coarse) and InstanSeg in boxes; export regions')
    s.add_argument('--out', required=True)
    det_args(s)
    s.set_defaults(fn=_detect)

    s = sub.add_parser('refine', help='refine the detections of a detect run')
    s.add_argument('--out', required=True)
    s.add_argument('--pixel-size', type=float, default=0.0)
    ref_args(s)
    s.set_defaults(fn=_refine)

    s = sub.add_parser('run', help='detect + refine')
    s.add_argument('--out', required=True)
    det_args(s)
    ref_args(s)
    s.set_defaults(fn=lambda a: (_detect(a), None if a.print_only else _refine(a)))

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
