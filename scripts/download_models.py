"""Download the detector models into ./models (run it yourself; nothing is downloaded on install).

  python scripts/download_models.py [--dest models]

StarDist  he_heavy_augment.pb          https://github.com/qupath/models (QuPath's StarDist H&E nucleus model)
InstanSeg brightfield_nuclei v0.1.1    https://github.com/instanseg/instanseg/releases (Apache-2.0 method; see the
                                        model README for the training-data licences)

The QuPath extensions are installed separately from QuPath (Extensions > Manage extensions), or by dropping the jars
from https://github.com/qupath/qupath-extension-stardist/releases and
https://github.com/qupath/qupath-extension-instanseg/releases into the QuPath user directory's extensions folder.
"""
import argparse
import hashlib
import os
import urllib.request
import zipfile

MODELS = {   # name: (url, file, sha256 of the file the package was developed and tested with)
    'stardist': ('https://github.com/qupath/models/raw/main/stardist/he_heavy_augment.pb', 'he_heavy_augment.pb',
                 '00033ae84b03ef82faec2ffad4dc7cd12666738ab70b339f35d3a6e5a5379bab'),
    'instanseg': ('https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.1/brightfield_nuclei.zip',
                  'brightfield_nuclei.zip', '427aa9cfe961b6d0012e2abf6befc6a9f76398db5d28dc2302ad911327d1c448'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch(url, path, want=None):
    """Stream to path.part, check the SHA-256 (unless want is None), then move into place."""
    print('downloading', url)
    h = hashlib.sha256()
    with urllib.request.urlopen(url) as r, open(path + '.part', 'wb') as f:
        for b in iter(lambda: r.read(1 << 20), b''):
            f.write(b)
            h.update(b)
    if want is not None and h.hexdigest() != want:
        os.remove(path + '.part')
        raise SystemExit(f'sha256 mismatch for {os.path.basename(path)}: got {h.hexdigest()}, expected {want} '
                         '(the upstream file changed; check it, then pass --no-verify to accept it)')
    os.replace(path + '.part', path)
    print(f'  {os.path.basename(path)}  {os.path.getsize(path):,} bytes  sha256 {h.hexdigest()}'
          + ('  verified' if want is not None else '  (not verified)'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='models')
    ap.add_argument('--only', choices=sorted(MODELS))
    ap.add_argument('--no-verify', action='store_true', help='accept files whose SHA-256 differs from the tested one')
    a = ap.parse_args()
    os.makedirs(a.dest, exist_ok=True)
    for k, (url, name, want) in MODELS.items():
        if a.only and k != a.only:
            continue
        path = os.path.join(a.dest, name)
        if not os.path.exists(path):
            fetch(url, path, None if a.no_verify else want)
        elif not a.no_verify and sha256(path) != want:
            raise SystemExit(f'{path} exists but its sha256 differs from the tested file; delete it or pass --no-verify')
        if name.endswith('.zip'):
            out = os.path.join(a.dest, 'brightfield_nuclei')
            with zipfile.ZipFile(path) as z:
                for m in z.namelist():                    # refuse paths that escape the destination
                    if os.path.isabs(m) or '..' in m.replace('\\', '/').split('/'):
                        raise SystemExit(f'unsafe path in archive: {m}')
                z.extractall(out)
            print('  extracted to', out, '(pass it as --instanseg-model)')


if __name__ == '__main__':
    main()
