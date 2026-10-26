"""Running the detection step in headless QuPath."""
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import resources

GROOVY = {name: str(resources.files('rns').joinpath('groovy', name)) for name in ('detect_boxes.groovy', 'set_user_dir.groovy')}


def _version(path):
    m = re.search(r'QuPath[-_ ]?v?(\d+)\.(\d+)\.(\d+)', path)
    return tuple(int(x) for x in m.groups()) if m else (-1, -1, -1)


def find_qupath(explicit=None):
    """QuPath console executable: --qupath, then $RNS_QUPATH (an error if either is given but invalid), then the
    newest version found in common install locations, then PATH."""
    for src, c in (('--qupath', explicit), ('RNS_QUPATH', os.environ.get('RNS_QUPATH'))):
        if c:
            r = c if os.path.isfile(c) else shutil.which(c)
            if not r:
                raise FileNotFoundError(f'{src}={c!r} is not an existing file or a command on PATH')
            return r
    pats = []
    if platform.system() == 'Windows':
        base = os.environ.get('LOCALAPPDATA', '')
        pats += [os.path.join(base, 'QuPath*', 'QuPath*(console).exe'), r'C:\Program Files\QuPath*\QuPath*(console).exe']
    elif platform.system() == 'Darwin':
        pats += ['/Applications/QuPath*.app/Contents/MacOS/QuPath*']
    else:
        pats += ['/opt/QuPath*/bin/QuPath*', os.path.expanduser('~/QuPath*/bin/QuPath*')]
    hits = sorted({h for p in pats for h in glob.glob(p) if os.path.isfile(h)}, key=lambda h: (_version(h), h))
    if hits:
        return hits[-1]
    w = shutil.which('QuPath') or shutil.which('qupath')
    if w:
        return w
    raise FileNotFoundError('QuPath not found: pass --qupath or set RNS_QUPATH to the QuPath (console) executable')


def store_python_warning():
    """On Windows, QuPath started from Microsoft Store Python sees a virtualised registry, loses its user-directory
    preference and loads no extensions. Return a warning string if this Python looks like a Store/MSIX install."""
    exe = sys.executable.replace('/', '\\').lower()
    if platform.system() == 'Windows' and ('windowsapps' in exe or 'pythonsoftwarefoundation' in exe or 'pythoncore-' in exe):
        return ('this Python may be a Microsoft Store / packaged install; if QuPath reports that StarDist is missing, '
                'run the printed command from a normal shell instead (rns detect ... --print-only)')
    return None


def detect_command(qupath, image, boxes_csv, out_dir, stardist_model, instanseg_model=None, pixel_size_um=0.0,
                   fine_thr=0.35, fine_px=0.35, coarse_thr=0.5, coarse_px=0.5, papmap=True, regions=True):
    opts = f'fine_thr={fine_thr};fine_px={fine_px};coarse_thr={coarse_thr};coarse_px={coarse_px};' \
           f'papmap={int(bool(papmap))};regions={int(bool(regions))}'
    return [qupath, 'script', '-i', image, '-a', boxes_csv, '-a', out_dir, '-a', stardist_model,
            '-a', instanseg_model or '-', '-a', f'{pixel_size_um}', '-a', opts, GROOVY['detect_boxes.groovy']]


def run(cmd, java_mem='8g', echo_prefix='RNS_'):
    env = dict(os.environ)
    env.setdefault('JAVA_TOOL_OPTIONS', f'-Xmx{java_mem}')
    w = store_python_warning()
    if w:
        print('warning:', w, file=sys.stderr)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, errors='replace')
    lines = (proc.stdout or '').splitlines() + (proc.stderr or '').splitlines()
    for ln in lines:
        if ln.startswith(echo_prefix) or 'ERROR' in ln:
            print(ln)
    if proc.returncode != 0 or not any(ln.startswith('RNS_DONE') or ln.startswith('RNS_QUPATH_USER_DIR') for ln in lines):
        raise RuntimeError(f'QuPath step failed (exit code {proc.returncode}); see the lines above')
    return lines


def shell_command(cmd, java_mem='8g'):
    """The command as one line to paste into a shell (PowerShell on Windows, POSIX sh elsewhere), with the Java heap."""
    if platform.system() == 'Windows':
        def q(c):
            return "'" + str(c).replace("'", "''") + "'"
        return f"$env:JAVA_TOOL_OPTIONS='-Xmx{java_mem}'; & " + ' '.join(q(c) for c in cmd)
    import shlex
    return f'JAVA_TOOL_OPTIONS=-Xmx{java_mem} ' + ' '.join(shlex.quote(str(c)) for c in cmd)
