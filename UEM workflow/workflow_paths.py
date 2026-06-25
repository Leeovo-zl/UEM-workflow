from __future__ import annotations

import os
import re
from pathlib import Path

# Base root for the original raw images.
# Override this per machine/session via the TEM_RAW_ROOT env var.
_DEFAULT_RAW_ROOT = "data/raw"

THIS_DIR = Path(__file__).resolve().parent
RAW_ROOT = Path(os.environ.get("TEM_RAW_ROOT", _DEFAULT_RAW_ROOT))
if not RAW_ROOT.is_absolute():
    RAW_ROOT = (THIS_DIR / RAW_ROOT).resolve()

# Common derived folders
DISPLACEMENT_DIR = RAW_ROOT / "displacement"
FFT_BANDPASS_DIR = DISPLACEMENT_DIR / "fft_bandpass"
TRACE_COMBINED_DIR = DISPLACEMENT_DIR / "trace_combined"


def _natural_sort_key(path_obj: Path):
    """
    Natural sort key for folder names with numbers.
    Example: frame2 < frame10.
    """
    parts = re.split(r"(\d+)", path_obj.name)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def discover_workflow_folders(base_root: Path, required_files=None):
    """
    Discover frame folders dynamically by checking required file(s).
    required_files can be None, a string, or a list/tuple of strings.
    """
    if required_files is None:
        req = []
    elif isinstance(required_files, (str, Path)):
        req = [str(required_files)]
    else:
        req = [str(x) for x in required_files]

    if not base_root.exists():
        return []

    folders = []
    for d in base_root.iterdir():
        if not d.is_dir():
            continue
        if all((d / rf).exists() for rf in req):
            folders.append(d)

    return sorted(folders, key=_natural_sort_key)
