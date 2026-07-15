#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Apply ROI masks to extracted skeleton images in batch."""

import numpy as np
from skimage import color, img_as_ubyte, io
from skimage.draw import disk, polygon

from workflow_paths import RAW_ROOT, discover_workflow_folders


# ============================================================
# Batch parameters
# ============================================================

BASE_ROOT = RAW_ROOT

# Folder selection mode:
# - auto: find subfolders containing the required skeleton image.
# - range: use numbered folders such as BASE_ROOT / "sample_000001".
# - list: use explicit folder names from FOLDER_LIST.
MODE = "auto"
AUTO_START_NAME = None
AUTO_END_NAME = None

# Range mode.
START_IDX = 0
END_IDX = 63
FOLDER_PATTERN = "sample_{idx:06d}"

# List mode.
FOLDER_LIST = [
    "sample_a",
    "sample_b",
]


# ============================================================
# ROI configuration
# ============================================================

CFG_TEMPLATE = {
    "skeleton_name": "centerlines_ridge.png",
    "output_name": "skeleton_masked.png",
    # ROI1 keeps skeleton pixels inside this region.
    "roi1": {
        "mode": "rect",  # rect / circle / polygon / mask_file
        "rect": {"x0": 240, "y0": 228, "x1": 780, "y1": 860},
        "circle": {"cx": 550, "cy": 550, "r": 320},
        "polygon": {"points": [(200, 200), (900, 180), (940, 800), (240, 820)]},
        "mask_file": {"path": "roi_mask.png"},
    },
    # ROI2 removes skeleton pixels inside this region.
    "roi2": {
        "mode": "rect",  # rect / circle / polygon / mask_file
        "rect": {"x0": 280, "y0": 248, "x1": 740, "y1": 824},
        "circle": {"cx": 550, "cy": 550, "r": 150},
        "polygon": {"points": [(0, 0), (0, 0), (0, 0)]},
        "mask_file": {"path": "roi_exclude_mask.png"},
    },
}


# ============================================================
# Utility functions
# ============================================================

def load_gray01(path):
    img = io.imread(path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-12)


def build_roi_mask(shape, roi_cfg):
    """Build an ROI mask where True means inside the selected region."""
    h, w = shape
    mode = roi_cfg["mode"]
    mask = np.zeros((h, w), dtype=bool)

    if mode == "rect":
        x0, y0 = roi_cfg["rect"]["x0"], roi_cfg["rect"]["y0"]
        x1, y1 = roi_cfg["rect"]["x1"], roi_cfg["rect"]["y1"]
        mask[y0:y1, x0:x1] = True

    elif mode == "circle":
        cx, cy, r = roi_cfg["circle"]["cx"], roi_cfg["circle"]["cy"], roi_cfg["circle"]["r"]
        rr, cc = disk((cy, cx), r, shape=(h, w))
        mask[rr, cc] = True

    elif mode == "polygon":
        pts = roi_cfg["polygon"]["points"]
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        rr, cc = polygon(ys, xs, shape=(h, w))
        mask[rr, cc] = True

    elif mode == "mask_file":
        m = load_gray01(roi_cfg["mask_file"]["path"])
        if m.shape != (h, w):
            raise ValueError(f"Mask size mismatch: mask={m.shape}, image={(h, w)}")
        mask = m > 0.5

    else:
        raise ValueError("Unknown ROI mode. Expected rect, circle, polygon, or mask_file.")

    return mask


# ============================================================
# Per-folder processing
# ============================================================

def process_one_dir(folder, cfg):
    skel_path = folder / cfg["skeleton_name"]
    if not skel_path.exists():
        print(f"[SKIP] missing skeleton image: {skel_path}")
        return

    print(f"[RUN] {folder}")

    out_path = folder / cfg["output_name"]

    img = load_gray01(skel_path)
    mask_skel = img < 0.5

    roi1 = build_roi_mask(img.shape, cfg["roi1"])
    roi2 = build_roi_mask(img.shape, cfg["roi2"])

    keep = roi1 & (~roi2)

    result = np.ones_like(img, dtype=np.float32)
    result[mask_skel & keep] = 0.0

    io.imsave(out_path, img_as_ubyte(result))
    print(f"  [OK] output: {out_path}")


# ============================================================
# Batch entry point
# ============================================================

if __name__ == "__main__":
    if MODE == "auto":
        folders = discover_workflow_folders(BASE_ROOT, required_files=CFG_TEMPLATE["skeleton_name"])
        if AUTO_START_NAME is not None:
            folders = [f for f in folders if f.name >= AUTO_START_NAME]
        if AUTO_END_NAME is not None:
            folders = [f for f in folders if f.name <= AUTO_END_NAME]
    elif MODE == "range":
        folders = [BASE_ROOT / FOLDER_PATTERN.format(idx=i) for i in range(START_IDX, END_IDX + 1)]
    elif MODE == "list":
        folders = [BASE_ROOT / name for name in FOLDER_LIST]
    else:
        raise ValueError("MODE must be 'auto', 'range', or 'list'.")

    for folder in folders:
        if folder.exists():
            process_one_dir(folder, CFG_TEMPLATE)
        else:
            print(f"[SKIP] folder not found: {folder}")

    print("\n[DONE] ROI mask finished.")
