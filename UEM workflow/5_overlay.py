#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Overlay fitted curves on the corresponding filtered grayscale images."""

from pathlib import Path

import numpy as np
from skimage import color, img_as_ubyte, io
from skimage.morphology import dilation, disk

from workflow_paths import RAW_ROOT, discover_workflow_folders


# =========================================
# Batch parameters
# =========================================

BASE_ROOT = RAW_ROOT
MODE = "auto"  # "auto" / "range" / "list"
AUTO_START_NAME = None
AUTO_END_NAME = None

START_IDX = 0
END_IDX = 128
FOLDER_PATTERN = "sample_{idx:06d}"

FOLDER_LIST = [
    "sample_a",
    "sample_b",
]

SKEL_NAME = "snake_fitted_curve_thick.png"
GRAY_NAME = "filtered.png"

OUT_BW_NAME = "overlay_curve_on_gray_bw_thick.png"
OUT_COLOR_NAME = "overlay_curve_on_gray_color_thick.png"

LINE_THICKNESS = 3
THRESHOLD = 40


# =========================================
# Utility functions
# =========================================

def to_uint8_gray(im):
    """Convert an image to uint8 grayscale in the range 0..255."""
    im = im.astype(np.float32)
    if im.ndim == 3:
        if im.shape[2] == 4:
            im = im[:, :, :3]
        g = color.rgb2gray(im)
    else:
        g = im
    g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    return (g * 255).astype(np.uint8)


def match_size_center(skel_g: np.ndarray, gray_g: np.ndarray) -> np.ndarray:
    """Center-crop or pad the skeleton image to match the grayscale image size."""
    gh, gw = gray_g.shape
    sh, sw = skel_g.shape
    if (sh, sw) == (gh, gw):
        return skel_g

    canvas = np.full((gh, gw), 255, dtype=np.uint8)

    src_top = max(0, (sh - gh) // 2)
    src_left = max(0, (sw - gw) // 2)
    src_bottom = min(sh, src_top + gh)
    src_right = min(sw, src_left + gw)

    skel_crop = skel_g[src_top:src_bottom, src_left:src_right]
    th, tw = skel_crop.shape

    dst_top = (gh - th) // 2
    dst_left = (gw - tw) // 2
    canvas[dst_top:dst_top + th, dst_left:dst_left + tw] = skel_crop
    return canvas


def process_one(folder: Path):
    skel_path = folder / SKEL_NAME
    gray_path = folder / GRAY_NAME

    if not skel_path.exists():
        print(f"[SKIP] not found: {skel_path}")
        return False
    if not gray_path.exists():
        print(f"[SKIP] not found: {gray_path}")
        return False

    out_bw = folder / OUT_BW_NAME
    out_color = folder / OUT_COLOR_NAME

    skel = io.imread(skel_path)
    gray = io.imread(gray_path)
    skel_g = to_uint8_gray(skel)
    gray_g = to_uint8_gray(gray)

    skel_g = match_size_center(skel_g, gray_g)

    mask = skel_g < THRESHOLD
    if LINE_THICKNESS > 0:
        mask = dilation(mask, disk(LINE_THICKNESS))

    overlay_bw = gray_g.copy()
    overlay_bw[mask] = 0

    gray_rgb = np.dstack([gray_g, gray_g, gray_g])
    overlay_color = gray_rgb.copy()
    overlay_color[mask] = [255, 140, 0]

    io.imsave(out_bw, img_as_ubyte(overlay_bw))
    io.imsave(out_color, img_as_ubyte(overlay_color))

    print(f"[DONE] {folder.name} -> {out_bw.name}, {out_color.name}")
    return True


# =========================================
# Batch entry point
# =========================================

def main():
    if MODE == "auto":
        folders = discover_workflow_folders(BASE_ROOT, required_files=[SKEL_NAME, GRAY_NAME])
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

    print(f"BASE_ROOT = {BASE_ROOT.resolve()}")
    print(f"MODE = {MODE} | total = {len(folders)}")
    print(f"IN: {SKEL_NAME} + {GRAY_NAME}")
    print(f"OUT: {OUT_BW_NAME} / {OUT_COLOR_NAME}")
    print(f"PARAMS: THRESHOLD={THRESHOLD}, LINE_THICKNESS={LINE_THICKNESS}")

    ok = 0
    skip = 0
    for folder in folders:
        if not folder.exists():
            print(f"[SKIP] folder not found: {folder}")
            skip += 1
            continue
        try:
            if process_one(folder):
                ok += 1
            else:
                skip += 1
        except Exception as exc:
            print(f"[ERROR] {folder}: {type(exc).__name__}: {exc}")
            skip += 1

    print(f"\n[DONE] finished. ok={ok}, skipped/failed={skip}")


if __name__ == "__main__":
    main()
