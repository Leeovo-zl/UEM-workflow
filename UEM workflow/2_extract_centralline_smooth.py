#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch ridge-based centerline extraction.
"""

import numpy as np
from pathlib import Path

from skimage import io, color, exposure, filters, morphology, img_as_ubyte
from skimage.filters import sato
from scipy.ndimage import gaussian_filter

from workflow_paths import RAW_ROOT, discover_workflow_folders


# ============================================================
# Parameters (edit as needed)
# ============================================================

BASE_ROOT = RAW_ROOT

MODE = "auto"          # "auto" / "range" / "list"
AUTO_START_NAME = None
AUTO_END_NAME = None
START_IDX = 0
END_IDX   = 128
FOLDER_PATTERN = "sample_{idx:06d}"

FOLDER_LIST = ["t0f", "t1_4f", "t2_4f", "t3_4f", "t4_4f"]

IN_NAME = "sharp.png"

OUT_SKEL_RIDGE = "centerlines_ridge.png"
OUT_OVER_RIDGE = "overlay_centerlines_ridge.png"

PARAMS = {
    # Preprocessing
    "clahe_clip": 0.35,      # 0.1~0.5
    "denoise_sigma": 0.4,    # 0.3~1.2

    # Ridge detection: sigma range should match ridge width (pixels)
    "black_ridges": False,  # False=detect bright centerline; True=detect dark centerline
    "ridge_sigma_min": 6,
    "ridge_sigma_max": 18,
    "ridge_sigma_step": 3,

    # Ridge thresholding
    "th_mode": "percentile",  # "percentile" or "otsu"
    "th_percentile": 70,    # higher -> fewer ridges

    # Post-processing
    "min_obj_area": 12,       # remove small fragments
    "close_disk": 2,          # bridge small gaps
    "prune_spur_iters": 8,   # remove short spurs
    "keep_largest_component": False,  # keep only the largest connected component after thresholding

    # Visualization only
    "thicken_radius": 1,

    # Overlay color
    "overlay_color": (1.0, 0.0, 0.0),
    "overlay_thickness": 1,
}


# ============================================================
# Utilities
# ============================================================

_NEIGH8 = [(-1,-1),(-1,0),(-1,1),
           ( 0,-1),       ( 0,1),
           ( 1,-1),( 1,0),( 1,1)]

def to_gray01(arr):
    if arr.ndim == 2:
        g = arr.astype(np.float32)
    else:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        g = color.rgb2gray(arr).astype(np.float32)
    return (g - g.min()) / (g.max() - g.min() + 1e-12)

def to_rgb01(arr):
    if arr.ndim == 2:
        g = to_gray01(arr)
        return np.dstack([g, g, g])
    else:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        rgb = arr.astype(np.float32)
        return (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)

def save_whitebg_blackline(line_bool, out_path):
    out = np.ones(line_bool.shape, dtype=np.float32)
    out[line_bool] = 0.0
    io.imsave(out_path, img_as_ubyte(out))

def overlay(raw_img, line_bool, out_overlay_path, color_rgb=(1.0, 0.0, 0.0), thickness=1):
    base = to_rgb01(raw_img)
    line = morphology.dilation(line_bool, morphology.disk(int(thickness)))
    out = base.copy()
    out[line] = np.array(color_rgb).reshape(1, 1, 3)
    io.imsave(out_overlay_path, img_as_ubyte(out))

def neighbors(skel, r, c):
    H, W = skel.shape
    out = []
    for dr, dc in _NEIGH8:
        rr, cc = r+dr, c+dc
        if 0 <= rr < H and 0 <= cc < W and skel[rr, cc]:
            out.append((rr, cc))
    return out

def prune_spurs(skel, iters=10):
    sk = skel.copy()
    for _ in range(int(iters)):
        pts = np.argwhere(sk)
        if pts.size == 0:
            break
        kill = []
        for r, c in pts:
            if len(neighbors(sk, int(r), int(c))) <= 1:
                kill.append((int(r), int(c)))
        if not kill:
            break
        rr, cc = zip(*kill)
        sk[np.array(rr), np.array(cc)] = False
    return sk


# ============================================================
# Core: ridge -> skeleton
# ============================================================

def ridge_skeleton(gray01, params):
    # CLAHE + Gaussian smoothing
    img = exposure.equalize_adapthist(gray01, clip_limit=params["clahe_clip"])
    img = gaussian_filter(img, params["denoise_sigma"])

    # Ridge response (bright ridges)
    sigmas = np.arange(params["ridge_sigma_min"],
                       params["ridge_sigma_max"] + 1e-9,
                       params["ridge_sigma_step"], dtype=float)

    resp = sato(img, sigmas=sigmas, black_ridges=bool(params["black_ridges"]))
    resp = (resp - resp.min()) / (resp.max() - resp.min() + 1e-12)

    # Thresholding
    if params["th_mode"].lower() == "otsu":
        th = filters.threshold_otsu(resp)
    else:
        th = np.percentile(resp, float(params["th_percentile"]))

    mask = resp >= th

    # Post-process: connect + remove small objects
    if int(params["close_disk"]) > 0:
        mask = morphology.closing(mask, morphology.disk(int(params["close_disk"])))
    mask = morphology.remove_small_objects(mask, min_size=int(params["min_obj_area"]))

    # Optional: keep only the largest connected component.
    if bool(params.get("keep_largest_component", False)):
        labels = morphology.label(mask, connectivity=2)
        if labels.max() > 0:
            counts = np.bincount(labels.ravel())
            counts[0] = 0  # ignore background
            largest_label = int(np.argmax(counts))
            mask = labels == largest_label

    # Skeletonize
    skel = morphology.skeletonize(mask)

    # Prune spurs
    it = int(params.get("prune_spur_iters", 0))
    if it > 0:
        skel = prune_spurs(skel, iters=it)

    return skel


# ============================================================
# Batch processing
# ============================================================

def process_one_folder(folder: Path):
    img_path = folder / IN_NAME
    if not img_path.exists():
        print(f"[SKIP] not found: {img_path}")
        return

    raw = io.imread(img_path)
    gray = to_gray01(raw)

    skel = ridge_skeleton(gray, PARAMS)

    # Optional visualization thickening
    skel_vis = skel
    if int(PARAMS["thicken_radius"]) > 0:
        skel_vis = morphology.dilation(skel_vis, morphology.disk(int(PARAMS["thicken_radius"])))

    save_whitebg_blackline(skel_vis, folder / OUT_SKEL_RIDGE)
    overlay(raw, skel_vis, folder / OUT_OVER_RIDGE,
            color_rgb=PARAMS["overlay_color"],
            thickness=PARAMS["overlay_thickness"])

    print(f"[DONE] {folder.name} -> {OUT_SKEL_RIDGE}, {OUT_OVER_RIDGE}")


def main():
    if MODE == "auto":
        folders = discover_workflow_folders(BASE_ROOT, required_files=IN_NAME)
        if AUTO_START_NAME is not None:
            folders = [f for f in folders if f.name >= AUTO_START_NAME]
        if AUTO_END_NAME is not None:
            folders = [f for f in folders if f.name <= AUTO_END_NAME]
    elif MODE == "range":
        folders = [BASE_ROOT / FOLDER_PATTERN.format(idx=i) for i in range(START_IDX, END_IDX + 1)]
    else:
        folders = [BASE_ROOT / name for name in FOLDER_LIST]

    print(f"BASE_ROOT = {BASE_ROOT.resolve()}")
    print(f"MODE={MODE} total={len(folders)}")
    print(f"Input={IN_NAME} | Output={OUT_SKEL_RIDGE}")

    print("Ridge params:",
          f"black_ridges={PARAMS['black_ridges']},",
          f"sigmas={PARAMS['ridge_sigma_min']}..{PARAMS['ridge_sigma_max']} step={PARAMS['ridge_sigma_step']},",
          f"th_mode={PARAMS['th_mode']}, th_percentile={PARAMS['th_percentile']},",
          f"keep_largest_component={PARAMS['keep_largest_component']}")

    n_ok, n_fail = 0, 0
    for folder in folders:
        if not folder.exists():
            print(f"[SKIP] folder not found: {folder}")
            n_fail += 1
            continue
        try:
            process_one_folder(folder)
            n_ok += 1
        except Exception as e:
            print(f"[ERROR] {folder}: {type(e).__name__}: {e}")
            n_fail += 1

    print(f"\n[DONE] ok={n_ok}, failed/skipped={n_fail}")


if __name__ == "__main__":
    main()
