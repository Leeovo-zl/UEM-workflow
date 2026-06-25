#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Measure centroid displacement from fitted outline images."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import color, io
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk

from workflow_paths import DISPLACEMENT_DIR, RAW_ROOT, discover_workflow_folders


# =============================
# Batch parameters
# =============================

BASE_ROOT = RAW_ROOT
MODE = "auto"  # "auto" / "range" / "list"

START_IDX = 0
END_IDX = 63
FOLDER_PATTERN = "sample_{idx:06d}"
FOLDER_LIST = [
    "sample_a",
    "sample_b",
]

IMG_NAME = "snake_fitted_curve_thick.png"
BASE_FRAME_INDEX = 0
# Time step: 13.9 ps (experimental frame rate)
# DT_PS = 13.9
# DT_S  = DT_PS * 1e-12

# Time step: 1/(8*f) ns (simulation frame rate)
DT_NS = 1 / (8 * 5.25)
DT_S = DT_NS * 1e-9
NM_PER_PX = 0.36 # simulation

OVERLAY_DIR = DISPLACEMENT_DIR
OUT_DATA_CSV = OVERLAY_DIR / "centroid_results_all.csv"

MARK_SIZE = 20
MARK_LINEWIDTH = 2
FALLBACK_THRESHOLD = 220


# =============================
# Utility functions
# =============================

def to_gray_u8(im):
    if im.ndim == 3:
        if im.shape[2] == 4:
            im = im[:, :, :3]
        g = color.rgb2gray(im).astype(np.float32)
    else:
        g = im.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    return (g * 255).astype(np.uint8)


def load_gray(path: Path):
    im = io.imread(str(path))
    return to_gray_u8(im)


def average_span_lengths(mask):
    row_spans = []
    for row in mask:
        cols = np.flatnonzero(row)
        if cols.size > 0:
            row_spans.append(cols[-1] - cols[0] + 1)

    col_spans = []
    for col in mask.T:
        rows = np.flatnonzero(col)
        if rows.size > 0:
            col_spans.append(rows[-1] - rows[0] + 1)

    if not row_spans or not col_spans:
        return None, None

    width_px = float(np.mean(row_spans))
    height_px = float(np.mean(col_spans))
    return height_px, width_px


def centroid_from_outline(gray_u8):
    th = threshold_otsu(gray_u8)
    if th <= 1:
        print(f"[WARN] Otsu threshold abnormal ({th:.2f}), fallback to {FALLBACK_THRESHOLD}.")
        th = FALLBACK_THRESHOLD

    print(f"[INFO] threshold = {th:.2f}")

    edge = gray_u8 < th
    edge = dilation(edge, disk(1))
    filled = binary_fill_holes(edge)

    lbl = label(filled)
    if lbl.max() == 0:
        return None, None, filled, None, None

    props = regionprops(lbl)
    largest = max(props, key=lambda r: r.area)

    cy, cx = largest.centroid
    area = largest.area
    height_px, width_px = average_span_lengths(largest.image)
    return (cy, cx), area, filled, height_px, width_px


def pad_to(H, W, img):
    """Place an image in the center of a white canvas and return its offset."""
    if img.shape[0] > H or img.shape[1] > W:
        raise ValueError(f"Target canvas {(H, W)} is smaller than image {img.shape}.")

    canvas = np.full((H, W), 255, dtype=np.uint8)
    dst_top = (H - img.shape[0]) // 2
    dst_left = (W - img.shape[1]) // 2
    canvas[dst_top:dst_top + img.shape[0], dst_left:dst_left + img.shape[1]] = img
    return canvas, (dst_top, dst_left)


def shift_centroid(centroid, offset):
    return centroid[0] + offset[0], centroid[1] + offset[1]


def get_target_folders():
    if MODE == "auto":
        return discover_workflow_folders(BASE_ROOT, required_files=IMG_NAME)
    if MODE == "range":
        return [BASE_ROOT / FOLDER_PATTERN.format(idx=i) for i in range(START_IDX, END_IDX + 1)]
    if MODE == "list":
        return [BASE_ROOT / name for name in FOLDER_LIST]
    raise ValueError("MODE must be 'auto', 'range', or 'list'.")


def path_for_csv(path: Path):
    try:
        return str(path.resolve().relative_to(BASE_ROOT.resolve()))
    except ValueError:
        return path.name


def result_fieldnames():
    return [
        "run_tag",
        "t",
        "base_frame_index",
        "dt_s",
        "nm_per_px",
        "image1_path",
        "image1_center_row",
        "image1_center_col",
        "image1_area_px",
        "image1_height_px",
        "image1_width_px",
        "image2_path",
        "image2_center_row",
        "image2_center_col",
        "image2_area_px",
        "image2_height_px",
        "image2_width_px",
        "delta_row_px",
        "delta_col_px",
        "delta_area_px",
        "delta_height_px",
        "delta_width_px",
        "delta_row_nm",
        "delta_col_nm",
        "delta_area_nm2",
        "delta_height_nm",
        "delta_width_nm",
    ]


def main():
    folders = get_target_folders()
    if not folders:
        raise RuntimeError(f"No valid folders found under: {BASE_ROOT}")

    if BASE_FRAME_INDEX < 0 or BASE_FRAME_INDEX >= len(folders):
        raise IndexError(f"BASE_FRAME_INDEX out of range: {BASE_FRAME_INDEX}, total={len(folders)}")

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    img1_dir = folders[BASE_FRAME_INDEX]
    img1_path = img1_dir / IMG_NAME
    if not img1_path.exists():
        raise FileNotFoundError(f"Baseline image not found: {img1_path}")

    print(f"[BASE] Using baseline image: {img1_path}")
    g1 = load_gray(img1_path)
    h1, w1 = g1.shape

    c1, area1, _, height1_px, width1_px = centroid_from_outline(g1)
    if c1 is None:
        raise RuntimeError("Failed to find closed region in baseline image.")

    print(
        f"[BASE] Image1 centroid = ({c1[0]:.2f}, {c1[1]:.2f}), "
        f"area = {area1}, height = {height1_px}, width = {width1_px}"
    )

    fieldnames = result_fieldnames()
    with open(OUT_DATA_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"[CSV] Results will be written to: {OUT_DATA_CSV}")
        print(f"[SCALE] NM_PER_PX = {NM_PER_PX} nm/px")
        print(f"[TIME] DT_S = {DT_S} s")
        print(f"[MODE] {MODE}, total folders = {len(folders)}")

        for idx, img2_dir in enumerate(folders):
            tag = img2_dir.name
            t_s = idx * DT_S

            img2_path = img2_dir / IMG_NAME
            print(f"\n==== PROCESS TAG = {tag} (t = {t_s:.6e} s) ====")
            print(f"[IMG2] {img2_path}")

            if not img2_path.exists():
                print("[WARN] not found, skip.")
                continue

            g2 = load_gray(img2_path)
            h2, w2 = g2.shape

            c2, area2, _, height2_px, width2_px = centroid_from_outline(g2)
            if c2 is None:
                print("[WARN] no valid region in second image, skip this tag.")
                continue

            if (h1, w1) != (h2, w2):
                print("[WARN] size mismatch, center-pad with white for alignment.")
                H, W = max(h1, h2), max(w1, w2)
                g1_use, off1 = pad_to(H, W, g1)
                g2_use, off2 = pad_to(H, W, g2)
                c1_use = shift_centroid(c1, off1)
                c2_use = shift_centroid(c2, off2)
            else:
                H, W = h1, w1
                g1_use, g2_use = g1, g2
                c1_use, c2_use = c1, c2

            dy_px = c2_use[0] - c1_use[0]
            dx_px = c2_use[1] - c1_use[1]
            darea_px = area2 - area1
            dheight_px = height2_px - height1_px
            dwidth_px = width2_px - width1_px

            dy_nm = dy_px * NM_PER_PX
            dx_nm = dx_px * NM_PER_PX
            darea_nm2 = darea_px * (NM_PER_PX ** 2)
            dheight_nm = dheight_px * NM_PER_PX
            dwidth_nm = dwidth_px * NM_PER_PX

            out_overlay = OVERLAY_DIR / f"centroid_overlay_{tag}.png"
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            ax.imshow(g1_use, cmap="gray", alpha=0.7)
            ax.imshow(g2_use, cmap="gray", alpha=0.4)

            ax.scatter(
                c1_use[1],
                c1_use[0],
                c="red",
                s=MARK_SIZE,
                edgecolor="black",
                linewidth=MARK_LINEWIDTH,
            )
            ax.scatter(
                c2_use[1],
                c2_use[0],
                c="blue",
                s=MARK_SIZE,
                edgecolor="black",
                linewidth=MARK_LINEWIDTH,
            )
            ax.plot([c1_use[1], c2_use[1]], [c1_use[0], c2_use[0]], "w--")

            ax.axis("off")
            plt.savefig(out_overlay, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"   -> saved overlay: {out_overlay}")

            writer.writerow({
                "run_tag": tag,
                "t": t_s,
                "base_frame_index": BASE_FRAME_INDEX,
                "dt_s": DT_S,
                "nm_per_px": NM_PER_PX,
                "image1_path": path_for_csv(img1_path),
                "image1_center_row": c1_use[0],
                "image1_center_col": c1_use[1],
                "image1_area_px": area1,
                "image1_height_px": height1_px,
                "image1_width_px": width1_px,
                "image2_path": path_for_csv(img2_path),
                "image2_center_row": c2_use[0],
                "image2_center_col": c2_use[1],
                "image2_area_px": area2,
                "image2_height_px": height2_px,
                "image2_width_px": width2_px,
                "delta_row_px": dy_px,
                "delta_col_px": dx_px,
                "delta_area_px": darea_px,
                "delta_height_px": dheight_px,
                "delta_width_px": dwidth_px,
                "delta_row_nm": dy_nm,
                "delta_col_nm": dx_nm,
                "delta_area_nm2": darea_nm2,
                "delta_height_nm": dheight_nm,
                "delta_width_nm": dwidth_nm,
            })

    print("\n[DONE] Batch centroid analysis finished.")
    print(f"[OUT] Results saved to: {OUT_DATA_CSV}")


if __name__ == "__main__":
    main()
