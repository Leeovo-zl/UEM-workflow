#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from skimage import io, color, img_as_float32
from skimage.draw import polygon2mask
from skimage.filters import gaussian
from skimage.measure import find_contours, regionprops, label
from skimage.transform import resize
from skimage import morphology

from workflow_paths import RAW_ROOT, DISPLACEMENT_DIR, discover_workflow_folders

# =========================
# Parameters (edit as needed)
# =========================

BASE_ROOT  = RAW_ROOT
MODE = "auto"  # "auto" / "range" / "list"
SUBDIR_FMT = "sample_{tag}"
START_IDX = 0
END_IDX = 35
FOLDER_LIST = [
    "sample_a",
]
EXCLUDE_FOLDER_NAMES = set()

# experiment Time step
# TAG_DT_NS = 13.9
# T_UNIT_S  = 1e-12

# Simulation Time step
TAG_DT_NS = 1/(8*5.25)
T_UNIT_S  = 1e-9

# Pixel to nm scale
PIXEL_NM = 0.36    # simulation
# PIXEL_NM = 0.3    #  4GHz experiment
# PIXEL_NM = 0.525    # 5.25GHz experiment
# Input filenames
OVERLAY_NAME       = "overlay_curve_on_gray_color_thick.png"
EXTERNAL_GRAY_NAME = "filtered.png"
POLY_CSV_NAME      = "snake_points.csv"

# Output
DISPLACEMENT_DIR_NAME = "displacement"
GLOBAL_OUT_CSV = DISPLACEMENT_DIR / "spots_inside_polygon_single.csv"
FRAME_OUT_CSV_NAME = "spots_inside_polygon_single_frame.csv"

# Polygon source: True -> from overlay contour, False -> from CSV
USE_OVERLAY_CONTOUR = True

# Dark-spot detection.
SPOT_POLARITY = "dark"

# HSV threshold for orange contour
HSV_H_RANGE = (0.04, 0.12)
HSV_S_MIN   = 0.5
HSV_V_MIN   = 0.4

# Darkest-region parameters
DARK_REGION_SMOOTH_SIGMA = 2.0
DARK_REGION_PERCENTILE = 40
DARK_REGION_MIN_AREA_PX = 50
DARK_REGION_CLOSE_RADIUS = 2

# Center filter
CENTER_ONLY        = True
CENTER_RADIUS_FRAC = 0.2

# Area filter
MIN_SPOT_AREA_PX = 200  # 0 = disable
MAX_SPOT_AREA_PX = 0  # 0 = disable

# Annotation styles
CONTOUR_COLOR = (1.0, 0.5, 0.0)
SPOT_EDGE     = "deepskyblue"
SPOT_FACE     = "none"
SPOT_LINE_W   = 5
DRAW_CENTER_DOTS = True

# Output naming
OUT_ANNOTATED_FMT = "annotated_spots_{pol}_thick.png"
OUT_WHITE_FMT     = "spots_on_white_{pol}_thick.png"
OUT_ON_EXT_FMT    = "spots_on_external_gray_{pol}_thick.png"

CSV_FIELDNAMES = [
    "tag", "t", "polarity", "base_dir", "row", "col", "sigma", "area_px",
    "pixel_nm",
    "frame_center_row", "frame_center_col",
    "dist_frame_center_px", "dist_frame_center_nm",
    "row0_t0", "col0_t0",
    "delta_row_px", "delta_col_px", "delta_disp_px",
    "delta_row_nm", "delta_col_nm", "delta_disp_nm",
]


# =========================
# Utilities
# =========================

def load_gray_float(img_path):
    im = io.imread(str(img_path))

    if im.ndim == 2:
        im_rgb = np.dstack([im, im, im])
        gray  = im.astype(np.float32)
    elif im.ndim == 3 and im.shape[2] == 4:
        im_rgb = color.rgba2rgb(im)
        gray   = color.rgb2gray(im_rgb)
    elif im.ndim == 3 and im.shape[2] == 3:
        im_rgb = im
        gray   = color.rgb2gray(im_rgb)
    else:
        raise ValueError(f"Unsupported image shape: {im.shape}")

    gray = img_as_float32(gray)
    gray = (gray - np.nanmin(gray)) / (np.nanmax(gray) - np.nanmin(gray) + 1e-12)
    return im_rgb, gray


def write_rows_csv(csv_path: Path, rows, fieldnames):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def bootstrap_frame_csvs_from_global(base_root: Path, global_csv: Path, fieldnames):
    """
    If per-frame CSVs do not exist yet but a legacy global CSV exists,
    split that global CSV into per-frame CSVs once so later single-frame reruns
    can overwrite only their own frame CSV without losing the others.
    """
    if not global_csv.exists():
        return

    try:
        df = pd.read_csv(global_csv)
    except Exception:
        return

    if df.empty or "base_dir" not in df.columns:
        return

    for base_dir_str, subdf in df.groupby("base_dir", dropna=False):
        if not isinstance(base_dir_str, str) or not base_dir_str.strip():
            continue

        frame_dir = Path(base_dir_str)
        if not frame_dir.exists():
            continue

        try:
            frame_dir.resolve().relative_to(base_root.resolve())
        except Exception:
            continue

        frame_csv = frame_dir / FRAME_OUT_CSV_NAME
        if frame_csv.exists():
            continue

        rows = subdf.to_dict("records")
        write_rows_csv(frame_csv, rows, fieldnames)


def _parse_tag_index(tag_value) -> Optional[int]:
    text = str(tag_value)
    m = re.search(r"m(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", text)
    return int(m.group(1)) if m else None


def aggregate_frame_csvs(base_root: Path, global_csv: Path, fieldnames):
    frame_rows = []

    for folder in discover_workflow_folders(base_root):
        frame_csv = folder / FRAME_OUT_CSV_NAME
        if not frame_csv.exists():
            continue
        with frame_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [{k: row.get(k, "") for k in fieldnames} for row in reader]
            frame_rows.append((folder, rows))

    all_rows = []
    for _, rows in frame_rows:
        all_rows.extend(rows)

    # Use frame 0000 as the delta reference for each polarity.
    ref_by_pol = {}
    ref_candidates = []
    for row in all_rows:
        tag_idx = _parse_tag_index(row.get("tag", ""))
        if tag_idx == 0:
            ref_candidates.append(row)

    for row in ref_candidates:
        pol = row.get("polarity", "")
        if pol in ref_by_pol:
            continue
        try:
            ref_by_pol[pol] = (float(row["row"]), float(row["col"]))
        except Exception:
            continue

    # Recompute time and all delta fields consistently from the true tag index.
    for row in all_rows:
        tag_idx = _parse_tag_index(row.get("tag", ""))
        if tag_idx is not None:
            row["t"] = float(tag_idx) * TAG_DT_NS * T_UNIT_S

        pol = row.get("polarity", "")
        if pol not in ref_by_pol:
            row["row0_t0"] = np.nan
            row["col0_t0"] = np.nan
            row["delta_row_px"] = np.nan
            row["delta_col_px"] = np.nan
            row["delta_disp_px"] = np.nan
            row["delta_row_nm"] = np.nan
            row["delta_col_nm"] = np.nan
            row["delta_disp_nm"] = np.nan
            continue

        row0, col0 = ref_by_pol[pol]
        r = float(row["row"])
        c = float(row["col"])
        delta_row_px = r - row0
        delta_col_px = c - col0
        delta_disp_px = float(np.hypot(delta_row_px, delta_col_px))

        row["row0_t0"] = row0
        row["col0_t0"] = col0
        row["delta_row_px"] = delta_row_px
        row["delta_col_px"] = delta_col_px
        row["delta_disp_px"] = delta_disp_px
        row["delta_row_nm"] = delta_row_px * PIXEL_NM
        row["delta_col_nm"] = delta_col_px * PIXEL_NM
        row["delta_disp_nm"] = delta_disp_px * PIXEL_NM

    # Rewrite each frame CSV with recomputed delta values.
    rows_by_dir = {}
    for row in all_rows:
        rows_by_dir.setdefault(row.get("base_dir", ""), []).append(row)

    for folder, _ in frame_rows:
        frame_csv = folder / FRAME_OUT_CSV_NAME
        rows = rows_by_dir.get(str(folder), [])
        write_rows_csv(frame_csv, rows, fieldnames)

    # Write global CSV sorted by tag/polarity for readability.
    def _sort_key(row):
        tag_idx = _parse_tag_index(row.get("tag", ""))
        return (999999 if tag_idx is None else tag_idx, str(row.get("polarity", "")), str(row.get("base_dir", "")))

    write_rows_csv(global_csv, sorted(all_rows, key=_sort_key), fieldnames)


def extract_orange_polygon(img_rgb):
    hsv = color.rgb2hsv(img_rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    mask = (h >= HSV_H_RANGE[0]) & (h <= HSV_H_RANGE[1]) & (s >= HSV_S_MIN) & (v >= HSV_V_MIN)
    contours = find_contours(mask.astype(float), level=0.5)
    if len(contours) == 0:
        raise RuntimeError("No orange contour found. Adjust HSV thresholds.")

    return max(contours, key=lambda c: c.shape[0])


def load_polygon_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]
    if "row" in cols and "col" in cols:
        r = df[df.columns[cols.index("row")]].to_numpy(dtype=float)
        c = df[df.columns[cols.index("col")]].to_numpy(dtype=float)
    elif "y" in cols and "x" in cols:
        r = df[df.columns[cols.index("y")]].to_numpy(dtype=float)
        c = df[df.columns[cols.index("x")]].to_numpy(dtype=float)
    else:
        raise ValueError("CSV must contain columns 'row,col' or 'y,x'")
    return np.stack([r, c], axis=1)


def polygon_mask_from_vertices(shape, poly_rc):
    return polygon2mask(shape, poly_rc)


def polygon_centroid(mask_bool):
    lbl = label(mask_bool)
    if lbl.max() == 0:
        return None
    reg = max(regionprops(lbl), key=lambda r: r.area)
    return reg.centroid


def detect_darkest_region(work_gray01, poly_mask, ctr):
    work = work_gray01.copy()
    if DARK_REGION_SMOOTH_SIGMA > 0:
        work = gaussian(work, sigma=DARK_REGION_SMOOTH_SIGMA, preserve_range=True)

    search_mask = poly_mask.copy()
    H, W = work.shape
    approx_R = min(ctr[0], ctr[1], H - 1 - ctr[0], W - 1 - ctr[1])
    if CENTER_ONLY:
        yy, xx = np.indices(work.shape)
        center_mask = ((yy - ctr[0]) ** 2 + (xx - ctr[1]) ** 2) <= (CENTER_RADIUS_FRAC * approx_R) ** 2
        search_mask &= center_mask

    vals = work[search_mask]
    if vals.size == 0:
        return []

    thresh = np.percentile(vals, DARK_REGION_PERCENTILE)
    dark_mask = (work <= thresh) & search_mask
    if DARK_REGION_CLOSE_RADIUS > 0:
        dark_mask = morphology.closing(dark_mask, morphology.disk(DARK_REGION_CLOSE_RADIUS))
    if DARK_REGION_MIN_AREA_PX > 0:
        dark_mask = morphology.remove_small_objects(
            dark_mask,
            max_size=max(int(DARK_REGION_MIN_AREA_PX) - 1, 0),
        )

    lbl = label(dark_mask)
    if lbl.max() == 0:
        return []

    regs = regionprops(lbl, intensity_image=1.0 - work)
    if not regs:
        return []

    # Prefer the darkest substantial region nearest the center.
    def _center(reg):
        center = getattr(reg, "centroid_weighted", None)
        if center is None or center[0] != center[0]:
            center = reg.centroid
        return center

    def _score(reg):
        rr, cc = _center(reg)
        dist = float(np.hypot(rr - ctr[0], cc - ctr[1]))
        return (dist, -reg.area)

    reg = sorted(regs, key=_score)[0]
    rr, cc = _center(reg)
    radius = float(np.sqrt(reg.area / np.pi))
    sigma = radius / np.sqrt(2.0)
    return np.array([[float(rr), float(cc), float(sigma)]], dtype=float)


def filter_and_collect(spots, poly_mask, ctr, H, W, tag, base_dir, polarity):
    rows = []
    approx_R = min(ctr[0], ctr[1], H - 1 - ctr[0], W - 1 - ctr[1])

    for r, c, s in spots:
        r = float(r); c = float(c)
        if not poly_mask[int(np.clip(r, 0, H-1)), int(np.clip(c, 0, W-1))]:
            continue

        area_px = float(np.pi * (float(s) ** 2))

        if MIN_SPOT_AREA_PX > 0 and area_px < MIN_SPOT_AREA_PX:
            continue

        if MAX_SPOT_AREA_PX > 0 and area_px > MAX_SPOT_AREA_PX:
            continue

        if CENTER_ONLY:
            dist = float(np.hypot(r - ctr[0], c - ctr[1]))
            if dist > CENTER_RADIUS_FRAC * approx_R:
                continue

        rows.append({
            "tag": tag,
            "polarity": polarity,
            "base_dir": str(base_dir),
            "row": r,
            "col": c,
            "sigma": float(s),
            "area_px": area_px,
        })
    return rows, approx_R


def draw_outputs(gray, poly_rc, ctr, approx_R, rows, out_annotated, out_white, out_on_external, external_gray):
    H, W = gray.shape
    df = pd.DataFrame(rows)

    # 1) Annotated on original
    fig, ax = plt.subplots(figsize=(W/150, H/150), dpi=150)
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    ax.plot(poly_rc[:, 1], poly_rc[:, 0], color=CONTOUR_COLOR, lw=2.0)

    for _, row in df.iterrows():
        r, c, s = row["row"], row["col"], row["sigma"]
        rad = np.sqrt(2) * s
        ax.add_patch(plt.Circle((c, r), rad, edgecolor=SPOT_EDGE, facecolor=SPOT_FACE, lw=SPOT_LINE_W))
        if DRAW_CENTER_DOTS:
            ax.plot(c, r, "o", ms=3, color="deepskyblue")

    ax.plot(ctr[1], ctr[0], "x", color="gold", ms=6, mew=2)
    if CENTER_ONLY:
        ax.add_patch(plt.Circle((ctr[1], ctr[0]), CENTER_RADIUS_FRAC*approx_R,
                                edgecolor="gold", facecolor="none", lw=1.0, ls="--"))
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_annotated, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    # 2) White background
    fig, ax = plt.subplots(figsize=(W/150, H/150), dpi=150)
    ax.imshow(np.ones((H, W)), cmap="gray", vmin=0, vmax=1)
    for _, row in df.iterrows():
        rr, cc, sig = row["row"], row["col"], row["sigma"]
        rad = np.sqrt(2) * sig
        ax.add_patch(plt.Circle((cc, rr), rad, edgecolor=SPOT_EDGE, facecolor="none", lw=SPOT_LINE_W))
        if DRAW_CENTER_DOTS:
            ax.plot(cc, rr, "o", ms=3, color="black")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_white, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

    # 3) Overlay on external gray
    fig, ax = plt.subplots(figsize=(W/150, H/150), dpi=150)
    ax.imshow(external_gray, cmap="gray", vmin=0, vmax=1)
    for _, row in df.iterrows():
        rr, cc, sig = row["row"], row["col"], row["sigma"]
        rad = np.sqrt(2) * sig
        ax.add_patch(plt.Circle((cc, rr), rad, edgecolor=SPOT_EDGE, facecolor="none", lw=SPOT_LINE_W))
        if DRAW_CENTER_DOTS:
            ax.plot(cc, rr, "o", ms=3, color="black")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_on_external, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


# =========================
# Process one tag
# =========================

def process_one_tag(tag: str, base_dir: Optional[Path] = None, frame_index: Optional[int] = None):
    if base_dir is None:
        base_dir = BASE_ROOT / SUBDIR_FMT.format(tag=tag)
    print(f"\n==== Processing tag={tag}, dir={base_dir} ====")

    overlay_path = base_dir / OVERLAY_NAME
    external_path = base_dir / EXTERNAL_GRAY_NAME
    poly_csv_path = base_dir / POLY_CSV_NAME

    if not overlay_path.exists():
        print(f"[WARN] overlay not found: {overlay_path}, skip")
        return
    if not external_path.exists():
        print(f"[WARN] external gray not found: {external_path}, skip")
        return

    # Time (s)
    tag_idx = _parse_tag_index(tag)
    if tag_idx is None:
        tag_idx = int(frame_index) if frame_index is not None else 0
    t_s = tag_idx * TAG_DT_NS * T_UNIT_S

    # 1) overlay -> polygon
    img_rgb, gray = load_gray_float(overlay_path)
    H, W = gray.shape

    if USE_OVERLAY_CONTOUR:
        poly_rc = extract_orange_polygon(img_rgb)
    else:
        if not poly_csv_path.exists():
            print(f"[WARN] polygon CSV not found: {poly_csv_path}, skip")
            return
        poly_rc = load_polygon_from_csv(poly_csv_path)

    poly_mask = polygon_mask_from_vertices(gray.shape, poly_rc)
    ctr = polygon_centroid(poly_mask)
    if ctr is None:
        print("[WARN] polygon centroid failed, skip")
        return

    # 2) external gray (for overlay)
    _, ext_gray = load_gray_float(external_path)
    if ext_gray.shape != (H, W):
        ext_gray = resize(ext_gray, (H, W), preserve_range=True, anti_aliasing=True).astype(np.float32)

    # 3) frame center in pixel coordinates
    frame_ctr = ( (H - 1) / 2.0, (W - 1) / 2.0 )

    all_rows = []
    spots = detect_darkest_region(gray, poly_mask, ctr)
    rows, approx_R = filter_and_collect(spots, poly_mask, ctr, H, W, tag, base_dir, SPOT_POLARITY)

    for rr in rows:
        r = float(rr["row"]); c = float(rr["col"])

        # Distance to frame center
        drow_fc = r - frame_ctr[0]
        dcol_fc = c - frame_ctr[1]
        dist_frame_center_px = float(np.hypot(drow_fc, dcol_fc))
        dist_frame_center_nm = dist_frame_center_px * PIXEL_NM

        rr.update({
            "t": float(t_s),
            "pixel_nm": float(PIXEL_NM),

            "frame_center_row": float(frame_ctr[0]),
            "frame_center_col": float(frame_ctr[1]),
            "dist_frame_center_px": dist_frame_center_px,
            "dist_frame_center_nm": dist_frame_center_nm,

            "row0_t0": np.nan,
            "col0_t0": np.nan,
            "delta_row_px": np.nan,
            "delta_col_px": np.nan,
            "delta_disp_px": np.nan,
            "delta_row_nm": np.nan,
            "delta_col_nm": np.nan,
            "delta_disp_nm": np.nan,
        })

    all_rows.extend(rows)

    # Output images
    out_annotated = base_dir / OUT_ANNOTATED_FMT.format(pol=SPOT_POLARITY)
    out_white     = base_dir / OUT_WHITE_FMT.format(pol=SPOT_POLARITY)
    out_on_ext    = base_dir / OUT_ON_EXT_FMT.format(pol=SPOT_POLARITY)

    draw_outputs(gray, poly_rc, ctr, approx_R, rows,
                 out_annotated, out_white, out_on_ext, ext_gray)

    print(f"[OK] Saved ({SPOT_POLARITY}): {out_annotated.name}, {out_white.name}, {out_on_ext.name}")
    print(f"[INFO] ({SPOT_POLARITY}) kept spots: {len(rows)}")

    frame_csv = base_dir / FRAME_OUT_CSV_NAME
    write_rows_csv(frame_csv, all_rows, CSV_FIELDNAMES)

    if len(all_rows) > 0:
        print(f"[INFO] Wrote {len(all_rows)} rows to frame CSV: {frame_csv.name}")
    else:
        print("[INFO] No valid spots in this frame")


# =========================
# Main batch
# =========================

def get_targets():
    if MODE == "auto":
        dirs = discover_workflow_folders(BASE_ROOT, required_files=[OVERLAY_NAME, EXTERNAL_GRAY_NAME])
        return [(d.name, d) for d in dirs if d.name not in EXCLUDE_FOLDER_NAMES]
    if MODE == "range":
        tags = [f"{i:04d}" for i in range(START_IDX, END_IDX + 1)]
        return [
            (tag, BASE_ROOT / SUBDIR_FMT.format(tag=tag))
            for tag in tags
            if (BASE_ROOT / SUBDIR_FMT.format(tag=tag)).name not in EXCLUDE_FOLDER_NAMES
        ]
    if MODE == "list":
        return [(name, BASE_ROOT / name) for name in FOLDER_LIST if name not in EXCLUDE_FOLDER_NAMES]
    raise ValueError("MODE must be 'auto', 'range', or 'list'.")

if __name__ == "__main__":
    GLOBAL_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    targets = get_targets()
    bootstrap_frame_csvs_from_global(BASE_ROOT, GLOBAL_OUT_CSV, CSV_FIELDNAMES)

    for idx, (tag, base_dir) in enumerate(targets):
        process_one_tag(tag, base_dir=base_dir, frame_index=idx)

    aggregate_frame_csvs(BASE_ROOT, GLOBAL_OUT_CSV, CSV_FIELDNAMES)

    print("\nDone.")
    print(f"All spot results saved to: {GLOBAL_OUT_CSV.resolve()}")
