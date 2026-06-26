#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fit a closed active-contour curve around masked skeleton images."""

from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay
from skimage import color, img_as_ubyte, io
from skimage.draw import line as draw_line
from skimage.filters import gaussian
from skimage.morphology import dilation, disk, remove_small_objects
from skimage.segmentation import active_contour

from workflow_paths import RAW_ROOT, discover_workflow_folders


# =========================================
# Batch parameters
# =========================================

BASE_ROOT = RAW_ROOT
MODE = "auto"  # "auto" / "range" / "list"
AUTO_START_NAME = None
AUTO_END_NAME = None

START_IDX = 0
END_IDX = 64
FOLDER_PATTERN = "sample_{idx:06d}"

FOLDER_LIST = [
    "sample_a",
    "sample_b",
]

IN_NAME = "skeleton_masked.png"
OUT_CURVE_NAME = "snake_fitted_curve_thick.png"
OUT_OVER_NAME = "snake_fitted_overlay_thick.png"


# =========================================
# Alpha-shape initialization parameters
# =========================================

SKELETON_THRESHOLD = 0.55
MIN_OBJECT_SIZE = 50
PRE_SMOOTH_SIGMA = 0.0

OUTER_ONLY = True
OUTER_FRAC = 0.45

SAMPLE_STEP = 2
ALPHA = 0.002
INIT_POINTS = 2000


# =========================================
# Active-contour parameters
# =========================================

ALPHA_SNAKE = 0.1
BETA_SNAKE = 0.6
GAMMA_SNAKE = 0.1
W_LINE = -1.5
W_EDGE = 1
ITERATIONS = 1000


# =========================================
# Output controls
# =========================================

RENDER_MODE = "line"  # "dots" or "line"
CLOSE_LOOP = True
THICKEN = 3


# =========================================
# Utility functions
# =========================================

def to_gray01(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        g = color.rgb2gray(img).astype(np.float32)
    else:
        g = img.astype(np.float32)
    return (g - g.min()) / (g.max() - g.min() + 1e-12)


def resample_closed_contour_xy(xy, n_points):
    """Return a closed contour resampled to a fixed number of arc-length points."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 10:
        raise RuntimeError("Contour too short to resample.")

    if np.linalg.norm(xy[0] - xy[-1]) > 1e-6:
        xy = np.vstack([xy, xy[0]])

    d = np.diff(xy, axis=0)
    seg = np.sqrt((d**2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        raise RuntimeError("Contour arc-length is zero.")

    t = np.linspace(0, s[-1], int(n_points), endpoint=False)
    x = np.interp(t, s, xy[:, 0])
    y = np.interp(t, s, xy[:, 1])
    return np.stack([x, y], axis=1)


def polygon_area_xy(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def alpha_shape_boundary(points_xy, alpha):
    """Extract the largest alpha-shape boundary loop from an (x, y) point cloud."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 50:
        raise RuntimeError("Too few points for alpha shape.")

    tri = Delaunay(pts)
    simplices = tri.simplices

    a_pts = pts[simplices[:, 0]]
    b_pts = pts[simplices[:, 1]]
    c_pts = pts[simplices[:, 2]]

    a = np.linalg.norm(b_pts - c_pts, axis=1)
    b = np.linalg.norm(a_pts - c_pts, axis=1)
    c = np.linalg.norm(a_pts - b_pts, axis=1)

    s = (a + b + c) / 2.0
    area2 = np.maximum(s * (s - a) * (s - b) * (s - c), 0.0)
    area = np.sqrt(area2)

    radius = (a * b * c) / np.maximum(4.0 * area, 1e-12)
    keep = radius < (1.0 / max(alpha, 1e-12))
    kept = simplices[keep]
    if kept.shape[0] < 10:
        raise RuntimeError("Alpha kept too few triangles. Adjust ALPHA.")

    from collections import Counter, defaultdict

    def edge_key(i, j):
        return (i, j) if i < j else (j, i)

    edge_counter = Counter()
    for tri_indices in kept:
        edge_counter[edge_key(tri_indices[0], tri_indices[1])] += 1
        edge_counter[edge_key(tri_indices[1], tri_indices[2])] += 1
        edge_counter[edge_key(tri_indices[2], tri_indices[0])] += 1

    boundary_edges = [edge for edge, count in edge_counter.items() if count == 1]
    if len(boundary_edges) < 10:
        raise RuntimeError("Boundary edges too few. Adjust ALPHA or point filtering.")

    adj = defaultdict(list)
    for i, j in boundary_edges:
        adj[i].append(j)
        adj[j].append(i)

    used_edges = set()
    loops = []

    def pop_one_unused_edge():
        for i, js in adj.items():
            for j in js:
                edge = edge_key(i, j)
                if edge not in used_edges:
                    return i, j
        return None

    while True:
        seed = pop_one_unused_edge()
        if seed is None:
            break

        i0, j0 = seed
        loop = [i0, j0]
        used_edges.add(edge_key(i0, j0))
        cur = j0

        for _ in range(200000):
            nbrs = adj[cur]
            if not nbrs:
                break

            nxt = None
            for candidate in nbrs:
                edge = edge_key(cur, candidate)
                if edge not in used_edges:
                    nxt = candidate
                    break

            if nxt is None:
                if loop[0] in nbrs:
                    loop.append(loop[0])
                break

            used_edges.add(edge_key(cur, nxt))
            loop.append(nxt)
            cur = nxt

            if cur == loop[0]:
                break

        xy = pts[np.array(loop[:-1], dtype=int)]
        if xy.shape[0] >= 10:
            loops.append(xy)

    if not loops:
        raise RuntimeError("No loops extracted from alpha-shape boundary.")

    areas = [polygon_area_xy(loop) for loop in loops]
    return loops[int(np.argmax(areas))]


def build_init_contour_from_skeleton(gray01):
    g = gray01.copy()

    if PRE_SMOOTH_SIGMA and PRE_SMOOTH_SIGMA > 0:
        g = gaussian(g, sigma=float(PRE_SMOOTH_SIGMA), preserve_range=True)

    sk = g < float(SKELETON_THRESHOLD)
    if MIN_OBJECT_SIZE and MIN_OBJECT_SIZE > 0:
        sk = remove_small_objects(sk, max_size=max(int(MIN_OBJECT_SIZE) - 1, 0))

    ys, xs = np.nonzero(sk)
    if ys.size < 200:
        raise RuntimeError("Too few skeleton pixels. Check SKELETON_THRESHOLD.")

    if SAMPLE_STEP and SAMPLE_STEP > 1:
        sel = (ys % SAMPLE_STEP == 0) & (xs % SAMPLE_STEP == 0)
        ys, xs = ys[sel], xs[sel]

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

    if OUTER_ONLY:
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        r98 = np.percentile(r, 98)
        keep = r > (OUTER_FRAC * r98)
        pts = pts[keep]
        if pts.shape[0] < 200:
            pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

    xy = alpha_shape_boundary(pts, ALPHA)
    xy_rs = resample_closed_contour_xy(xy, INIT_POINTS)
    return np.stack([xy_rs[:, 1], xy_rs[:, 0]], axis=1)


def run_snake(img_gray01, init_curve_rc):
    try:
        return active_contour(
            img_gray01,
            init_curve_rc,
            alpha=ALPHA_SNAKE,
            beta=BETA_SNAKE,
            gamma=GAMMA_SNAKE,
            w_line=W_LINE,
            w_edge=W_EDGE,
            max_num_iter=ITERATIONS,
        )
    except TypeError:
        try:
            return active_contour(
                img_gray01,
                init_curve_rc,
                alpha=ALPHA_SNAKE,
                beta=BETA_SNAKE,
                gamma=GAMMA_SNAKE,
                w_line=W_LINE,
                w_edge=W_EDGE,
                max_iterations=ITERATIONS,
            )
        except TypeError:
            return active_contour(
                img_gray01,
                init_curve_rc,
                alpha=ALPHA_SNAKE,
                beta=BETA_SNAKE,
                gamma=GAMMA_SNAKE,
                w_line=W_LINE,
                w_edge=W_EDGE,
            )


def render_curve_mask(h, w, snake_rc):
    rr = np.clip(np.round(snake_rc[:, 0]).astype(int), 0, h - 1)
    cc = np.clip(np.round(snake_rc[:, 1]).astype(int), 0, w - 1)

    canvas = np.zeros((h, w), dtype=bool)

    if RENDER_MODE.lower() == "dots":
        canvas[rr, cc] = True
    elif RENDER_MODE.lower() == "line":
        for i in range(len(rr)):
            j = (i + 1) % len(rr) if CLOSE_LOOP else i + 1
            if (not CLOSE_LOOP) and (j >= len(rr)):
                break
            r_line, c_line = draw_line(rr[i], cc[i], rr[j], cc[j])
            r_line = np.clip(r_line, 0, h - 1)
            c_line = np.clip(c_line, 0, w - 1)
            canvas[r_line, c_line] = True
    else:
        raise ValueError("RENDER_MODE must be 'dots' or 'line'.")

    if THICKEN and THICKEN > 0:
        canvas = dilation(canvas, disk(int(THICKEN)))
    return canvas


# =========================================
# Per-folder processing
# =========================================

def process_one_folder(folder: Path):
    in_path = folder / IN_NAME
    if not in_path.exists():
        print(f"[SKIP] not found: {in_path}")
        return

    out_curve = folder / OUT_CURVE_NAME
    out_over = folder / OUT_OVER_NAME

    img = io.imread(in_path)
    gray = to_gray01(img)
    h, w = gray.shape

    init_rc = build_init_contour_from_skeleton(gray)
    snake_rc = run_snake(gray, init_rc)
    curve_mask = render_curve_mask(h, w, snake_rc)

    curve_img = np.ones((h, w), dtype=np.float32)
    curve_img[curve_mask] = 0.0
    io.imsave(out_curve, img_as_ubyte(curve_img))

    overlay = np.dstack([gray, gray, gray])
    overlay[curve_mask] = [0.0, 0.0, 0.0]
    io.imsave(out_over, img_as_ubyte(overlay))

    print(f"[DONE] {folder.name} -> {out_curve.name}, {out_over.name}")


# =========================================
# Batch entry point
# =========================================

def main():
    if MODE == "auto":
        folders = discover_workflow_folders(BASE_ROOT, required_files=IN_NAME)
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
    print(f"IN = {IN_NAME} | OUT = {OUT_CURVE_NAME}, {OUT_OVER_NAME}")
    print(f"Alpha-shape: ALPHA={ALPHA}, OUTER_ONLY={OUTER_ONLY}, OUTER_FRAC={OUTER_FRAC}")

    n_ok = 0
    n_skip = 0
    for folder in folders:
        if not folder.exists():
            print(f"[SKIP] folder not found: {folder}")
            n_skip += 1
            continue
        try:
            process_one_folder(folder)
            n_ok += 1
        except Exception as exc:
            print(f"[ERROR] {folder}: {type(exc).__name__}: {exc}")
            n_skip += 1

    print(f"\n[DONE] finished. ok={n_ok}, skipped/failed={n_skip}")


if __name__ == "__main__":
    main()
