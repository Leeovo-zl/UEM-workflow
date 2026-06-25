#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch enhancement for noisy LTEM images.
Workflow:
1. light pre-filter
2. large-scale Gaussian background subtraction
3. FFT bandpass cleanup
4. mild TV denoising
5. weak unsharp enhancement
"""

from pathlib import Path
from typing import Tuple

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import unsharp_mask
from skimage.restoration import denoise_tv_chambolle

from workflow_paths import RAW_ROOT


def load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return np.loadtxt(p, delimiter=",").astype(np.float32)

    img = iio.imread(p)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img.astype(np.float32)


def safe_percentile(arr, pr=(5, 95)):
    vmin, vmax = np.nanpercentile(arr, pr)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return None, None
    return float(vmin), float(vmax)


def save_gray(arr, out_path, pr=(5, 95), dpi=300):
    vmin, vmax = safe_percentile(arr, pr)
    plt.figure()
    plt.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()


def normalize_percentile(arr: np.ndarray, pr=(0.5, 99.5)) -> np.ndarray:
    lo, hi = np.percentile(arr, pr)
    out = (arr - lo) / max(hi - lo, 1e-12)
    return np.clip(out, 0, 1).astype(np.float32)


def subtract_gaussian_background(
    img: np.ndarray, sigma: float, mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    bg = gaussian_filter(img, sigma=sigma)
    if mode == "subtract":
        corrected = img - bg
    elif mode == "divide":
        corrected = img / np.maximum(bg, 1e-6)
    else:
        raise ValueError("background_mode must be 'subtract' or 'divide'")
    return bg.astype(np.float32), corrected.astype(np.float32)


def _normalize_img_for_fft(arr: np.ndarray) -> np.ndarray:
    mx = float(np.nanmax(arr))
    if mx <= 1.5:
        return arr
    if mx <= 255:
        return arr / 255.0
    return arr / (mx + 1e-12)


def _freq_radius(h, w, eps=1e-12):
    u = (np.arange(w) - w // 2) / w
    v = (np.arange(h) - h // 2) / h
    U, V = np.meshgrid(u, v)
    return np.sqrt(U * U + V * V) + eps


def _build_transfer(R, mode, low_cut, high_cut, method, order):
    method = method.lower()
    if method == "butterworth":

        def LP(fc):
            return 1.0 / (1 + (R / fc) ** (2 * order))

        def HP(fc):
            return 1.0 / (1 + (fc / R) ** (2 * order))

    elif method == "gaussian":

        def LP(fc):
            fc = max(fc, 1e-6)
            return np.exp(-0.5 * (R / fc) ** 2)

        def HP(fc):
            return 1.0 - LP(fc)

    else:
        raise ValueError("method must be 'gaussian' or 'butterworth'")

    mode = mode.lower()
    if mode == "lowpass":
        H = LP(high_cut)
    elif mode == "highpass":
        H = HP(low_cut)
    elif mode == "bandpass":
        H = HP(low_cut) * LP(high_cut)
    elif mode == "bandstop":
        H = 1 - (HP(low_cut) * LP(high_cut))
    else:
        raise ValueError("unsupported mode")

    return np.clip(H, 0, 1)


def freq_filter_image(
    img2d,
    mode,
    low_cut,
    high_cut,
    method,
    order,
    post_smooth,
    post_smooth_sigma,
    norm_percentiles,
):
    img = _normalize_img_for_fft(img2d)
    h, w = img.shape

    R = _freq_radius(h, w)
    H = _build_transfer(R, mode, low_cut, high_cut, method, order)

    F = np.fft.fftshift(np.fft.fft2(img))
    Ff = F * H
    out = np.fft.ifft2(np.fft.ifftshift(Ff)).real
    out = normalize_percentile(out, norm_percentiles)

    if post_smooth:
        out = gaussian_filter(out, sigma=post_smooth_sigma)
    return out.astype(np.float32)


def enhance_and_sharpen(img_norm, tv_weight, tv_niter, unsharp_radius, unsharp_amount):
    x_tv = denoise_tv_chambolle(
        img_norm,
        weight=tv_weight,
        max_num_iter=tv_niter,
    )
    x_sh = unsharp_mask(
        x_tv,
        radius=unsharp_radius,
        amount=unsharp_amount,
        preserve_range=True,
    ).astype(np.float32)
    return x_tv, x_sh


PARAMS = {
    "input_dir": str(RAW_ROOT),
    "input_glob": "*.png",
    "output_root": str(RAW_ROOT),
    "filter_type": "median",
    "median_size": 5,
    "gaussian_sigma": 1.0,
    "background_sigma": 24.0,
    "background_mode": "subtract",
    "pad_frac": 0.15,
    "mode": "bandpass",
    "low_cut": 0.002,
    "high_cut": 0.03,
    "method": "butterworth",
    "order": 3,
    "post_smooth": True,
    "post_smooth_sigma": 0.8,
    "norm_percentiles": (0.5, 99.5),
    "viz_percentiles": (1, 99),
    "tv_weight": 0.08,
    "tv_niter": 40,
    "unsharp_radius": 1.0,
    "unsharp_amount": 0.8,
}


def process_one_image(img_path: Path, p: dict, output_root: Path):
    stem = img_path.stem
    print(f"[PROC] {img_path}")

    outdir = output_root / stem
    outdir.mkdir(parents=True, exist_ok=True)

    raw_vis = outdir / "gray.png"
    pre_vis = outdir / "pre.png"
    bg_vis = outdir / "background.png"
    corr_vis = outdir / "bg_corrected.png"
    filt_vis = outdir / "filtered.png"
    sharp_vis = outdir / "sharp.png"

    img = load_array(str(img_path))
    save_gray(img, raw_vis, pr=p["viz_percentiles"])

    if p["filter_type"] == "median":
        img_pre = median_filter(img, size=p["median_size"])
    elif p["filter_type"] == "gaussian":
        img_pre = gaussian_filter(img, sigma=p["gaussian_sigma"])
    else:
        img_pre = img
    save_gray(img_pre, pre_vis, pr=p["viz_percentiles"])

    bg, img_corr = subtract_gaussian_background(
        img_pre,
        sigma=p["background_sigma"],
        mode=p["background_mode"],
    )
    img_corr = normalize_percentile(img_corr, p["norm_percentiles"])
    save_gray(bg, bg_vis, pr=p["viz_percentiles"])
    save_gray(img_corr, corr_vis, pr=p["viz_percentiles"])

    pad_frac = p["pad_frac"]
    if pad_frac > 0:
        py = int(img_corr.shape[0] * pad_frac)
        px = int(img_corr.shape[1] * pad_frac)
        if py > 0 or px > 0:
            img_pad = np.pad(img_corr, ((py, py), (px, px)), mode="reflect")
        else:
            img_pad = img_corr
    else:
        py = px = 0
        img_pad = img_corr

    filt = freq_filter_image(
        img_pad,
        mode=p["mode"],
        low_cut=p["low_cut"],
        high_cut=p["high_cut"],
        method=p["method"],
        order=p["order"],
        post_smooth=p["post_smooth"],
        post_smooth_sigma=p["post_smooth_sigma"],
        norm_percentiles=p["norm_percentiles"],
    )
    if py > 0:
        filt = filt[py:-py, :]
    if px > 0:
        filt = filt[:, px:-px]
    save_gray(filt, filt_vis, pr=p["viz_percentiles"])

    _, sharp = enhance_and_sharpen(
        filt,
        tv_weight=p["tv_weight"],
        tv_niter=p["tv_niter"],
        unsharp_radius=p["unsharp_radius"],
        unsharp_amount=p["unsharp_amount"],
    )
    save_gray(sharp, sharp_vis, pr=p["viz_percentiles"])

    print(f"  [OK] saved into folder: {outdir}")


if __name__ == "__main__":
    p = PARAMS
    input_dir = Path(p["input_dir"])
    output_root = Path(p["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    img_files = sorted(input_dir.glob(p["input_glob"]))
    if not img_files:
        raise SystemExit(f"[ERROR] No files found in {input_dir} matching {p['input_glob']}")

    print(f"[INFO] Found {len(img_files)} images")
    for img_path in img_files:
        process_one_image(img_path, p, output_root)

    print(f"\n[DONE] All images processed -> {output_root.resolve()}")
