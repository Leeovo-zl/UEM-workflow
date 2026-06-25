#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandpass -> Fit -> Handedness -> SNR report
(2) Bandpass around 4 GHz / 8 GHz in FFT domain
(3) Handedness via z = x + i y and mean Im(conj(z) * dz/dt)
(4) Noise floor and SNR report outputs
(5) Decompose fitted motion into CW/CCW circular components and report fractions

Outputs:
- bandpassed_timeseries.csv
- fit_params.csv
- snr_total.csv
- snr_report.txt
- fit_expressions_all.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path

from workflow_paths import DISPLACEMENT_DIR, FFT_BANDPASS_DIR

# ============================================================
# Parameters (edit as needed)
# ============================================================

# Input CSVs
CSV_PATH_INPUT1 = DISPLACEMENT_DIR / "centroid_results_all.csv"
CSV_PATH_INPUT2 = DISPLACEMENT_DIR / "spots_inside_polygon_single.csv"

# Select which input(s) to process: "input1" / "input2" / "both"
INPUT_SOURCE = "both"

# Output directories under FFT_BANDPASS_DIR
OUT_DIR_INPUT1 = FFT_BANDPASS_DIR / "askyframe"
OUT_DIR_INPUT2 = FFT_BANDPASS_DIR / "vortex_core"

# Column names
COL_T = "t"
COL_X = "delta_col_nm"
COL_Y = "delta_row_nm"
COL_AREA = "delta_area_nm2"
COL_WIDTH = "delta_width_nm"
COL_HEIGHT = "delta_height_nm"

# If multiple spots per time exist, choose one (prefer closest to t0 or center).
INPUT_POLARITY = "dark"  # "dark" / "bright" / None to disable filtering

# Time unit for input column: 's' / 'ms' / 'us' / 'ns' / 'ps'
TIME_UNIT = "s"

# If y uses drow (positive downward), set True for y_phys = -y_img
FLIP_Y_TO_PHYSICAL = True

# Target frequencies (Hz)
FREQS_HZ = [5.25e9, 10.5e9]

# Bandpass half-width (Hz)
BAND_HALF_WIDTH_HZ = {
    5.25e9: 1.05e9,
    10.5e9: 0.40e9,
}

# Preprocessing
DETREND_LINEAR = True
USE_HANN = True

# LS fit on bandpassed totals
DO_SINGLE_FREQ_FIT = True

# ============================================================
# Utilities
# ============================================================

def unit_scale(unit: str) -> float:
    unit = unit.lower()
    scales = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
    if unit not in scales:
        raise ValueError(f"Unknown TIME_UNIT={unit}. Use one of {list(scales.keys())}.")
    return scales[unit]


def detrend_linear(t, y):
    A = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    trend = A @ coef
    return y - trend, coef


def fft_bandpass_real(y, dt, f0, half_width, use_hann=True):
    n = len(y)
    if use_hann:
        win = np.hanning(n)
        y_w = y * win
    else:
        win = np.ones(n)
        y_w = y

    Y = np.fft.fft(y_w)
    freqs = np.fft.fftfreq(n, d=dt)

    mask_pos = (freqs >= (f0 - half_width)) & (freqs <= (f0 + half_width))
    mask_neg = (freqs <= -(f0 - half_width)) & (freqs >= -(f0 + half_width))
    mask = mask_pos | mask_neg

    Y_f = np.zeros_like(Y)
    Y_f[mask] = Y[mask]
    y_bp = np.fft.ifft(Y_f).real

    gain = np.mean(win)
    if gain > 1e-12:
        y_bp = y_bp / gain

    return y_bp


def ls_fit_multi_cos_sin(t, y, freqs_hz):
    cols = []
    for f_hz in freqs_hz:
        w = 2 * np.pi * f_hz
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    M = np.column_stack(cols)

    coef, *_ = np.linalg.lstsq(M, y, rcond=None)

    coeffs = []
    amps = []
    phis = []
    for i, _ in enumerate(freqs_hz):
        a = float(coef[2 * i])
        b = float(coef[2 * i + 1])
        coeffs.append((a, b))
        amps.append(float(np.hypot(a, b)))
        phis.append(float(np.arctan2(b, a)))

    y_fit = M @ coef
    return coeffs, amps, phis, y_fit


def handedness_from_complex(t, x, y):
    z = x + 1j * y
    dt = np.median(np.diff(t))
    dz = np.diff(z) / dt
    zc = z[:-1]
    L = np.imag(np.conj(zc) * dz)
    val = np.mean(L[np.isfinite(L)])
    if val > 0:
        return "CCW", float(val)
    if val < 0:
        return "CW", float(val)
    return "UNSURE", float(val)


def cw_ccw_fraction_from_ab(a_x, b_x, a_y, b_y):
    A = complex(a_x, a_y)
    B = complex(b_x, b_y)
    Zp = 0.5 * (A - 1j * B)
    Zm = 0.5 * (A + 1j * B)
    P_ccw = (Zp.real ** 2 + Zp.imag ** 2)
    P_cw = (Zm.real ** 2 + Zm.imag ** 2)
    denom = P_ccw + P_cw
    frac_cw = P_cw / denom if denom > 0 else np.nan
    frac_ccw = P_ccw / denom if denom > 0 else np.nan
    return Zp, Zm, P_ccw, P_cw, frac_ccw, frac_cw


def robust_noise_std(residual):
    residual = np.asarray(residual, dtype=float)
    med = np.median(residual)
    mad = np.median(np.abs(residual - med))
    return float(1.4826 * mad)


def write_fit_expressions_all_txt(
    freqs_hz,
    x_terms,
    y_terms,
    x_amps,
    x_phis,
    y_amps,
    y_phis,
    outpath,
    area_terms=None,
    area_amps=None,
    area_phis=None,
    width_terms=None,
    width_amps=None,
    width_phis=None,
    height_terms=None,
    height_amps=None,
    height_phis=None,
):
    lines = []
    lines.append("=== Bandpass (multi-freq) + LS fit expressions ===\n")
    lines.append("Form: y(t) = sum_k [a_k cos(2pi f_k t) + b_k sin(2pi f_k t)]\n")
    lines.append("     = sum_k [A_k cos(2pi f_k t - phi_k)]\n\n")

    lines.append("[dx (LS-multi)] components (amp-phase parameters):\n")
    for i, f0 in enumerate(freqs_hz):
        lines.append(f"- f{i+1} = {f0:.6g} Hz | A{i+1} = {x_amps[i]:.6g} | phi{i+1} = {x_phis[i]:.6g} rad\n")

    lines.append("\n[dy (LS-multi)] components (amp-phase parameters):\n")
    for i, f0 in enumerate(freqs_hz):
        lines.append(f"- f{i+1} = {f0:.6g} Hz | A{i+1} = {y_amps[i]:.6g} | phi{i+1} = {y_phis[i]:.6g} rad\n")

    if area_amps is not None and area_phis is not None:
        lines.append("\n[darea (LS-multi)] components (amp-phase parameters):\n")
        for i, f0 in enumerate(freqs_hz):
            lines.append(f"- f{i+1} = {f0:.6g} Hz | A{i+1} = {area_amps[i]:.6g} | phi{i+1} = {area_phis[i]:.6g} rad\n")

    if width_amps is not None and width_phis is not None:
        lines.append("\n[dwidth (LS-multi)] components (amp-phase parameters):\n")
        for i, f0 in enumerate(freqs_hz):
            lines.append(f"- f{i+1} = {f0:.6g} Hz | A{i+1} = {width_amps[i]:.6g} | phi{i+1} = {width_phis[i]:.6g} rad\n")

    if height_amps is not None and height_phis is not None:
        lines.append("\n[dheight (LS-multi)] components (amp-phase parameters):\n")
        for i, f0 in enumerate(freqs_hz):
            lines.append(f"- f{i+1} = {f0:.6g} Hz | A{i+1} = {height_amps[i]:.6g} | phi{i+1} = {height_phis[i]:.6g} rad\n")

    lines.append("\n[x_fit_total(t)]\n")
    lines.append("x_fit_total(t) ~= " + (" + ".join(x_terms) if x_terms else "0") + "\n\n")
    lines.append("[y_fit_total(t)]\n")
    lines.append("y_fit_total(t) ~= " + (" + ".join(y_terms) if y_terms else "0") + "\n")
    if area_terms is not None:
        lines.append("\n[area_fit_total(t)]\n")
        lines.append("area_fit_total(t) ~= " + (" + ".join(area_terms) if area_terms else "0") + "\n")
    if width_terms is not None:
        lines.append("\n[width_fit_total(t)]\n")
        lines.append("width_fit_total(t) ~= " + (" + ".join(width_terms) if width_terms else "0") + "\n")
    if height_terms is not None:
        lines.append("\n[height_fit_total(t)]\n")
        lines.append("height_fit_total(t) ~= " + (" + ".join(height_terms) if height_terms else "0") + "\n")

    Path(outpath).write_text("".join(lines), encoding="utf-8")


def display_path(path: Path, base: Path = DISPLACEMENT_DIR.parent) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, RuntimeError, ValueError):
        return path.name


def _pick_first_available_col(df, preferred, candidates, label):
    if preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            print(f"[WARN] {label} column '{preferred}' not found, using '{c}' instead.")
            return c
    raise KeyError(f"{label} column not found. Tried: {preferred} and {candidates}. Available: {list(df.columns)}")


def _pick_optional_col(df, preferred, candidates, label):
    try:
        return _pick_first_available_col(df, preferred, candidates, label)
    except KeyError:
        print(f"[INFO] {label} column not found. Skipping {label.lower()} branch.")
        return None


def _filter_and_select_rows(df, t_col):
    if INPUT_POLARITY and "polarity" in df.columns:
        pol = df["polarity"].astype(str).str.lower()
        df_pol = df[pol == str(INPUT_POLARITY).lower()].copy()
        if df_pol.empty:
            print(f"[WARN] No rows for polarity='{INPUT_POLARITY}'. Using all polarities.")
        else:
            df = df_pol

    if df[t_col].duplicated().any():
        print("[WARN] Multiple spots per time detected. Selecting one per time.")
        if "delta_disp_nm" in df.columns:
            df = df.sort_values([t_col, "delta_disp_nm"], ascending=[True, True])
            df = df.groupby(t_col, as_index=False).first()
        elif "disp_t0_nm" in df.columns:
            df = df.sort_values([t_col, "disp_t0_nm"], ascending=[True, True])
            df = df.groupby(t_col, as_index=False).first()
        else:
            df = df.sort_values(t_col).drop_duplicates(t_col, keep="first")
    return df


# ============================================================
# Main
# ============================================================

def process_dataset(csv_path: Path, out_dir: Path, with_area: bool, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"[{label}] Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if COL_T not in df.columns:
        cand = [c for c in df.columns if c.lower() in ("t", "time", "timestamp", "time_s", "time_ns", "time_ps")]
        if not cand:
            raise KeyError(f"[{label}] Time column '{COL_T}' not found. Available columns: {list(df.columns)}")
        print(f"[WARN] [{label}] COL_T='{COL_T}' not found, using '{cand[0]}' instead.")
        t_col = cand[0]
    else:
        t_col = COL_T

    df = _filter_and_select_rows(df, t_col)

    x_col = _pick_first_available_col(df, COL_X, ["delta_col_nm", "dcol_t0_nm", "delta_col_px", "dcol_t0_px"], "X")
    y_col = _pick_first_available_col(df, COL_Y, ["delta_row_nm", "drow_t0_nm", "delta_row_px", "drow_t0_px"], "Y")
    area_col = None
    if with_area:
        area_col = _pick_first_available_col(df, COL_AREA, ["delta_area_nm2", "area_nm2", "delta_area_px2", "area_px2"], "AREA")
    width_col = _pick_optional_col(df, COL_WIDTH, ["delta_width_nm", "delta_width_px", "image2_width_px"], "WIDTH")
    height_col = _pick_optional_col(df, COL_HEIGHT, ["delta_height_nm", "delta_height_px", "image2_height_px"], "HEIGHT")
    with_size = width_col is not None and height_col is not None
    if width_col is None and height_col is not None:
        print("[WARN] HEIGHT column found without WIDTH column. Skipping width/height branch.")
    if height_col is None and width_col is not None:
        print("[WARN] WIDTH column found without HEIGHT column. Skipping width/height branch.")
    if not with_size:
        width_col = None
        height_col = None

    t_raw = df[t_col].to_numpy(dtype=float)
    x_raw = df[x_col].to_numpy(dtype=float)
    y_raw = df[y_col].to_numpy(dtype=float)
    area_raw = df[area_col].to_numpy(dtype=float) if with_area else None
    width_raw = df[width_col].to_numpy(dtype=float) if with_size else None
    height_raw = df[height_col].to_numpy(dtype=float) if with_size else None

    t = t_raw * unit_scale(TIME_UNIT)
    order = np.argsort(t)
    t = t[order]
    x = x_raw[order]
    y = y_raw[order]
    area = area_raw[order] if with_area else None
    width = width_raw[order] if with_size else None
    height = height_raw[order] if with_size else None

    if FLIP_Y_TO_PHYSICAL:
        y = -y

    dt = np.median(np.diff(t))
    if not np.all(np.isfinite([dt])) or dt <= 0:
        raise ValueError(f"[{label}] Invalid dt from time column. Check TIME_UNIT and time data.")

    if DETREND_LINEAR:
        x_d, _ = detrend_linear(t, x)
        y_d, _ = detrend_linear(t, y)
        area_d = detrend_linear(t, area)[0] if with_area else None
        width_d = detrend_linear(t, width)[0] if with_size else None
        height_d = detrend_linear(t, height)[0] if with_size else None
    else:
        x_d = x.copy()
        y_d = y.copy()
        area_d = area.copy() if with_area else None
        width_d = width.copy() if with_size else None
        height_d = height.copy() if with_size else None

    series_out = {"t_s": t, "x_detr": x_d, "y_detr": y_d}
    if with_area:
        series_out["area_detr"] = area_d
    if with_size:
        series_out["width_detr"] = width_d
        series_out["height_detr"] = height_d

    x_bp_total = np.zeros_like(x_d)
    y_bp_total = np.zeros_like(y_d)
    area_bp_total = np.zeros_like(area_d) if with_area else None
    width_bp_total = np.zeros_like(width_d) if with_size else None
    height_bp_total = np.zeros_like(height_d) if with_size else None
    handedness_per_freq = {}

    for f0 in FREQS_HZ:
        hw = BAND_HALF_WIDTH_HZ.get(f0, 0.5e9)
        x_bp_f = fft_bandpass_real(x_d, dt, f0, hw, use_hann=USE_HANN)
        y_bp_f = fft_bandpass_real(y_d, dt, f0, hw, use_hann=USE_HANN)
        x_bp_total += x_bp_f
        y_bp_total += y_bp_f
        series_out[f"x_bp_{f0/1e9:.2f}GHz"] = x_bp_f
        series_out[f"y_bp_{f0/1e9:.2f}GHz"] = y_bp_f

        if with_area:
            area_bp_f = fft_bandpass_real(area_d, dt, f0, hw, use_hann=USE_HANN)
            area_bp_total += area_bp_f
            series_out[f"area_bp_{f0/1e9:.2f}GHz"] = area_bp_f
        if with_size:
            width_bp_f = fft_bandpass_real(width_d, dt, f0, hw, use_hann=USE_HANN)
            height_bp_f = fft_bandpass_real(height_d, dt, f0, hw, use_hann=USE_HANN)
            width_bp_total += width_bp_f
            height_bp_total += height_bp_f
            series_out[f"width_bp_{f0/1e9:.2f}GHz"] = width_bp_f
            series_out[f"height_bp_{f0/1e9:.2f}GHz"] = height_bp_f

        rot_f, rot_val_f = handedness_from_complex(t, x_bp_f, y_bp_f)
        handedness_per_freq[f0] = (rot_f, rot_val_f)

    series_out["x_bp_total"] = x_bp_total
    series_out["y_bp_total"] = y_bp_total
    if with_area:
        series_out["area_bp_total"] = area_bp_total
    if with_size:
        series_out["width_bp_total"] = width_bp_total
        series_out["height_bp_total"] = height_bp_total

    total_snr = {}
    fit_params = {}
    x_terms = []
    y_terms = []
    area_terms = [] if with_area else None
    width_terms = [] if with_size else None
    height_terms = [] if with_size else None
    x_amps = x_phis = y_amps = y_phis = None
    area_amps = area_phis = None
    width_amps = width_phis = None
    height_amps = height_phis = None

    for f0 in FREQS_HZ:
        rot_f, rot_val_f = handedness_per_freq[f0]
        fit_params[f"handedness_bp_{f0/1e9:.2f}GHz"] = rot_f
        fit_params[f"handedness_bp_metric_{f0/1e9:.2f}GHz"] = float(rot_val_f)

    if DO_SINGLE_FREQ_FIT:
        x_coeffs, x_amps, x_phis, x_fit_total = ls_fit_multi_cos_sin(t, x_bp_total, FREQS_HZ)
        y_coeffs, y_amps, y_phis, y_fit_total = ls_fit_multi_cos_sin(t, y_bp_total, FREQS_HZ)
        if with_area:
            area_coeffs, area_amps, area_phis, area_fit_total = ls_fit_multi_cos_sin(t, area_bp_total, FREQS_HZ)
        else:
            area_coeffs = [None] * len(FREQS_HZ)
            area_fit_total = None
        if with_size:
            width_coeffs, width_amps, width_phis, width_fit_total = ls_fit_multi_cos_sin(t, width_bp_total, FREQS_HZ)
            height_coeffs, height_amps, height_phis, height_fit_total = ls_fit_multi_cos_sin(t, height_bp_total, FREQS_HZ)
        else:
            width_coeffs = [None] * len(FREQS_HZ)
            height_coeffs = [None] * len(FREQS_HZ)
            width_fit_total = None
            height_fit_total = None

        series_out["x_fit_total"] = x_fit_total
        series_out["y_fit_total"] = y_fit_total
        series_out["x_resid_total"] = x_d - x_fit_total
        series_out["y_resid_total"] = y_d - y_fit_total
        if with_area:
            series_out["area_fit_total"] = area_fit_total
            series_out["area_resid_total"] = area_d - area_fit_total
        if with_size:
            series_out["width_fit_total"] = width_fit_total
            series_out["width_resid_total"] = width_d - width_fit_total
            series_out["height_fit_total"] = height_fit_total
            series_out["height_resid_total"] = height_d - height_fit_total

        noise_x_total = robust_noise_std(series_out["x_resid_total"])
        noise_y_total = robust_noise_std(series_out["y_resid_total"])
        noise_xy_total = float(np.hypot(noise_x_total, noise_y_total))

        amp_rms_x_total = float(np.sqrt(np.mean(x_fit_total ** 2)))
        amp_rms_y_total = float(np.sqrt(np.mean(y_fit_total ** 2)))
        amp_rms_xy_total = float(np.sqrt(np.mean(x_fit_total ** 2 + y_fit_total ** 2)))

        snr_x_total_lin = amp_rms_x_total / (noise_x_total + 1e-15)
        snr_y_total_lin = amp_rms_y_total / (noise_y_total + 1e-15)
        snr_xy_total_lin = amp_rms_xy_total / (noise_xy_total + 1e-15)

        total_snr = {
            "noise_x_total_nm": noise_x_total,
            "noise_y_total_nm": noise_y_total,
            "noise_xy_total_nm": noise_xy_total,
            "amp_rms_x_total_nm": amp_rms_x_total,
            "amp_rms_y_total_nm": amp_rms_y_total,
            "amp_rms_xy_total_nm": amp_rms_xy_total,
            "snr_x_total_linear": float(snr_x_total_lin),
            "snr_y_total_linear": float(snr_y_total_lin),
            "snr_xy_total_linear": float(snr_xy_total_lin),
            "snr_x_total_db": float(20 * np.log10(snr_x_total_lin + 1e-15)),
            "snr_y_total_db": float(20 * np.log10(snr_y_total_lin + 1e-15)),
            "snr_xy_total_db": float(20 * np.log10(snr_xy_total_lin + 1e-15)),
        }

        if with_area:
            noise_area_total = robust_noise_std(series_out["area_resid_total"])
            amp_rms_area_total = float(np.sqrt(np.mean(area_fit_total ** 2)))
            snr_area_total_lin = amp_rms_area_total / (noise_area_total + 1e-15)
            total_snr.update({
                "noise_area_total_nm2": noise_area_total,
                "amp_rms_area_total_nm2": amp_rms_area_total,
                "snr_area_total_linear": float(snr_area_total_lin),
                "snr_area_total_db": float(20 * np.log10(snr_area_total_lin + 1e-15)),
            })
        if with_size:
            noise_width_total = robust_noise_std(series_out["width_resid_total"])
            amp_rms_width_total = float(np.sqrt(np.mean(width_fit_total ** 2)))
            snr_width_total_lin = amp_rms_width_total / (noise_width_total + 1e-15)
            noise_height_total = robust_noise_std(series_out["height_resid_total"])
            amp_rms_height_total = float(np.sqrt(np.mean(height_fit_total ** 2)))
            snr_height_total_lin = amp_rms_height_total / (noise_height_total + 1e-15)
            total_snr.update({
                "noise_width_total_nm": noise_width_total,
                "amp_rms_width_total_nm": amp_rms_width_total,
                "snr_width_total_linear": float(snr_width_total_lin),
                "snr_width_total_db": float(20 * np.log10(snr_width_total_lin + 1e-15)),
                "noise_height_total_nm": noise_height_total,
                "amp_rms_height_total_nm": amp_rms_height_total,
                "snr_height_total_linear": float(snr_height_total_lin),
                "snr_height_total_db": float(20 * np.log10(snr_height_total_lin + 1e-15)),
            })

        Pcw_sum = 0.0
        Pccw_sum = 0.0
        for i, f0 in enumerate(FREQS_HZ):
            ax, bx = x_coeffs[i]
            ay, by = y_coeffs[i]
            Ax = x_amps[i]
            phix = x_phis[i]
            Ay = y_amps[i]
            phiy = y_phis[i]

            fit_params[f"fit_x_a_cos_{f0/1e9:.2f}GHz"] = float(ax)
            fit_params[f"fit_x_b_sin_{f0/1e9:.2f}GHz"] = float(bx)
            fit_params[f"fit_x_A_{f0/1e9:.2f}GHz"] = float(Ax)
            fit_params[f"fit_x_phi_rad_{f0/1e9:.2f}GHz"] = float(phix)
            fit_params[f"fit_y_a_cos_{f0/1e9:.2f}GHz"] = float(ay)
            fit_params[f"fit_y_b_sin_{f0/1e9:.2f}GHz"] = float(by)
            fit_params[f"fit_y_A_{f0/1e9:.2f}GHz"] = float(Ay)
            fit_params[f"fit_y_phi_rad_{f0/1e9:.2f}GHz"] = float(phiy)

            if with_area:
                aa, ba = area_coeffs[i]
                Aarea = area_amps[i]
                phiarea = area_phis[i]
                fit_params[f"fit_area_a_cos_{f0/1e9:.2f}GHz"] = float(aa)
                fit_params[f"fit_area_b_sin_{f0/1e9:.2f}GHz"] = float(ba)
                fit_params[f"fit_area_A_{f0/1e9:.2f}GHz"] = float(Aarea)
                fit_params[f"fit_area_phi_rad_{f0/1e9:.2f}GHz"] = float(phiarea)
            if with_size:
                aw, bw = width_coeffs[i]
                Awidth = width_amps[i]
                phiwidth = width_phis[i]
                fit_params[f"fit_width_a_cos_{f0/1e9:.2f}GHz"] = float(aw)
                fit_params[f"fit_width_b_sin_{f0/1e9:.2f}GHz"] = float(bw)
                fit_params[f"fit_width_A_{f0/1e9:.2f}GHz"] = float(Awidth)
                fit_params[f"fit_width_phi_rad_{f0/1e9:.2f}GHz"] = float(phiwidth)

                ah, bh = height_coeffs[i]
                Aheight = height_amps[i]
                phiheight = height_phis[i]
                fit_params[f"fit_height_a_cos_{f0/1e9:.2f}GHz"] = float(ah)
                fit_params[f"fit_height_b_sin_{f0/1e9:.2f}GHz"] = float(bh)
                fit_params[f"fit_height_A_{f0/1e9:.2f}GHz"] = float(Aheight)
                fit_params[f"fit_height_phi_rad_{f0/1e9:.2f}GHz"] = float(phiheight)

            delta_fit = float(np.arctan2(np.sin(phiy - phix), np.cos(phiy - phix)))
            fit_params[f"fit_delta_phi_{f0/1e9:.2f}GHz_rad"] = delta_fit

            Zp, Zm, P_ccw, P_cw, frac_ccw, frac_cw = cw_ccw_fraction_from_ab(ax, bx, ay, by)
            fit_params[f"circ_P_ccw_{f0/1e9:.2f}GHz"] = float(P_ccw)
            fit_params[f"circ_P_cw_{f0/1e9:.2f}GHz"] = float(P_cw)
            fit_params[f"circ_frac_ccw_{f0/1e9:.2f}GHz"] = float(frac_ccw)
            fit_params[f"circ_frac_cw_{f0/1e9:.2f}GHz"] = float(frac_cw)
            fit_params[f"circ_Zp_re_{f0/1e9:.2f}GHz"] = float(Zp.real)
            fit_params[f"circ_Zp_im_{f0/1e9:.2f}GHz"] = float(Zp.imag)
            fit_params[f"circ_Zm_re_{f0/1e9:.2f}GHz"] = float(Zm.real)
            fit_params[f"circ_Zm_im_{f0/1e9:.2f}GHz"] = float(Zm.imag)

            Pcw_sum += float(P_cw)
            Pccw_sum += float(P_ccw)
            x_terms.append(f"({Ax:.6g})*cos(2pi*{f0:.6g}*t - ({phix:.6g}))")
            y_terms.append(f"({Ay:.6g})*cos(2pi*{f0:.6g}*t - ({phiy:.6g}))")
            if with_area:
                area_terms.append(f"({Aarea:.6g})*cos(2pi*{f0:.6g}*t - ({phiarea:.6g}))")
            if with_size:
                width_terms.append(f"({Awidth:.6g})*cos(2pi*{f0:.6g}*t - ({phiwidth:.6g}))")
                height_terms.append(f"({Aheight:.6g})*cos(2pi*{f0:.6g}*t - ({phiheight:.6g}))")

        den = Pcw_sum + Pccw_sum
        fit_params["circ_P_cw_sum"] = float(Pcw_sum)
        fit_params["circ_P_ccw_sum"] = float(Pccw_sum)
        fit_params["circ_frac_cw_sum"] = float(Pcw_sum / den) if den > 0 else np.nan
        fit_params["circ_frac_ccw_sum"] = float(Pccw_sum / den) if den > 0 else np.nan

    snr_lines = []
    snr_lines.append("=== SNR / Noise floor report ===")
    snr_lines.append(f"Label: {label}")
    snr_lines.append(f"CSV: {display_path(csv_path)}")
    snr_lines.append(f"Time unit input: {TIME_UNIT}  -> converted to seconds")
    snr_lines.append(f"dt = {dt:.3e} s, total T = {(t[-1] - t[0]):.3e} s, df~={1 / (t[-1] - t[0]):.3e} Hz")
    snr_lines.append(f"Detrend linear: {DETREND_LINEAR}, Hann window in bandpass: {USE_HANN}")
    snr_lines.append(f"Flip Y to physical: {FLIP_Y_TO_PHYSICAL} (y = -drow)")
    snr_lines.append(f"Area branch enabled: {with_area}")
    snr_lines.append(f"Width/Height branch enabled: {with_size}")
    snr_lines.append("")
    snr_lines.append("=== Handedness per frequency (from bandpass components) ===")
    for f0 in FREQS_HZ:
        rot_f, rot_val_f = handedness_per_freq[f0]
        snr_lines.append(f"[{f0/1e9:.2f} GHz] handedness={rot_f} | metric_mean_Im(conj(z)*dz/dt)={rot_val_f:.6g}")
    snr_lines.append("")

    if DO_SINGLE_FREQ_FIT and total_snr:
        noise_line = (
            f"Noise std (robust, residual = detrended - total fit): "
            f"noise_x={total_snr['noise_x_total_nm']:.4g} nm, "
            f"noise_y={total_snr['noise_y_total_nm']:.4g} nm, "
        )
        if with_area:
            noise_line += f"noise_area={total_snr['noise_area_total_nm2']:.4g} nm^2, "
        if with_size:
            noise_line += (
                f"noise_width={total_snr['noise_width_total_nm']:.4g} nm, "
                f"noise_height={total_snr['noise_height_total_nm']:.4g} nm, "
            )
        noise_line += f"noise_xy={total_snr['noise_xy_total_nm']:.4g} nm"
        snr_lines.append(noise_line)
        snr_lines.append("")
        snr_lines.append("=== TOTAL fit (sum of target freqs) SNR vs original(detr) ===")
        snr_lines.append(f"[TOTAL] RMS(x_fit)={total_snr['amp_rms_x_total_nm']:.4g} nm | noise_x_total={total_snr['noise_x_total_nm']:.4g} nm | SNRx_total={total_snr['snr_x_total_linear']:.3g} ({total_snr['snr_x_total_db']:.2f} dB)")
        snr_lines.append(f"[TOTAL] RMS(y_fit)={total_snr['amp_rms_y_total_nm']:.4g} nm | noise_y_total={total_snr['noise_y_total_nm']:.4g} nm | SNRy_total={total_snr['snr_y_total_linear']:.3g} ({total_snr['snr_y_total_db']:.2f} dB)")
        if with_area:
            snr_lines.append(f"[TOTAL] RMS(area_fit)={total_snr['amp_rms_area_total_nm2']:.4g} nm^2 | noise_area_total={total_snr['noise_area_total_nm2']:.4g} nm^2 | SNRarea_total={total_snr['snr_area_total_linear']:.3g} ({total_snr['snr_area_total_db']:.2f} dB)")
        if with_size:
            snr_lines.append(f"[TOTAL] RMS(width_fit)={total_snr['amp_rms_width_total_nm']:.4g} nm | noise_width_total={total_snr['noise_width_total_nm']:.4g} nm | SNRwidth_total={total_snr['snr_width_total_linear']:.3g} ({total_snr['snr_width_total_db']:.2f} dB)")
            snr_lines.append(f"[TOTAL] RMS(height_fit)={total_snr['amp_rms_height_total_nm']:.4g} nm | noise_height_total={total_snr['noise_height_total_nm']:.4g} nm | SNRheight_total={total_snr['snr_height_total_linear']:.3g} ({total_snr['snr_height_total_db']:.2f} dB)")
        snr_lines.append(f"[TOTAL] RMS(xy_fit)={total_snr['amp_rms_xy_total_nm']:.4g} nm | noise_xy_total={total_snr['noise_xy_total_nm']:.4g} nm | SNRxy_total={total_snr['snr_xy_total_linear']:.3g} ({total_snr['snr_xy_total_db']:.2f} dB)")
        snr_lines.append("")
        snr_lines.append("=== CW/CCW fractions (from fitted coefficients, power = |Z|^2) ===")
        for f0 in FREQS_HZ:
            snr_lines.append(
                f"[{f0/1e9:.2f} GHz] frac_CW={fit_params.get(f'circ_frac_cw_{f0/1e9:.2f}GHz', np.nan):.3f} | "
                f"frac_CCW={fit_params.get(f'circ_frac_ccw_{f0/1e9:.2f}GHz', np.nan):.3f} | "
                f"P_CW={fit_params.get(f'circ_P_cw_{f0/1e9:.2f}GHz', np.nan):.6g} | "
                f"P_CCW={fit_params.get(f'circ_P_ccw_{f0/1e9:.2f}GHz', np.nan):.6g}"
            )
        snr_lines.append(f"[SUM] frac_CW={fit_params.get('circ_frac_cw_sum', np.nan):.3f} | frac_CCW={fit_params.get('circ_frac_ccw_sum', np.nan):.3f}")

    pd.DataFrame(series_out).to_csv(out_dir / "bandpassed_timeseries.csv", index=False)
    fit_params_row = {"freqs_Hz": ",".join([f"{f:.6g}" for f in FREQS_HZ]), **fit_params}
    pd.DataFrame([fit_params_row]).to_csv(out_dir / "fit_params.csv", index=False)
    if DO_SINGLE_FREQ_FIT and total_snr:
        pd.DataFrame([total_snr]).to_csv(out_dir / "snr_total.csv", index=False)
    (out_dir / "snr_report.txt").write_text("\n".join(snr_lines), encoding="utf-8")

    if DO_SINGLE_FREQ_FIT:
        write_fit_expressions_all_txt(
            FREQS_HZ,
            x_terms,
            y_terms,
            x_amps,
            x_phis,
            y_amps,
            y_phis,
            out_dir / "fit_expressions_all.txt",
            area_terms=area_terms,
            area_amps=area_amps,
            area_phis=area_phis,
            width_terms=width_terms,
            width_amps=width_amps,
            width_phis=width_phis,
            height_terms=height_terms,
            height_amps=height_amps,
            height_phis=height_phis,
        )

    print(f"[DONE] {label}")
    print(f"Outputs saved to: {display_path(out_dir)}")
    print("Key outputs:")
    print("- bandpassed_timeseries.csv")
    print("- fit_params.csv   (includes CW/CCW fractions)")
    print("- snr_report.txt   (prints CW/CCW fractions)")
    if DO_SINGLE_FREQ_FIT and total_snr:
        print("- snr_total.csv")
        print("- fit_expressions_all.txt")


def main():
    source = str(INPUT_SOURCE).lower()
    jobs = []
    if source == "input1":
        jobs.append(("input1", CSV_PATH_INPUT1, OUT_DIR_INPUT1, True))
    elif source == "input2":
        jobs.append(("input2", CSV_PATH_INPUT2, OUT_DIR_INPUT2, False))
    elif source == "both":
        jobs.append(("input1", CSV_PATH_INPUT1, OUT_DIR_INPUT1, True))
        jobs.append(("input2", CSV_PATH_INPUT2, OUT_DIR_INPUT2, False))
    else:
        raise ValueError("INPUT_SOURCE must be 'input1', 'input2', or 'both'.")

    for label, csv_path, out_dir, with_area in jobs:
        process_dataset(csv_path, out_dir, with_area, label)


if __name__ == "__main__":
    main()
