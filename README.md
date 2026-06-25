# UEM Workflow

This repository contains a cleaned Python workflow for processing image-based UEM data and extracting motion descriptors from fitted contours and dark spot positions. The scripts are organized as numbered steps in `UEM workflow/` and are intended to be run in order.

The repository contains code only. Raw images, intermediate data, and local machine-specific paths are intentionally not included.

## Repository Layout

```text
UEM workflow/
  workflow_paths.py
  1_image_enhance_full.py
  2_extract_centralline_smooth.py
  3_mask.py
  4_fit_skech_full.py
  5_overlay.py
  6_centroid_displacement.py
  7_black_spot_detector.py
  8_fft_bandpass_filter_Pchirality.py
```

## Path Configuration

All scripts use `workflow_paths.py` for shared input and output locations. By default, the workflow expects raw data under:

```text
UEM workflow/data/raw
```

For real runs, set the environment variable `TEM_RAW_ROOT` to the folder that contains the image-frame subfolders. Relative paths are resolved from the script directory.

```powershell
$env:TEM_RAW_ROOT = "path/to/your/raw/data"
```

Derived outputs are written under the configured raw-data root, mainly in the `displacement/` subfolder.

## Workflow Steps

### 1. Image Enhancement

`1_image_enhance_full.py`

Enhances raw image frames and writes processed image outputs for downstream contour extraction. This step standardizes contrast and prepares the images used by the later shape-analysis steps.

### 2. Centerline Extraction and Smoothing

`2_extract_centralline_smooth.py`

Extracts the main centerline or boundary-related curve from the enhanced image output, then applies smoothing to produce a more stable curve representation for masking and fitting.

### 3. Mask Generation

`3_mask.py`

Generates mask images for the target region. The masks define the region used for contour fitting and help exclude irrelevant background or non-target structures.

### 4. Fitted Curve Generation

`4_fit_skech_full.py`

Fits and regularizes the target contour or curve from the masked image data. The fitted result is used as the primary outline for later overlay visualization and centroid-based displacement measurements.

### 5. Overlay Generation

`5_overlay.py`

Creates overlay images that combine fitted curves with grayscale image data. These overlays are useful for checking whether the fitted contour follows the intended image feature.

### 6. Centroid Displacement Measurement

`6_centroid_displacement.py`

Measures frame-by-frame centroid displacement from the fitted outline images. The script uses a baseline frame as the displacement reference and outputs `centroid_results_all.csv` with pixel and nanometer-scale displacement, area, width, and height changes.

### 7. Dark Spot Detection

`7_black_spot_detector.py`

Detects the darkest region inside the fitted polygon and records its position. The script outputs `spots_inside_polygon_single.csv`, including spot coordinates, distance to the frame center, and displacement relative to the first frame.

### 8. FFT Bandpass and Chirality Analysis

`8_fft_bandpass_filter_Pchirality.py`

Applies FFT-domain bandpass filtering to the displacement time series, fits target-frequency cosine/sine components, estimates handedness, decomposes fitted motion into clockwise and counter-clockwise circular components, and writes CSV/TXT summaries. PNG plotting output is not generated in this cleaned version.

## Typical Run Order

Run the scripts from the `UEM workflow/` folder in numerical order:

```powershell
python 1_image_enhance_full.py
python 2_extract_centralline_smooth.py
python 3_mask.py
python 4_fit_skech_full.py
python 5_overlay.py
python 6_centroid_displacement.py
python 7_black_spot_detector.py
python 8_fft_bandpass_filter_Pchirality.py
```

Before running, review the parameter block at the top of each script and adjust mode, frame range, time step, and pixel-to-nanometer scale as needed for the dataset.

## Main Outputs

Common downstream outputs include:

- `displacement/centroid_results_all.csv`
- `displacement/spots_inside_polygon_single.csv`
- `displacement/fft_bandpass/askyframe/bandpassed_timeseries.csv`
- `displacement/fft_bandpass/askyframe/fit_params.csv`
- `displacement/fft_bandpass/askyframe/snr_report.txt`
- `displacement/fft_bandpass/vortex_core/bandpassed_timeseries.csv`
- `displacement/fft_bandpass/vortex_core/fit_params.csv`
- `displacement/fft_bandpass/vortex_core/snr_report.txt`

## Notes

- The scripts use generic sample folder names and shared path configuration to avoid hard-coded local paths.
- Raw data and generated results are not tracked in this repository.
- Review scale and timing parameters before applying the workflow to a new dataset.
