# PIP-FUCCI cell-cycle phase analysis

Code used to quantify G1, S and G2/M durations from live-cell PIP-FUCCI imaging (Incucyte) in WT, p53KO, Per2KO and p53/Per2 double-KO (dKO) cells.

## Upstream steps (not included in this repository)

1. Time-lapse imaging on an Incucyte (see Methods for details and imaging set up)
2. Raw image export, then manual single-cell tracking (nucleus coordinates recorded per frame).
3. Extraction of green (CDT1) and red (Geminin) nuclear intensities at the tracked coordinates on the original images.
4. Per-cell normalization of each channel to 0–1, and merging of all cells of a genotype into one CSV.

This repository starts from the merged CSV.

## Repository contents

| File | Purpose |
|---|---|
| `pip_fucci_analyzer.py` | Core analysis: QC, signal smoothing, G1/S and S/G2 transition detection, manual exclusion, outlier detection, plots |
| `improved_workflow.py` | Workflow wrapper: runs the analysis, saves manual exclusions, exports plot-ready tables |
| `cell_review_jupyter.py` | Interactive trace-by-trace review used for manual curation |
| `PIP_FUCCI_analysis.ipynb` | Notebook that runs the full analysis for one genotype |
| `exclusions/<genotype>.json` | Cells excluded during manual review for the published analysis || `data/` | Place input CSVs here (see below) |

## Input format

One CSV per genotype, one row per cell per timepoint:

| Column | Description |
|---|---|
| `Cell_ID` | Unique cell identifier, e.g. `Well4_WT_Field3_Cell1` |
| `Time` | Time in hours |
| `CDT1_Signal` | Normalized green (PIP/CDT1 degron) intensity, 0–1 |
| `Geminin_Signal` | Normalized red (Geminin) intensity, 0–1 |
| `Green_Raw`, `Red_Raw` | Raw intensities (used for raw-signal QC) |

Expected file names: `data/combined_all_conditions_<genotype>.csv`, where genotype is `WT`, `p53KO`, `Per2KO` or `dKO`.

## Method summary

Signals are smoothed with a Gaussian filter (sigma = 1 frame).

**Quality control**
- Raw-signal QC: a cell is rejected if both its maximum raw green and its maximum raw red intensity are below the 10th percentile of all cells in that dataset. If only one channel is below its threshold, the cell is flagged but kept, and the normalized checks below are skipped for it.
- Normalized QC: a cell is rejected for low dynamic range (< 0.15 in either channel), for not starting in G1 (CDT1/Geminin ratio < 2, or Geminin > CDT1, in the first 1.5 h), for fewer than 10 frames, for tracking gaps (> 2 times the median interval), or for non-monotonic or duplicate timepoints.
- Cells with a detected G1 shorter than 1 h are rejected.

**Transition detection**
- **G1/S:** the first time smoothed CDT1 signal stays below 50% of its maximum in the first 3 h, for at least 0.5 h.
- **S/G2:** the search starts at least 3 h after G1/S, from the CDT1 minimum. S/G2 is the first time CDT1 stays above that minimum + 0.07 for at least 0.5 h (sensitivity `medium`).

**Phase durations**
- G1 = G1/S − track start
- S = S/G2 − G1/S
- G2/M = track end − S/G2

**Curation and outliers**
- Remaining cells with a complete S phase were reviewed visually. Excluded cells are listed in `exclusions/`.
- Outliers in S and G2/M duration (outside Q1 − 1.5 IQR to Q3 + 1.5 IQR) were removed.
- The final per-cell table is `results/<genotype>/results_no_outliers.csv`.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook PIP_FUCCI_analysis.ipynb
```

Set `GENOTYPE` in the settings cell and run all cells.

- By default the published exclusions are applied.
- Set `REDO_MANUAL_REVIEW = True` to curate the cells interactively instead.

Outputs (CSVs and SVG/PDF figures) are written to `results/<genotype>/`.

## Requirements

Python 3.12 with numpy, pandas, scipy, matplotlib, seaborn and jupyter (see `requirements.txt`).

## Citation

If you use this code, please cite the paper.
