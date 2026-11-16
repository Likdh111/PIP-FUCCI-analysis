# Data: PIP-FUCCI single-cell intensity traces

Input data for `PIP_FUCCI_analysis.ipynb`. There is one file per genotype, with one row per cell per frame. The published analysis results are in `../published_results/`, and the manually excluded cells are in `../exclusions/`.

## Experiment summary
- **Imaging:** Incucyte [model], [objective], one image every 15 min, for [duration]. See the paper's Methods.
- **Reporter:** PIP-FUCCI. The green channel is the PIP degron reporter ("CDT1"); the red channel is Geminin.
- **Tracking:** cells were tracked manually from birth to the following mitosis. Mean nuclear intensity was measured at each frame in a fixed-size region (area 366 px) centred on the tracked nucleus, on the original, unscaled images.

## Files

| File | Contents |
|---|---|
| `combined_all_conditions_WT.csv` | WT traces (142 cells, 11 fields) |
| `combined_all_conditions_p53KO.csv` | p53KO traces (164 cells, 13 fields) |
| `combined_all_conditions_Per2KO.csv` | Per2KO traces (211 cells, 19 fields) |
| `combined_all_conditions_dKO.csv` | dKO traces (175 cells, 21 fields) |
| `imaging_fields_used.csv` | Incucyte well and image number of every analysed field, with the number of cells tracked and retained per field |

## Columns of `combined_all_conditions_<genotype>.csv`

| Column | Description |
|---|---|
| `Cell_ID` | Unique cell identifier: `Well<n>_<genotype>_Field<n>_Cell<n>` |
| `Time` | Time since the first tracked frame of this cell (h) |
| `Timepoint` | Incucyte frame index of the measurement |
| `Green_Raw` | Mean green (PIP/CDT1 reporter) intensity in the measurement region (a.u.) |
| `Red_Raw` | Mean red (Geminin reporter) intensity in the measurement region (a.u.) |
| `CDT1_Signal` | `Green_Raw`, min-max normalized to 0–1 for each cell |
| `Geminin_Signal` | `Red_Raw`, min-max normalized to 0–1 for each cell |
| `TrackID` | Track number from the tracking step |
| `NucleusX`, `NucleusY` | Tracked nucleus position in the image (pixels) |
| `Well_ID` | Well number within the genotype. Wells 1-3 = rows A-C plate 1; wells 4-6 = rows A-C plate 2 |
| `Condition` | Genotype |
| `Field` | Incucyte image (field) number within the well |
| `Cell_Number` | Cell number within the field |
| `Folder` | Source folder name, formatted `<Incucyte well>_<image number>`. rows D-F = rows A-C of the biological replicate well (e.g D1_9 = plate 2, well A1, field 9) |

## Published results (`../published_results/<genotype>/`)

| File | Contents |
|---|---|
| `pip_fucci_analysis_results.csv` | All tracked cells: QC result (`QC_pass`, `QC_flags`), detected transition times (`G1S_time`, `SG2_time`), `G1_duration`, `S_phase_duration`, and QC metrics |
| `results_no_outliers.csv` | Cells passing QC, manual curation and outlier removal, including cells without a complete S phase |
| `cell_data_wide_format.csv` | Final phase durations (complete cell cycles; used for figures): `G1_duration`, `S_phase_duration`, `G2M_duration` (= `End_time` − `SG2_time`), in hours |
| `summary_statistics.csv` | Mean, SD, median and quartiles of the G1, S and G2/M durations |

Final cell numbers (rows in `cell_data_wide_format.csv`): WT 74, p53KO 111, Per2KO 76, dKO 121.

Running `PIP_FUCCI_analysis.ipynb` on these input files writes new results to `../results/<genotype>/`. Those results are identical to the files in `../published_results/`.

## Raw images
[Raw time-lapse images are available at: BioImage Archive / Zenodo, accession/DOI …] `imaging_fields_used.csv` lists the Incucyte well and image number for every analysed field.
