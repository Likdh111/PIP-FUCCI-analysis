import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
from pathlib import Path
import logging
import warnings

# ---------------------------
# Logger utilities
# ---------------------------
_logger = logging.getLogger("pip_fucci_analyzer")
if not _logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

# Default level INFO so behavior remains similar to prints
_logger.setLevel(logging.INFO)

# Change how much detail the analyzer prints (e.g. logging.DEBUG for more).
def set_analyzer_logging_level(level):
    """
    Set the logger level for the analyzer. Example: set_analyzer_logging_level(logging.DEBUG)
    """
    _logger.setLevel(level)


# ---------------------------
# SignalLoader: minimal loader + standardization
# ---------------------------
class SignalLoader:
    """
    Loads CSV, standardizes column names expected by the analyzer:
    Required canonical names: Time, CDT1_Signal, Geminin_Signal, Cell_ID
    Optional raw columns: Green_Raw, Red_Raw
    """

    # Read the input CSV and standardize its column names.
    def __init__(self, csv_file_path: str):
        self.csv_file_path = str(csv_file_path)
        self.df = pd.read_csv(self.csv_file_path)
        self._standardize_columns()

    # Rename column variants to Time / CDT1_Signal / Geminin_Signal / Cell_ID / Green_Raw / Red_Raw and check required columns exist.
    def _standardize_columns(self):
        # Map common variants to canonical names
        mapping = {}
        cols = list(self.df.columns)
        for col in cols:
            lc = col.lower()
            if 'time' in lc and 'timepoint' not in lc:
                if 'Time' not in self.df.columns and 'Time' not in mapping.values():
                    mapping[col] = 'Time'
            elif ('green' in lc or 'venus' in lc or 'cdt1' in lc) and 'raw' not in lc:
                if 'CDT1_Signal' not in self.df.columns and 'CDT1_Signal' not in mapping.values():
                    mapping[col] = 'CDT1_Signal'
            elif ('red' in lc or 'cherry' in lc or 'geminin' in lc) and 'raw' not in lc:
                if 'Geminin_Signal' not in self.df.columns and 'Geminin_Signal' not in mapping.values():
                    mapping[col] = 'Geminin_Signal'
            elif 'cell' in lc and 'id' in lc:
                if 'Cell_ID' not in self.df.columns and 'Cell_ID' not in mapping.values():
                    mapping[col] = 'Cell_ID'
            # Raw columns
            elif 'green' in lc and 'raw' in lc:
                mapping[col] = 'Green_Raw'
            elif 'red' in lc and 'raw' in lc:
                mapping[col] = 'Red_Raw'

        if mapping:
            _logger.debug(f"Renaming columns: {mapping}")
            self.df = self.df.rename(columns=mapping)

        # Validate required columns present
        required = ['Time', 'CDT1_Signal', 'Geminin_Signal', 'Cell_ID']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(self.df.columns)}")

        # Force types: Time float, Cell_ID as str for grouping stability
        try:
            self.df['Time'] = self.df['Time'].astype(float)
        except Exception:
            # attempt to coerce
            self.df['Time'] = pd.to_numeric(self.df['Time'], errors='coerce')

        self.df['Cell_ID'] = self.df['Cell_ID'].astype(str)

    # Return a copy of the loaded, standardized table.
    def get_dataframe(self):
        return self.df.copy()


# ---------------------------
# QCModule: raw-intensity QC + normalized-signal QC
# ---------------------------
class QCModule:
    """
    Implements check_raw_signal_quality and check_cell_quality logic.
    Does not change any thresholds or default parameters.
    """
    # Optional diagnostic: histograms of per-cell maximum raw intensity with the QC cutoffs (not used in the main workflow).
    def plot_raw_qc_diagnostic(self):
        """Visualize raw signal QC threshold selection"""
        if not self.use_raw_qc:
            return
        
        cell_maxes = self.df.groupby('Cell_ID').agg({
            'Green_Raw': 'max', 
            'Red_Raw': 'max'
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Green/CDT1
        ax = axes[0]
        ax.hist(cell_maxes['Green_Raw'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(self.green_raw_threshold, color='red', linewidth=2, 
                   label=f'10th percentile: {self.green_raw_threshold:.0f}')
        ax.set_xlabel('Max CDT1 Raw Signal', fontsize=11)
        ax.set_ylabel('Number of Cells', fontsize=11)
        ax.set_title('CDT1/Green Signal Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        
        # Red/Geminin
        ax = axes[1]
        ax.hist(cell_maxes['Red_Raw'], bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(self.red_raw_threshold, color='darkred', linewidth=2,
                   label=f'10th percentile: {self.red_raw_threshold:.0f}')
        ax.set_xlabel('Max Geminin Raw Signal', fontsize=11)
        ax.set_ylabel('Number of Cells', fontsize=11)
        ax.set_title('Geminin/Red Signal Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'raw_qc_thresholds.png', dpi=300)
        print(f"✓ Raw QC diagnostic plot saved")
        plt.show()
        
        # Print statistics
        print("\n" + "="*60)
        print("RAW SIGNAL QC DIAGNOSTIC")
        print("="*60)
        print(f"\nCDT1/Green Channel:")
        print(f"  Mean max: {cell_maxes['Green_Raw'].mean():.1f}")
        print(f"  Median max: {cell_maxes['Green_Raw'].median():.1f}")
        print(f"  10th percentile: {self.green_raw_threshold:.1f}")
        print(f"  Cells below threshold: {(cell_maxes['Green_Raw'] < self.green_raw_threshold).sum()}")
        
        print(f"\nGeminin/Red Channel:")
        print(f"  Mean max: {cell_maxes['Red_Raw'].mean():.1f}")
        print(f"  Median max: {cell_maxes['Red_Raw'].median():.1f}")
        print(f"  10th percentile: {self.red_raw_threshold:.1f}")
        print(f"  Cells below threshold: {(cell_maxes['Red_Raw'] < self.red_raw_threshold).sum()}")
        print("="*60)

    # Store QC settings and, if raw-signal QC is on, determine the raw-intensity cutoffs.
    def __init__(self, df: pd.DataFrame, use_raw_qc=True,
                 green_raw_threshold=None, red_raw_threshold=None):
        self.df = df  # expects standardized df
        self.use_raw_qc = use_raw_qc
        self.green_raw_threshold = green_raw_threshold
        self.red_raw_threshold = red_raw_threshold

        # Auto-detect thresholds if requested and columns present
        if self.use_raw_qc:
            self._auto_detect_raw_thresholds()

    # Set raw-intensity cutoffs to the 10th percentile of per-cell maximum green and red intensity.
    def _auto_detect_raw_thresholds(self):
        # Check presence
        if ('Green_Raw' not in self.df.columns) or ('Red_Raw' not in self.df.columns):
            _logger.warning("Green_Raw/Red_Raw columns not found. Disabling raw QC.")
            self.use_raw_qc = False
            return

        cell_maxes = self.df.groupby('Cell_ID').agg({'Green_Raw': 'max', 'Red_Raw': 'max'})
        if self.green_raw_threshold is None:
            try:
                self.green_raw_threshold = cell_maxes['Green_Raw'].quantile(0.10)
            except Exception:
                self.green_raw_threshold = float(cell_maxes['Green_Raw'].min())

        if self.red_raw_threshold is None:
            try:
                self.red_raw_threshold = cell_maxes['Red_Raw'].quantile(0.10)
            except Exception:
                self.red_raw_threshold = float(cell_maxes['Red_Raw'].min())

        _logger.info(f"\nRaw Signal QC Enabled:")
        _logger.info(f"  Green/CDT1 threshold (10th percentile): {self.green_raw_threshold:.1f}")
        _logger.info(f"  Red/Geminin threshold (10th percentile): {self.red_raw_threshold:.1f}")

    # Flag a cell whose maximum raw green and/or red intensity is below the cutoff.
    def check_raw_signal_quality(self, cell_data: pd.DataFrame):
        """
        Returns (pass_qc: bool, qc_flags: list, metrics: dict)
        """
        if not self.use_raw_qc:
            return True, [], {}

        if 'Green_Raw' not in cell_data.columns or 'Red_Raw' not in cell_data.columns:
            return True, [], {}

        green_max = float(cell_data['Green_Raw'].max())
        red_max = float(cell_data['Red_Raw'].max())

        qc_flags = []
        metrics = {'green_raw_max': green_max, 'red_raw_max': red_max}

        if green_max < self.green_raw_threshold:
            qc_flags.append(f'GREEN_RAW_TOO_LOW({green_max:.0f}<{self.green_raw_threshold:.0f})')
        if red_max < self.red_raw_threshold:
            qc_flags.append(f'RED_RAW_TOO_LOW({red_max:.0f}<{self.red_raw_threshold:.0f})')
        if green_max < self.green_raw_threshold and red_max < self.red_raw_threshold:
            qc_flags.append('BOTH_SIGNALS_WEAK_RAW')

        pass_qc = len(qc_flags) == 0
        return pass_qc, qc_flags, metrics

    # Full per-cell QC (raw intensity, dynamic range, starts in G1, time gaps, track length, saturation); returns pass/fail, flags and metrics.
    def check_cell_quality(self, cell_data: pd.DataFrame, cell_id: str):
        """
        Extended QC: first raw QC (if enabled), then normalized QC.
        Returns same triple: (pass_qc, qc_flags, qc_metrics)
        """
        times = np.array(cell_data['Time'].values, dtype=float)
        cdt1_signal = np.array(cell_data['CDT1_Signal'].values, dtype=float)
        geminin_signal = np.array(cell_data['Geminin_Signal'].values, dtype=float)

        qc_flags = []
        qc_metrics = {}

        # Raw QC first
        raw_pass, raw_flags, raw_metrics = self.check_raw_signal_quality(cell_data)
        qc_flags.extend(raw_flags)
        qc_metrics.update(raw_metrics)

        if not raw_pass:
            # Fill additional metrics for reporting consistency
            qc_metrics.update({
                'cdt1_dynamic_range': float(np.ptp(cdt1_signal)) if len(cdt1_signal) > 0 else np.nan,
                'geminin_dynamic_range': float(np.ptp(geminin_signal)) if len(geminin_signal) > 0 else np.nan,
                'track_duration': float(times[-1] - times[0]) if len(times) > 1 else 0.0,
                'n_frames': int(len(times))
            })
            # decide pass/fail based on critical flags
            critical_flags = [
                'FEW_FRAMES', 'TRACKING_GAP', 'NON_MONOTONIC_TIME',
                'DUPLICATE_TIMEPOINTS', 'NOT_STARTING_IN_G1', 'HIGH_EARLY_GEMININ',
                'CDT1_LOW_DYNAMIC_RANGE', 'GEMININ_LOW_DYNAMIC_RANGE',
                'GREEN_RAW_TOO_LOW', 'RED_RAW_TOO_LOW', 'BOTH_SIGNALS_WEAK_RAW'
            ]
            pass_qc = not any(f in critical_flags for f in qc_flags)
            return pass_qc, qc_flags, qc_metrics

        # Normalized-signal QC
        # Dynamic range
        cdt1_dynamic_range = float(np.max(cdt1_signal) - np.min(cdt1_signal)) if len(cdt1_signal) > 0 else 0.0
        geminin_dynamic_range = float(np.max(geminin_signal) - np.min(geminin_signal)) if len(geminin_signal) > 0 else 0.0

        qc_metrics['cdt1_dynamic_range'] = cdt1_dynamic_range
        qc_metrics['geminin_dynamic_range'] = geminin_dynamic_range

        if cdt1_dynamic_range < 0.15:
            qc_flags.append('CDT1_LOW_DYNAMIC_RANGE')
        if geminin_dynamic_range < 0.15:
            qc_flags.append('GEMININ_LOW_DYNAMIC_RANGE')

        # Early G1 check: cell must start in G1 (CDT1 high, Geminin low)
        if len(times) >= 3:
            early_window = times < (times[0] + 1.5)
            if np.sum(early_window) >= 3:
                early_cdt1 = cdt1_signal[early_window]
                early_geminin = geminin_signal[early_window]
                mean_early_cdt1 = float(np.mean(early_cdt1))
                mean_early_geminin = float(np.mean(early_geminin))
                cdt1_gem_ratio = float(mean_early_cdt1 / mean_early_geminin) if mean_early_geminin > 0.01 else float(mean_early_cdt1 / 0.01)

                qc_metrics['early_CDT1_mean'] = mean_early_cdt1
                qc_metrics['early_Geminin_mean'] = mean_early_geminin
                qc_metrics['early_CDT1_Geminin_ratio'] = cdt1_gem_ratio

                if cdt1_gem_ratio < 2.0:
                    qc_flags.append('NOT_STARTING_IN_G1')
                if mean_early_geminin > mean_early_cdt1:
                    qc_flags.append('HIGH_EARLY_GEMININ')

        # Time issues
        time_diffs = np.diff(times) if len(times) > 1 else np.array([])
        if time_diffs.size > 0:
            if np.any(time_diffs <= 0):
                qc_flags.append('NON_MONOTONIC_TIME')
            if np.any(time_diffs < 0.001):
                qc_flags.append('DUPLICATE_TIMEPOINTS')

            median_interval = float(np.median(time_diffs))
            max_gap = float(np.max(time_diffs))
            qc_metrics['max_gap_hours'] = max_gap
            qc_metrics['median_interval'] = median_interval
            if max_gap > 2 * median_interval:
                qc_flags.append('TRACKING_GAP')
        else:
            qc_metrics['max_gap_hours'] = 0.0
            qc_metrics['median_interval'] = 0.0

        # Track length & frames
        track_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        qc_metrics['track_duration'] = track_duration
        qc_metrics['n_frames'] = int(len(times))
        if track_duration < 8:
            qc_flags.append('SHORT_TRACK')
        if len(times) < 10:
            qc_flags.append('FEW_FRAMES')

        # Saturation checks
        if len(cdt1_signal) > 0 and np.sum(cdt1_signal >= 0.98) > len(cdt1_signal) * 0.1:
            qc_flags.append('CDT1_SATURATED')
        if len(geminin_signal) > 0 and np.sum(geminin_signal >= 0.98) > len(geminin_signal) * 0.1:
            qc_flags.append('GEMININ_SATURATED')

        # Final decision based on critical flags
        critical_flags = [
            'FEW_FRAMES', 'TRACKING_GAP', 'NON_MONOTONIC_TIME',
            'DUPLICATE_TIMEPOINTS', 'NOT_STARTING_IN_G1', 'HIGH_EARLY_GEMININ',
            'CDT1_LOW_DYNAMIC_RANGE', 'GEMININ_LOW_DYNAMIC_RANGE',
            'GREEN_RAW_TOO_LOW', 'RED_RAW_TOO_LOW', 'BOTH_SIGNALS_WEAK_RAW'
        ]
        pass_qc = not any(f in critical_flags for f in qc_flags)
        return pass_qc, qc_flags, qc_metrics


# ---------------------------
# TransitionDetector: smoothing + G1-S and S-G2 detection
# ---------------------------
class TransitionDetector:
    """
    Methods for smoothing signals and detecting G1-S and S-G2 transitions.
    """

    # No stored settings; detection parameters are passed to each method.
    def __init__(self):
        pass

    # Gaussian smoothing of a trace (sigma = kernel_size / 3 frames).
    @staticmethod
    def smooth_signal(signal, kernel_size=3):
        signal = np.asarray(signal).flatten()
        if len(signal) < kernel_size:
            return signal
        return gaussian_filter1d(signal, sigma=kernel_size/3)

    # Find the longest stretch within 0.05 of the trace minimum; return its start index, end index and mean level.
    @staticmethod
    def find_sustained_minimum(signal, times, window_size=5):
        global_min_idx = int(np.argmin(signal))
        min_val = float(signal[global_min_idx])

        threshold = min_val + 0.05
        low_region = signal <= threshold

        stretches = []
        start = None
        for i, is_low in enumerate(low_region):
            if is_low and start is None:
                start = i
            elif not is_low and start is not None:
                stretches.append((start, i - 1))
                start = None
        if start is not None:
            stretches.append((start, len(signal) - 1))

        if not stretches:
            return global_min_idx, global_min_idx, min_val

        longest = max(stretches, key=lambda x: x[1] - x[0])
        sustained_mean = float(np.mean(signal[longest[0]:longest[1] + 1]))
        return int(longest[0]), int(longest[1]), sustained_mean

    # G1/S = first time smoothed CDT1 stays below 50% of its early-G1 (first 3 h) maximum for the sustained-drop window.
    def find_g1s_transition(self, cdt1_signal, geminin_signal, times,
                            min_g1_duration=None, sustained_drop_duration=0.5):
        """
        G1/S transition: CDT1 drops below 50% of its early-G1 maximum.
        Returns dictionary possibly containing G1S_time, G1S_CDT1_value, etc.
        """
        cdt1_smooth = self.smooth_signal(cdt1_signal)
        geminin_smooth = self.smooth_signal(geminin_signal)

        results = {}
        median_interval = np.median(np.diff(times)) if len(times) > 1 else 0.25
        sustained_drop_window = max(2, int(sustained_drop_duration / median_interval))

        # early G1 window (first 3 hours)
        g1_window_hours = 3.0
        time_in_g1 = times < (times[0] + g1_window_hours)
        if np.sum(time_in_g1) < 3:
            return results

        g1_cdt1 = cdt1_smooth[time_in_g1]
        if len(g1_cdt1) == 0:
            return results

        cdt1_max = float(np.max(g1_cdt1))
        threshold_50pct = 0.5 * cdt1_max

        g1s_idx = None
        consecutive_below = 0
        for i in range(len(cdt1_smooth)):
            if cdt1_smooth[i] < threshold_50pct:
                consecutive_below += 1
                if consecutive_below >= sustained_drop_window:
                    g1s_idx = i - sustained_drop_window + 1
                    break
            else:
                consecutive_below = 0

        if g1s_idx is not None:
            results['G1S_time'] = float(times[g1s_idx])
            results['G1S_CDT1_value'] = float(cdt1_smooth[g1s_idx])
            results['G1S_Geminin_value'] = float(geminin_smooth[g1s_idx])
            results['CDT1_max'] = cdt1_max
            results['G1_duration'] = float(times[g1s_idx] - times[0])

        # Geminin validation
        gem_min_idx = int(np.argmin(geminin_smooth))
        results['Geminin_min_time'] = float(times[gem_min_idx])
        results['Geminin_min_value'] = float(geminin_smooth[gem_min_idx])

        min_gem = geminin_smooth[gem_min_idx]
        for i in range(gem_min_idx + 1, len(geminin_smooth)):
            if geminin_smooth[i] > min_gem + 0.05:
                results['Geminin_rise_time'] = float(times[i])
                results['Geminin_rise_value'] = float(geminin_smooth[i])
                break

        return results

    # S/G2 = first sustained rise of CDT1 above its S-phase minimum + threshold, searched from >= 3 h after G1/S.
    def find_sg2_transition(self, cdt1_signal, times, sensitivity='medium',
                            sustained_rise_duration=0.5, g1s_time=None,
                            min_s_phase_hours=3.0):

        cdt1_smooth = self.smooth_signal(cdt1_signal)
        results = {}
        median_interval = np.median(np.diff(times)) if len(times) > 1 else 0.25
        required_consecutive = max(2, int(sustained_rise_duration / median_interval))

        if g1s_time is None:
            return results

        # Define the earliest point we trust as being in S phase
        search_start_time = g1s_time + min_s_phase_hours
        search_start_idx = int(np.searchsorted(times, search_start_time))
        search_start_idx = min(search_start_idx, len(cdt1_smooth) - 1)

        window_signal = cdt1_smooth[search_start_idx:]
        if len(window_signal) < required_consecutive:
            return results

        # Find the true CDT1 minimum within the search window
        # This is critical: we must not search for a rise before CDT1 
        # has actually bottomed out
        global_min_in_window = int(np.argmin(window_signal))
        rise_search_start_idx = search_start_idx + global_min_in_window

        # Compute sustained_min around this true minimum
        _, _, sustained_min = self.find_sustained_minimum(
            window_signal, times[search_start_idx:], window_size=5)

        results['CDT1_sustained_min'] = sustained_min
        results['CDT1_min_start_time'] = float(times[rise_search_start_idx])

        thresholds = {'low': 0.10, 'medium': 0.07, 'high': 0.05}
        threshold_increase = thresholds.get(sensitivity, 0.07)
        rise_threshold = sustained_min + threshold_increase

        # Search for sustained rise only after the true minimum
        sg2_idx = None
        consecutive_count = 0

        for i in range(rise_search_start_idx, len(cdt1_smooth)):
            if cdt1_smooth[i] > rise_threshold:
                consecutive_count += 1
                if consecutive_count >= required_consecutive:
                    sg2_idx = i - required_consecutive + 1
                    break
            else:
                consecutive_count = 0

        if sg2_idx is not None:
            results['SG2_time'] = float(times[sg2_idx])
            results['SG2_CDT1_value'] = float(cdt1_smooth[sg2_idx])
            results['S_phase_duration'] = None

        return results


# ---------------------------
# Main analyzer: orchestrates modules
# ---------------------------
class ImprovedPIPFUCCIAnalyzer:
    """
    Main user-facing class: loads data, runs QC and transition detection,
    handles exclusions and outliers, and produces plots.
    """

    # Load data, create the output folder, and set up QC and transition detection.
    def __init__(self, csv_file_path, output_dir=None,
                 green_raw_threshold=None, red_raw_threshold=None,
                 use_raw_qc=True):
        # Store parameters
        self.csv_file_path = csv_file_path
        self.use_raw_qc = use_raw_qc
        self.green_raw_threshold = green_raw_threshold
        self.red_raw_threshold = red_raw_threshold

        # Load data and standardize
        loader = SignalLoader(csv_file_path)
        self.data = loader.get_dataframe()

        # Output directory (defaults to the input file's folder)
        if output_dir is None:
            self.output_dir = Path(csv_file_path).parent
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store modules
        self.qc = QCModule(self.data, use_raw_qc=self.use_raw_qc,
                            green_raw_threshold=self.green_raw_threshold,
                            red_raw_threshold=self.red_raw_threshold)
        self.detector = TransitionDetector()

        # Results placeholder
        self.results = None
        self.median_interval_hours = None

        # Temporal resolution check (small sample)
        self._check_temporal_resolution_sample()

        _logger.info(f"Input file: {csv_file_path}")
        _logger.info(f"Output directory: {self.output_dir}")
        _logger.info(f"Loaded data: {len(self.data)} rows, {self.data['Cell_ID'].nunique()} cells")

    # Estimate the imaging interval from the first 5 cells and report it.
    def _check_temporal_resolution_sample(self):
        sample_cells = self.data['Cell_ID'].unique()[:5]
        intervals = []
        for cell_id in sample_cells:
            cd = self.data[self.data['Cell_ID'] == cell_id].sort_values('Time')
            if len(cd) > 1:
                times = np.array(cd['Time'].values, dtype=float)
                intervals.extend(np.diff(times))
        if intervals:
            median_interval = float(np.median(intervals))
            self.median_interval_hours = median_interval
            median_interval_min = median_interval * 60
            _logger.info(f"\n{'='*60}")
            _logger.info(f"Temporal Resolution: ~{median_interval_min:.1f} minutes/frame")
            _logger.info(f"{'='*60}")

    # Convenience wrappers around the QC and detection modules
    # Shortcut to TransitionDetector.smooth_signal.
    def smooth_signal(self, signal, kernel_size=3):
        return self.detector.smooth_signal(signal, kernel_size=kernel_size)

    # Shortcut to TransitionDetector.find_sustained_minimum.
    def find_sustained_minimum(self, signal, times, window_size=5):
        return self.detector.find_sustained_minimum(signal, times, window_size=window_size)

    # Shortcut to TransitionDetector.find_g1s_transition.
    def find_g1s_transition(self, cdt1_signal, geminin_signal, times,
                            min_g1_duration=None, sustained_drop_duration=0.5):
        return self.detector.find_g1s_transition(cdt1_signal, geminin_signal, times,
                                                min_g1_duration=min_g1_duration,
                                                sustained_drop_duration=sustained_drop_duration)

    # Shortcut to TransitionDetector.find_sg2_transition.
    def find_sg2_transition(self, cdt1_signal, times, sensitivity='medium',
                            sustained_rise_duration=0.5, g1s_time=None):
        return self.detector.find_sg2_transition(cdt1_signal, times,
                                                 sensitivity=sensitivity,
                                                 sustained_rise_duration=sustained_rise_duration,
                                                 g1s_time=g1s_time)

    # Shortcut to QCModule.check_raw_signal_quality.
    def check_raw_signal_quality(self, cell_data):
        return self.qc.check_raw_signal_quality(cell_data)

    # Shortcut to QCModule.check_cell_quality.
    def check_cell_quality(self, cell_data, cell_id):
        return self.qc.check_cell_quality(cell_data, cell_id)

    # Run QC plus G1/S and S/G2 detection on one cell; returns one row of results.
    def analyze_cell(self, cell_id, sg2_sensitivity='medium', min_g1_duration=None,
                     sustained_drop_duration=0.5, sustained_rise_duration=0.5,
                     skip_qc=False):
        """
        Analyze one cell.
        Returns a result dict or None if insufficient data.
        """
        cell_data = self.data[self.data['Cell_ID'] == str(cell_id)].copy()
        cell_data = cell_data.sort_values('Time').reset_index(drop=True)
        if len(cell_data) < 5:
            _logger.debug(f"Insufficient frames for cell {cell_id}; skipping.")
            return None

        pass_qc, qc_flags, qc_metrics = self.check_cell_quality(cell_data, str(cell_id))

        times = np.array(cell_data['Time'].values, dtype=float)
        cdt1_signal = np.array(cell_data['CDT1_Signal'].values, dtype=float)
        geminin_signal = np.array(cell_data['Geminin_Signal'].values, dtype=float)

        if pass_qc or skip_qc:
            g1s_results = self.find_g1s_transition(cdt1_signal, geminin_signal, times,
                                                   min_g1_duration=min_g1_duration,
                                                   sustained_drop_duration=sustained_drop_duration)

            # Post-detection validation: reject G1 shorter than 1 h
            if 'G1_duration' in g1s_results and g1s_results['G1_duration'] is not None:
                if g1s_results['G1_duration'] < 1.0:
                    qc_flags.append('UNREALISTIC_G1_DURATION')
                    pass_qc = False
                    g1s_results = {}

            g1s_time = g1s_results.get('G1S_time', None)
            sg2_results = self.find_sg2_transition(cdt1_signal, times, sg2_sensitivity,
                                                   sustained_rise_duration=sustained_rise_duration,
                                                   g1s_time=g1s_time)
        else:
            g1s_results = {}
            sg2_results = {}

        # Build result dict (guarantee keys exist)
        result = {
            'Cell_ID': str(cell_id),
            'Start_time': float(times[0]),
            'End_time': float(times[-1]),
            'Total_duration': float(times[-1] - times[0]),
            'Num_frames': int(len(times)),
            'QC_pass': bool(pass_qc),
            'QC_flags': '; '.join(qc_flags) if qc_flags else 'PASS'
        }
        # merge metrics and detection results
        result.update(qc_metrics if isinstance(qc_metrics, dict) else {})
        result.update(g1s_results if isinstance(g1s_results, dict) else {})
        result.update(sg2_results if isinstance(sg2_results, dict) else {})

        # Compute S_phase_duration and Has_complete_S_phase flag
        if ('G1S_time' in result) and ('SG2_time' in result):
            g1 = result.get('G1S_time')
            sg2 = result.get('SG2_time')
            if pd.notna(g1) and pd.notna(sg2):
                try:
                    result['S_phase_duration'] = float(sg2 - g1)
                    result['Has_complete_S_phase'] = True
                except Exception:
                    result['S_phase_duration'] = None
                    result['Has_complete_S_phase'] = False
            else:
                result['S_phase_duration'] = None
                result['Has_complete_S_phase'] = False
        else:
            result['S_phase_duration'] = None
            result['Has_complete_S_phase'] = False

        return result

    # Analyze every cell, save pip_fucci_analysis_results.csv and cells_failed_qc.csv, and print a summary.
    def analyze_all_cells(self, sg2_sensitivity='medium', min_g1_duration=None,
                          sustained_drop_duration=0.5, sustained_rise_duration=0.5,
                          skip_qc=False, save_failed_qc=True,
                          detect_outliers=True, outlier_iqr_multiplier=1.5):
        """
        Analyze all cells.
        """
        results = []
        cell_ids = list(self.data['Cell_ID'].unique())
        for cell_id in cell_ids:
            res = self.analyze_cell(cell_id, sg2_sensitivity=sg2_sensitivity,
                                    min_g1_duration=min_g1_duration,
                                    sustained_drop_duration=sustained_drop_duration,
                                    sustained_rise_duration=sustained_rise_duration,
                                    skip_qc=skip_qc)
            if res:
                results.append(res)

        self.results = pd.DataFrame(results)

        if detect_outliers:
            try:
                _ = self.detect_phase_outliers(iqr_multiplier=outlier_iqr_multiplier)
            except Exception as e:
                _logger.warning(f"Outlier detection failed: {e}")

        # Save results csv
        output_file = self.output_dir / 'pip_fucci_analysis_results.csv'
        self.results.to_csv(output_file, index=False)
        _logger.info(f"\nAll results saved to: {output_file}")

        # Save failed QC list if requested
        if save_failed_qc:
            qc_failed = self.results[self.results['QC_pass'] == False]
            if len(qc_failed) > 0:
                failed_file = self.output_dir / 'cells_failed_qc.csv'
                cols_to_save = ['Cell_ID', 'QC_flags', 'track_duration', 'n_frames',
                                'cdt1_dynamic_range', 'geminin_dynamic_range']
                if 'green_raw_max' in qc_failed.columns:
                    cols_to_save.extend(['green_raw_max', 'red_raw_max'])
                # keep only existing columns
                cols_to_save = [c for c in cols_to_save if c in qc_failed.columns]
                qc_failed[cols_to_save].to_csv(failed_file, index=False)
                _logger.info(f"Failed QC cells saved to: {failed_file}")

        # Print summary
        self._print_analysis_summary(sg2_sensitivity, sustained_drop_duration,
                                     sustained_rise_duration, detect_outliers,
                                     outlier_iqr_multiplier)

        return self.results

    # Print QC pass/fail counts, failure reasons and transition detection rates.
    def _print_analysis_summary(self, sg2_sensitivity, sustained_drop_duration,
                                sustained_rise_duration, detect_outliers,
                                outlier_iqr_multiplier):
        if self.results is None or len(self.results) == 0:
            _logger.info("No results to summarize.")
            return

        qc_passed = self.results[self.results['QC_pass'] == True]
        qc_failed = self.results[self.results['QC_pass'] == False]

        total = len(self.results)
        passed = len(qc_passed)
        failed = len(qc_failed)

        _logger.info(f"\n{'='*60}")
        _logger.info("PIP-FUCCI ANALYSIS SUMMARY")
        _logger.info(f"Sensitivity: {sg2_sensitivity}")
        _logger.info(f"Sustained durations: G1-S={sustained_drop_duration}h, S-G2={sustained_rise_duration}h")
        if self.qc.use_raw_qc:
            _logger.info(f"Raw signal QC: Enabled (Green≥{self.qc.green_raw_threshold:.0f}, Red≥{self.qc.red_raw_threshold:.0f})")
        if detect_outliers:
            _logger.info(f"Outlier detection: Enabled (IQR × {outlier_iqr_multiplier})")
        _logger.info(f"{'='*60}")
        _logger.info(f"Total cells: {total}")
        if total > 0:
            _logger.info(f"QC PASSED: {passed} ({passed/total*100:.1f}%)")
            _logger.info(f"QC FAILED: {failed} ({failed/total*100:.1f}%)")
        else:
            _logger.info("No cells analyzed.")

        if failed > 0:
            _logger.info("\nQC Failure Breakdown:")
            flag_counts = {}
            for flags_str in qc_failed['QC_flags']:
                for flag in str(flags_str).split('; '):
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1
            for flag, count in sorted(flag_counts.items(), key=lambda x: x[1], reverse=True):
                _logger.info(f"  {flag}: {count} cells")

        if passed > 0:
            complete_s = int(qc_passed['Has_complete_S_phase'].sum())
            _logger.info(f"\nAnalysis on QC-PASSED cells only (n={passed}):")
            _logger.info(f"Cells with complete S phase: {complete_s} ({complete_s/passed*100:.1f}%)")

            g1s_detected = int(qc_passed['G1S_time'].notna().sum()) if 'G1S_time' in qc_passed.columns else 0
            sg2_detected = int(qc_passed['SG2_time'].notna().sum()) if 'SG2_time' in qc_passed.columns else 0
            _logger.info("\nTransition Detection:")
            _logger.info(f"  G1-S: {g1s_detected}/{passed} ({g1s_detected/passed*100:.1f}%)")
            _logger.info(f"  S-G2: {sg2_detected}/{passed} ({sg2_detected/passed*100:.1f}%)")


    # ---------------------------
    # Outlier detection
    # ---------------------------
    # Flag cells with S or G2/M duration outside Q1 - k*IQR .. Q3 + k*IQR and save results_no_outliers.csv.
    def detect_phase_outliers(self, iqr_multiplier=1.5):
        """
        Detect S and G2/M outliers using IQR, mark QC_flags and QC_pass accordingly.
        Returns list of outliers found (tuples).
        """
        if self.results is None or len(self.results) == 0:
            _logger.info("No results to check for outliers.")
            return []

        qc_passed = self.results[self.results['QC_pass'] == True].copy()
        complete_s_phase = qc_passed[qc_passed['Has_complete_S_phase'] == True].copy()
        if len(complete_s_phase) == 0:
            _logger.info("No cells with complete S phase to check for outliers.")
            return []

        outliers_found = []
        outlier_details = []

        _logger.info(f"\n{'='*60}")
        _logger.info(f"OUTLIER DETECTION (IQR multiplier = {iqr_multiplier})")
        _logger.info(f"{'='*60}")

        # S phase
        s_durations = complete_s_phase['S_phase_duration'].dropna()
        s_phase_stats = {}
        if len(s_durations) >= 4:
            q1_s = s_durations.quantile(0.25)
            q3_s = s_durations.quantile(0.75)
            iqr_s = q3_s - q1_s
            lower_bound_s = q1_s - iqr_multiplier * iqr_s
            upper_bound_s = q3_s + iqr_multiplier * iqr_s
            s_phase_stats = {'q1': q1_s, 'q3': q3_s, 'iqr': iqr_s,
                             'lower': lower_bound_s, 'upper': upper_bound_s,
                             'mean': s_durations.mean(), 'median': s_durations.median()}
            _logger.info(f"\nS Phase Duration:")
            _logger.info(f"  Q1 = {q1_s:.2f}h, Q3 = {q3_s:.2f}h, IQR = {iqr_s:.2f}h")
            _logger.info(f"  Valid range: {lower_bound_s:.2f}h - {upper_bound_s:.2f}h")
            s_outliers = complete_s_phase[(complete_s_phase['S_phase_duration'] < lower_bound_s) |
                                          (complete_s_phase['S_phase_duration'] > upper_bound_s)]
            if len(s_outliers) > 0:
                _logger.info(f"  Found {len(s_outliers)} S phase outlier(s):")
                for _, cell in s_outliers.iterrows():
                    cid = cell['Cell_ID']
                    s_dur = float(cell['S_phase_duration'])
                    _logger.info(f"    Cell {cid}: S = {s_dur:.2f}h")
                    outlier_details.append({'Cell_ID': cid, 'Phase': 'S', 'Duration': s_dur,
                                            'Lower_Bound': lower_bound_s, 'Upper_Bound': upper_bound_s,
                                            'Deviation': 'Low' if s_dur < lower_bound_s else 'High'})
                    idx = int(self.results[self.results['Cell_ID'] == cid].index[0])
                    current_flags = str(self.results.at[idx, 'QC_flags'])
                    new_flags = 'OUTLIER_S_PHASE' if current_flags == 'PASS' else current_flags + '; OUTLIER_S_PHASE'
                    self.results.at[idx, 'QC_flags'] = new_flags
                    self.results.at[idx, 'QC_pass'] = False
                    outliers_found.append((cid, 'S', s_dur))
            else:
                _logger.info("  No S phase outliers detected")
        else:
            _logger.info(f"\nS Phase: Too few cells (n={len(s_durations)}) for outlier detection")

        # G2/M phase
        complete_s_phase = complete_s_phase.copy()
        complete_s_phase['G2M_duration'] = complete_s_phase['End_time'] - complete_s_phase['SG2_time']
        g2m_valid = complete_s_phase[complete_s_phase['G2M_duration'] > 0]
        g2m_phase_stats = {}
        if len(g2m_valid) >= 4:
            g2m_durations = g2m_valid['G2M_duration']
            q1_g2m = g2m_durations.quantile(0.25)
            q3_g2m = g2m_durations.quantile(0.75)
            iqr_g2m = q3_g2m - q1_g2m
            lower_bound_g2m = q1_g2m - iqr_multiplier * iqr_g2m
            upper_bound_g2m = q3_g2m + iqr_multiplier * iqr_g2m
            g2m_phase_stats = {'q1': q1_g2m, 'q3': q3_g2m, 'iqr': iqr_g2m,
                               'lower': lower_bound_g2m, 'upper': upper_bound_g2m,
                               'mean': g2m_durations.mean(), 'median': g2m_durations.median()}
            _logger.info(f"\nG2/M Phase Duration:")
            _logger.info(f"  Q1 = {q1_g2m:.2f}h, Q3 = {q3_g2m:.2f}h, IQR = {iqr_g2m:.2f}h")
            _logger.info(f"  Valid range: {lower_bound_g2m:.2f}h - {upper_bound_g2m:.2f}h")
            g2m_outliers = g2m_valid[(g2m_valid['G2M_duration'] < lower_bound_g2m) |
                                     (g2m_valid['G2M_duration'] > upper_bound_g2m)]
            if len(g2m_outliers) > 0:
                _logger.info(f"  Found {len(g2m_outliers)} G2/M phase outlier(s):")
                for _, cell in g2m_outliers.iterrows():
                    cid = cell['Cell_ID']
                    g2m_dur = float(cell['G2M_duration'])
                    _logger.info(f"    Cell {cid}: G2/M = {g2m_dur:.2f}h")
                    outlier_details.append({'Cell_ID': cid, 'Phase': 'G2/M', 'Duration': g2m_dur,
                                            'Lower_Bound': lower_bound_g2m, 'Upper_Bound': upper_bound_g2m,
                                            'Deviation': 'Low' if g2m_dur < lower_bound_g2m else 'High'})
                    idx = int(self.results[self.results['Cell_ID'] == cid].index[0])
                    current_flags = str(self.results.at[idx, 'QC_flags'])
                    if 'OUTLIER_S_PHASE' in current_flags:
                        new_flags = current_flags + '; OUTLIER_G2M_PHASE'
                    elif current_flags == 'PASS':
                        new_flags = 'OUTLIER_G2M_PHASE'
                    else:
                        new_flags = current_flags + '; OUTLIER_G2M_PHASE'
                    self.results.at[idx, 'QC_flags'] = new_flags
                    self.results.at[idx, 'QC_pass'] = False
                    outliers_found.append((cid, 'G2/M', g2m_dur))
            else:
                _logger.info("  No G2/M phase outliers detected")
        else:
            _logger.info(f"\nG2/M Phase: Too few cells (n={len(g2m_valid)}) for outlier detection")

        _logger.info(f"{'='*60}\n")

        # Save QC-passed results (no outliers) to file
        qc_passed_results = self.results[self.results['QC_pass'] == True]
        no_outliers_file = self.output_dir / 'results_no_outliers.csv'
        qc_passed_results.to_csv(no_outliers_file, index=False)
        _logger.info(f"✓ QC-passed cells (no outliers) saved to: {no_outliers_file}")

        # Save detailed summary file
        summary_file = self.output_dir / 'outlier_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PIP-FUCCI OUTLIER DETECTION SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"IQR Multiplier: {iqr_multiplier}\n")
            f.write(f"Input File: {self.csv_file_path}\n\n")

            f.write("="*70 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("="*70 + "\n")
            total_cells = len(self.results)
            qc_passed_count = len(qc_passed_results)
            outlier_count = len(set([x[0] for x in outliers_found]))
            f.write(f"Total cells analyzed: {total_cells}\n")
            f.write(f"Cells passing QC (no outliers): {qc_passed_count}\n")
            f.write(f"Cells flagged as outliers: {outlier_count}\n")
            f.write(f"Outlier rate: {outlier_count/total_cells*100:.1f}%\n\n")

            if s_phase_stats:
                f.write("="*70 + "\n")
                f.write("S PHASE STATISTICS\n")
                f.write("="*70 + "\n")
                f.write(f"Number of cells evaluated: {len(s_durations)}\n")
                f.write(f"Mean: {s_phase_stats['mean']:.2f} h\n")
                f.write(f"Median: {s_phase_stats['median']:.2f} h\n")
                f.write(f"Q1: {s_phase_stats['q1']:.2f} h\n")
                f.write(f"Q3: {s_phase_stats['q3']:.2f} h\n")
                f.write(f"IQR: {s_phase_stats['iqr']:.2f} h\n")
                f.write(f"Valid range: {s_phase_stats['lower']:.2f} - {s_phase_stats['upper']:.2f} h\n\n")

            if g2m_phase_stats:
                f.write("="*70 + "\n")
                f.write("G2/M PHASE STATISTICS\n")
                f.write("="*70 + "\n")
                f.write(f"Number of cells evaluated: {len(g2m_valid)}\n")
                f.write(f"Mean: {g2m_phase_stats['mean']:.2f} h\n")
                f.write(f"Median: {g2m_phase_stats['median']:.2f} h\n")
                f.write(f"Q1: {g2m_phase_stats['q1']:.2f} h\n")
                f.write(f"Q3: {g2m_phase_stats['q3']:.2f} h\n")
                f.write(f"IQR: {g2m_phase_stats['iqr']:.2f} h\n")
                f.write(f"Valid range: {g2m_phase_stats['lower']:.2f} - {g2m_phase_stats['upper']:.2f} h\n\n")

            if outlier_details:
                f.write("="*70 + "\n")
                f.write("DETAILED OUTLIER LIST\n")
                f.write("="*70 + "\n")
                f.write(f"{'Cell_ID':<15} {'Phase':<8} {'Duration':<12} {'Range':<25} {'Type':<8}\n")
                f.write("-"*70 + "\n")
                for detail in outlier_details:
                    range_str = f"{detail['Lower_Bound']:.2f}-{detail['Upper_Bound']:.2f}h"
                    f.write(f"{str(detail['Cell_ID']):<15} "
                            f"{detail['Phase']:<8} "
                            f"{detail['Duration']:<12.2f} "
                            f"{range_str:<25} "
                            f"{detail['Deviation']:<8}\n")
            else:
                f.write("="*70 + "\n")
                f.write("NO OUTLIERS DETECTED\n")
                f.write("="*70 + "\n")
                f.write("All cells with complete S phase are within expected ranges.\n\n")

        _logger.info(f"✓ Outlier summary report saved to: {summary_file}\n")

        if len(outliers_found) > 0:
            _logger.info(f"Total: {len(set([x[0] for x in outliers_found]))} cell(s) flagged as outliers")
            _logger.info(f"\nFor publication/statistics, use: results_no_outliers.csv")
        else:
            _logger.info("No outliers detected - all cells within expected ranges")

        return outliers_found

    # Mark the listed cells as failed (manual exclusion), save updated results and log the exclusion.
    def exclude_cells(self, cell_ids, reason="MANUAL_EXCLUSION", 
                    save_updated=True):
        """
        Manually exclude specific cells from analysis results.
        
        Parameters:
        -----------
        cell_ids : list or str
            Single cell ID or list of cell IDs to exclude
        reason : str, default="MANUAL_EXCLUSION"
            Reason for exclusion (will be added to QC_flags)
        save_updated : bool, default=True
            Whether to save updated results to CSV files
        
        Returns:
        --------
        dict : Summary of exclusion operation
        """
        if self.results is None or len(self.results) == 0:
            _logger.info("No results available. Run analyze_all_cells() first.")
            return None
        
        # Convert single cell_id to list
        if isinstance(cell_ids, (str, int)):
            cell_ids = [str(cell_ids)]
        else:
            cell_ids = [str(cid) for cid in cell_ids]
        
        # Find cells to exclude
        cells_found = []
        cells_not_found = []
        
        for cell_id in cell_ids:
            if cell_id in self.results['Cell_ID'].values:
                cells_found.append(cell_id)
            else:
                cells_not_found.append(cell_id)
        
        if not cells_found:
            _logger.warning(f"None of the specified cells found in results.")
            if cells_not_found:
                _logger.warning(f"Not found: {cells_not_found}")
            return None
        
        # Update QC status for excluded cells
        _logger.info(f"\n{'='*60}")
        _logger.info(f"MANUAL CELL EXCLUSION")
        _logger.info(f"{'='*60}")
        _logger.info(f"Excluding {len(cells_found)} cell(s): {cells_found}")
        _logger.info(f"Reason: {reason}")
        
        for cell_id in cells_found:
            idx = self.results[self.results['Cell_ID'] == cell_id].index[0]
            
            # Update QC_pass to False
            was_passing = self.results.at[idx, 'QC_pass']
            self.results.at[idx, 'QC_pass'] = False
            
            # Add reason to QC_flags
            current_flags = str(self.results.at[idx, 'QC_flags'])
            if current_flags == 'PASS' or current_flags == 'nan':
                new_flags = reason
            else:
                new_flags = current_flags + '; ' + reason
            self.results.at[idx, 'QC_flags'] = new_flags
            
            _logger.info(f"  Cell {cell_id}: QC_pass changed from {was_passing} to False")
        
        if cells_not_found:
            _logger.warning(f"\nCells not found in results: {cells_not_found}")
        
        # Save updated results
        if save_updated:
            # Save main results
            output_file = self.output_dir / 'pip_fucci_analysis_results.csv'
            self.results.to_csv(output_file, index=False)
            _logger.info(f"\n✓ Updated results saved to: {output_file}")
            
            # Save updated QC-passed only file
            qc_passed = self.results[self.results['QC_pass'] == True]
            no_outliers_file = self.output_dir / 'results_no_outliers.csv'
            qc_passed.to_csv(no_outliers_file, index=False)
            _logger.info(f"✓ Updated QC-passed results saved to: {no_outliers_file}")
            
            # Save exclusion log
            exclusion_log = self.output_dir / 'manual_exclusions.txt'
            with open(exclusion_log, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Exclusion Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Reason: {reason}\n")
                f.write(f"Cells excluded: {', '.join(cells_found)}\n")
                f.write(f"{'='*60}\n")
            _logger.info(f"✓ Exclusion logged to: {exclusion_log}")
        
        # Print updated summary
        qc_passed_count = (self.results['QC_pass'] == True).sum()
        total_count = len(self.results)
        
        _logger.info(f"\n{'='*60}")
        _logger.info(f"UPDATED SUMMARY")
        _logger.info(f"{'='*60}")
        _logger.info(f"Total cells: {total_count}")
        _logger.info(f"QC PASSED: {qc_passed_count} ({qc_passed_count/total_count*100:.1f}%)")
        _logger.info(f"QC FAILED: {total_count - qc_passed_count} ({(total_count-qc_passed_count)/total_count*100:.1f}%)")
        _logger.info(f"{'='*60}\n")
        
        return {
            'excluded': cells_found,
            'not_found': cells_not_found,
            'reason': reason,
            'qc_passed_count': qc_passed_count,
            'total_count': total_count
        }


    # Undo a manual exclusion for the listed cells (removes the MANUAL_EXCLUSION flag only).
    def restore_cells(self, cell_ids, save_updated=True):
        """
        Restore manually excluded cells (removes MANUAL_EXCLUSION flag).
        Note: Only removes manual exclusion - other QC flags remain.
        
        Parameters:
        -----------
        cell_ids : list or str
            Single cell ID or list of cell IDs to restore
        save_updated : bool, default=True
            Whether to save updated results to CSV files
        
        Returns:
        --------
        dict : Summary of restoration operation
        """
        if self.results is None or len(self.results) == 0:
            _logger.info("No results available. Run analyze_all_cells() first.")
            return None
        
        # Convert single cell_id to list
        if isinstance(cell_ids, (str, int)):
            cell_ids = [str(cell_ids)]
        else:
            cell_ids = [str(cid) for cid in cell_ids]
        
        _logger.info(f"\n{'='*60}")
        _logger.info(f"RESTORING MANUALLY EXCLUDED CELLS")
        _logger.info(f"{'='*60}")
        
        cells_restored = []
        cells_not_found = []
        
        for cell_id in cell_ids:
            if cell_id not in self.results['Cell_ID'].values:
                cells_not_found.append(cell_id)
                continue
            
            idx = self.results[self.results['Cell_ID'] == cell_id].index[0]
            current_flags = str(self.results.at[idx, 'QC_flags'])
            
            if 'MANUAL_EXCLUSION' not in current_flags:
                _logger.warning(f"  Cell {cell_id}: No manual exclusion flag found")
                continue
            
            # Remove MANUAL_EXCLUSION from flags
            flags_list = [f.strip() for f in current_flags.split(';')]
            flags_list = [f for f in flags_list if 'MANUAL_EXCLUSION' not in f]
            
            if not flags_list:
                new_flags = 'PASS'
                self.results.at[idx, 'QC_pass'] = True
            else:
                new_flags = '; '.join(flags_list)
                # Check if remaining flags are critical
                critical_flags = [
                    'FEW_FRAMES', 'TRACKING_GAP', 'NON_MONOTONIC_TIME',
                    'DUPLICATE_TIMEPOINTS', 'NOT_STARTING_IN_G1', 'HIGH_EARLY_GEMININ',
                    'CDT1_LOW_DYNAMIC_RANGE', 'GEMININ_LOW_DYNAMIC_RANGE',
                    'GREEN_RAW_TOO_LOW', 'RED_RAW_TOO_LOW', 'BOTH_SIGNALS_WEAK_RAW',
                    'OUTLIER_S_PHASE', 'OUTLIER_G2M_PHASE'
                ]
                has_critical = any(flag in critical_flags for flag in flags_list)
                self.results.at[idx, 'QC_pass'] = not has_critical
            
            self.results.at[idx, 'QC_flags'] = new_flags
            cells_restored.append(cell_id)
            _logger.info(f"  Cell {cell_id}: Restored (QC_pass={self.results.at[idx, 'QC_pass']})")
        
        if cells_not_found:
            _logger.warning(f"\nCells not found: {cells_not_found}")
        
        # Save updated results
        if save_updated and cells_restored:
            output_file = self.output_dir / 'pip_fucci_analysis_results.csv'
            self.results.to_csv(output_file, index=False)
            _logger.info(f"\n✓ Updated results saved to: {output_file}")
            
            qc_passed = self.results[self.results['QC_pass'] == True]
            no_outliers_file = self.output_dir / 'results_no_outliers.csv'
            qc_passed.to_csv(no_outliers_file, index=False)
            _logger.info(f"✓ Updated QC-passed results saved to: {no_outliers_file}")
        
        _logger.info(f"{'='*60}\n")
        
        return {
            'restored': cells_restored,
            'not_found': cells_not_found
        }


    # Return and print the cells flagged MANUAL_EXCLUSION.
    def list_excluded_cells(self):
        """
        List all manually excluded cells.
        
        Returns:
        --------
        DataFrame : Subset of results containing manually excluded cells
        """
        if self.results is None or len(self.results) == 0:
            _logger.info("No results available.")
            return None
        
        # Find manually excluded cells
        manual_exclusions = self.results[
            self.results['QC_flags'].str.contains('MANUAL_EXCLUSION', na=False)
        ].copy()
        
        if len(manual_exclusions) == 0:
            _logger.info("No manually excluded cells found.")
            return None
        
        _logger.info(f"\n{'='*60}")
        _logger.info(f"MANUALLY EXCLUDED CELLS (n={len(manual_exclusions)})")
        _logger.info(f"{'='*60}")
        
        for _, row in manual_exclusions.iterrows():
            _logger.info(f"\nCell {row['Cell_ID']}:")
            _logger.info(f"  QC Flags: {row['QC_flags']}")
            if pd.notna(row.get('G1_duration')):
                _logger.info(f"  G1 duration: {row['G1_duration']:.2f}h")
            if pd.notna(row.get('S_phase_duration')):
                _logger.info(f"  S duration: {row['S_phase_duration']:.2f}h")
        
        _logger.info(f"{'='*60}\n")
        
        return manual_exclusions


    # Prompt for cell IDs to exclude (simple alternative to cell_review_jupyter).
    def interactive_cell_exclusion(self):
        """
        Interactive mode for excluding cells. User is prompted to enter cell IDs.
        Useful in Jupyter notebooks.
        """
        if self.results is None or len(self.results) == 0:
            _logger.info("No results available. Run analyze_all_cells() first.")
            return
        
        print("\n" + "="*60)
        print("INTERACTIVE CELL EXCLUSION")
        print("="*60)
        print("Enter cell IDs to exclude (comma-separated) or 'done' to finish:")
        print("Example: 1, 5, 12 or just: 1")
        print("="*60)
        
        excluded_ids = []
        
        while True:
            user_input = input("\nCell IDs to exclude (or 'done'): ").strip()
            
            if user_input.lower() == 'done':
                break
            
            if not user_input:
                continue
            
            # Parse input
            try:
                ids = [s.strip() for s in user_input.split(',')]
                excluded_ids.extend(ids)
                print(f"  Added: {ids}")
            except Exception as e:
                print(f"  Error parsing input: {e}")
        
        if not excluded_ids:
            print("\nNo cells to exclude.")
            return
        
        # Ask for reason
        reason = input("\nReason for exclusion (press Enter for default): ").strip()
        if not reason:
            reason = "MANUAL_EXCLUSION"
        
        # Exclude cells
        result = self.exclude_cells(excluded_ids, reason=reason)
        
        return result



    # ---------------------------
    # Plotting functions
    # ---------------------------
    # Plot one cell's smoothed CDT1 and Geminin traces with the detected transitions.
    def plot_cell(self, cell_id, save_path=None):
        if self.results is None:
            _logger.info("Run analyze_all_cells() first")
            return

        cell_data = self.data[self.data['Cell_ID'] == str(cell_id)].copy()
        cell_data = cell_data.sort_values('Time').reset_index(drop=True)
        if len(cell_data) < 5:
            _logger.info(f"Insufficient data for cell {cell_id}")
            return

        matching_results = self.results[self.results['Cell_ID'] == str(cell_id)]
        if len(matching_results) == 0:
            _logger.info(f"No results found for cell {cell_id}")
            return
        result = matching_results.iloc[0]

        cdt1_smooth = self.smooth_signal(cell_data['CDT1_Signal'].values)
        geminin_smooth = self.smooth_signal(cell_data['Geminin_Signal'].values)
        times = np.array(cell_data['Time'].values, dtype=float)

        fig, ax = plt.subplots(figsize=(8, 4))
        cdt1_color = '#27ae60'
        geminin_color = '#e67e22'

        if pd.notna(result.get('G1S_time')) and pd.notna(result.get('SG2_time')):
            ax.axvspan(times[0], result['G1S_time'], alpha=0.08, color='green', label='G1', zorder=0)
            ax.axvspan(result['G1S_time'], result['SG2_time'], alpha=0.08, color='blue', label='S', zorder=0)
            ax.axvspan(result['SG2_time'], times[-1], alpha=0.08, color='gold', label='G2', zorder=0)

        ax.plot(times, cdt1_smooth, color=cdt1_color, linewidth=2, label='CDT1', zorder=3, alpha=0.9)
        ax.plot(times, geminin_smooth, color=geminin_color, linewidth=2, label='Geminin', zorder=3, alpha=0.9)

        if pd.notna(result.get('G1S_time')):
            ax.axvline(result['G1S_time'], color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
        if pd.notna(result.get('SG2_time')):
            ax.axvline(result['SG2_time'], color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

        ax.set_xlabel('Time (h)', fontsize=10)
        ax.set_ylabel('Signal Intensity (a.u.)', fontsize=10)

        title = f'Cell {cell_id}'
        if result.get('Has_complete_S_phase'):
            title += f' | G1: {result.get("G1_duration", 0):.1f}h, S: {result.get("S_phase_duration", 0):.1f}h'
        ax.set_title(title, fontsize=11, pad=10)

        ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
        sns.despine()
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f'cell_{cell_id}_plot.svg'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        _logger.info(f"Saved: {save_path}")
        plt.show()


    
    # Plot one cell's normalized traces next to its raw intensities.
    def plot_cell_dual(self, cell_id, save_path=None):
        """
        Plot both normalized and raw signals side-by-side for QC/inspection.
        
        Left panel: Normalized (0-1) - for phase transition detection
        Right panel: Raw intensities - for signal quality assessment
        """
        if self.results is None:
            _logger.info("Run analyze_all_cells() first")
            return

        cell_data = self.data[self.data['Cell_ID'] == str(cell_id)].copy()
        cell_data = cell_data.sort_values('Time').reset_index(drop=True)
        
        if len(cell_data) < 5:
            _logger.info(f"Insufficient data for cell {cell_id}")
            return

        # Check if raw columns exist
        has_raw = 'Green_Raw' in cell_data.columns and 'Red_Raw' in cell_data.columns
        
        if not has_raw:
            _logger.info(f"⚠ No raw data columns found. Plotting normalized only.")
            self.plot_cell(cell_id, save_path)
            return

        matching_results = self.results[self.results['Cell_ID'] == str(cell_id)]
        if len(matching_results) == 0:
            _logger.info(f"No results found for cell {cell_id}")
            return
        result = matching_results.iloc[0]

        times = np.array(cell_data['Time'].values, dtype=float)
        
        # Normalized signals
        cdt1_norm = np.array(cell_data['CDT1_Signal'].values, dtype=float)
        geminin_norm = np.array(cell_data['Geminin_Signal'].values, dtype=float)
        cdt1_smooth_norm = self.smooth_signal(cdt1_norm)
        geminin_smooth_norm = self.smooth_signal(geminin_norm)
        
        # Raw signals
        cdt1_raw = np.array(cell_data['Green_Raw'].values, dtype=float)
        geminin_raw = np.array(cell_data['Red_Raw'].values, dtype=float)
        cdt1_smooth_raw = self.smooth_signal(cdt1_raw)
        geminin_smooth_raw = self.smooth_signal(geminin_raw)

        # Colors
        cdt1_color = '#27ae60'
        geminin_color = '#e67e22'

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

        # --- LEFT PANEL: NORMALIZED (0-1) ---
        if pd.notna(result.get('G1S_time')) and pd.notna(result.get('SG2_time')):
            ax1.axvspan(times[0], result['G1S_time'], alpha=0.08, color='green', 
                        label='G1', zorder=0)
            ax1.axvspan(result['G1S_time'], result['SG2_time'], alpha=0.08, 
                        color='blue', label='S', zorder=0)
            ax1.axvspan(result['SG2_time'], times[-1], alpha=0.08, color='gold', 
                        label='G2', zorder=0)

        ax1.plot(times, cdt1_smooth_norm, color=cdt1_color, linewidth=2.5, 
                 label='CDT1 (normalized)', zorder=3, alpha=0.9)
        ax1.plot(times, geminin_smooth_norm, color=geminin_color, linewidth=2.5, 
                 label='Geminin (normalized)', zorder=3, alpha=0.9)

        if pd.notna(result.get('G1S_time')):
            ax1.axvline(result['G1S_time'], color='#3498db', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2, label='G1-S')
        if pd.notna(result.get('SG2_time')):
            ax1.axvline(result['SG2_time'], color='#e74c3c', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2, label='S-G2')
        
        ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, 
                    label='50% threshold', zorder=1)

        ax1.set_xlabel('Time (h)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Normalized Signal (0-1)', fontsize=11, fontweight='bold')
        ax1.set_title('Normalized Signals (for phase detection)', 
                      fontsize=12, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', frameon=True, framealpha=0.9, 
                   edgecolor='gray', fontsize=9)
        ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax1.set_ylim(-0.05, 1.05)

        # --- RIGHT PANEL: RAW INTENSITIES ---
        ax2.plot(times, cdt1_smooth_raw, color=cdt1_color, linewidth=2.5, 
                 label='CDT1 (raw)', zorder=3, alpha=0.9)
        ax2.plot(times, geminin_smooth_raw, color=geminin_color, linewidth=2.5, 
                 label='Geminin (raw)', zorder=3, alpha=0.9)

        if pd.notna(result.get('G1S_time')):
            ax2.axvline(result['G1S_time'], color='#3498db', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2)
        if pd.notna(result.get('SG2_time')):
            ax2.axvline(result['SG2_time'], color='#e74c3c', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2)
        
        # Add QC threshold lines if they exist
        if hasattr(self.qc, 'green_raw_threshold') and self.qc.green_raw_threshold:
            ax2.axhline(self.qc.green_raw_threshold, color=cdt1_color, 
                        linestyle=':', linewidth=1.5, alpha=0.6, 
                        label=f'CDT1 QC threshold ({self.qc.green_raw_threshold:.0f})', 
                        zorder=1)
        if hasattr(self.qc, 'red_raw_threshold') and self.qc.red_raw_threshold:
            ax2.axhline(self.qc.red_raw_threshold, color=geminin_color, 
                        linestyle=':', linewidth=1.5, alpha=0.6, 
                        label=f'Geminin QC threshold ({self.qc.red_raw_threshold:.0f})', 
                        zorder=1)

        ax2.set_xlabel('Time (h)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Raw Intensity (a.u.)', fontsize=11, fontweight='bold')
        ax2.set_title('Raw Signals (for quality assessment)', 
                      fontsize=12, fontweight='bold', pad=10)
        ax2.legend(loc='upper right', frameon=True, framealpha=0.9, 
                   edgecolor='gray', fontsize=9)
        ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax2.set_ylim(bottom=0)

        # Overall title
        title = f'Cell {cell_id}'
        if result.get('Has_complete_S_phase'):
            title += f' | G1: {result.get("G1_duration", 0):.1f}h, S: {result.get("S_phase_duration", 0):.1f}h'
        
        qc_status = "✓ PASS" if result.get('QC_pass') else "✗ FAIL"
        title += f' | QC: {qc_status}'
        
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.00)
        
        # Add signal strength annotation
        cdt1_max_raw = cdt1_raw.max()
        cdt1_min_raw = cdt1_raw.min()
        gem_max_raw = geminin_raw.max()
        
        info_text = (f'Signal Strength:\n'
                     f'CDT1: {cdt1_max_raw:.0f} → {cdt1_min_raw:.0f} '
                     f'({cdt1_min_raw/cdt1_max_raw*100:.1f}% retained)\n'
                     f'Geminin: {gem_max_raw:.0f} max')
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='gray', linewidth=1)
        ax2.text(0.98, 0.02, info_text, transform=ax2.transAxes, 
                 fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                 bbox=props, family='monospace')

        sns.despine()
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f'cell_{cell_id}_dual_plot.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        _logger.info(f"✓ Saved dual plot: {save_path}")
        plt.show()



    # As plot_cell_dual, but with separate y-axes for raw CDT1 and Geminin.
    def plot_cell_dual_separate_axes(self, cell_id, save_path=None):
        """
        Plot normalized and raw signals with dual Y-axes for raw panel.
        Left: Normalized (shared Y-axis)
        Right: Raw intensities (separate Y-axes for CDT1 and Geminin)
        """
        if self.results is None:
            _logger.info("Run analyze_all_cells() first")
            return
    
        cell_data = self.data[self.data['Cell_ID'] == str(cell_id)].copy()
        cell_data = cell_data.sort_values('Time').reset_index(drop=True)
        
        if len(cell_data) < 5:
            _logger.info(f"Insufficient data for cell {cell_id}")
            return
    
        has_raw = 'Green_Raw' in cell_data.columns and 'Red_Raw' in cell_data.columns
        
        if not has_raw:
            _logger.info(f"⚠ No raw data columns found. Plotting normalized only.")
            self.plot_cell(cell_id, save_path)
            return
    
        matching_results = self.results[self.results['Cell_ID'] == str(cell_id)]
        if len(matching_results) == 0:
            _logger.info(f"No results found for cell {cell_id}")
            return
        result = matching_results.iloc[0]
    
        times = np.array(cell_data['Time'].values, dtype=float)
        
        # Normalized signals
        cdt1_norm = np.array(cell_data['CDT1_Signal'].values, dtype=float)
        geminin_norm = np.array(cell_data['Geminin_Signal'].values, dtype=float)
        cdt1_smooth_norm = self.smooth_signal(cdt1_norm)
        geminin_smooth_norm = self.smooth_signal(geminin_norm)
        
        # Raw signals
        cdt1_raw = np.array(cell_data['Green_Raw'].values, dtype=float)
        geminin_raw = np.array(cell_data['Red_Raw'].values, dtype=float)
        cdt1_smooth_raw = self.smooth_signal(cdt1_raw)
        geminin_smooth_raw = self.smooth_signal(geminin_raw)
    
        # Colors
        cdt1_color = '#27ae60'
        geminin_color = '#e67e22'
    
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    
        # --- LEFT PANEL: NORMALIZED (unchanged) ---
        if pd.notna(result.get('G1S_time')) and pd.notna(result.get('SG2_time')):
            ax1.axvspan(times[0], result['G1S_time'], alpha=0.08, color='green', 
                        label='G1', zorder=0)
            ax1.axvspan(result['G1S_time'], result['SG2_time'], alpha=0.08, 
                        color='blue', label='S', zorder=0)
            ax1.axvspan(result['SG2_time'], times[-1], alpha=0.08, color='gold', 
                        label='G2', zorder=0)
    
        ax1.plot(times, cdt1_smooth_norm, color=cdt1_color, linewidth=2.5, 
                 label='CDT1 (normalized)', zorder=3, alpha=0.9)
        ax1.plot(times, geminin_smooth_norm, color=geminin_color, linewidth=2.5, 
                 label='Geminin (normalized)', zorder=3, alpha=0.9)
    
        if pd.notna(result.get('G1S_time')):
            ax1.axvline(result['G1S_time'], color='#3498db', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2, label='G1-S')
        if pd.notna(result.get('SG2_time')):
            ax1.axvline(result['SG2_time'], color='#e74c3c', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2, label='S-G2')
        
        ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, 
                    label='50% threshold', zorder=1)
    
        ax1.set_xlabel('Time (h)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Normalized Signal (0-1)', fontsize=11, fontweight='bold')
        ax1.set_title('Normalized Signals (for phase detection)', 
                      fontsize=12, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', frameon=True, framealpha=0.9, 
                   edgecolor='gray', fontsize=9)
        ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax1.set_ylim(-0.05, 1.05)
    
        # --- RIGHT PANEL: RAW WITH DUAL Y-AXES ---
        # Primary axis for CDT1
        ax2.plot(times, cdt1_smooth_raw, color=cdt1_color, linewidth=2.5, 
                 label='CDT1 (raw)', zorder=3, alpha=0.9)
        ax2.set_ylabel('CDT1 Raw Intensity (a.u.)', fontsize=11, fontweight='bold', 
                       color=cdt1_color)
        ax2.tick_params(axis='y', labelcolor=cdt1_color)
        ax2.set_ylim(bottom=0)
        
        # Secondary axis for Geminin
        ax2_twin = ax2.twinx()
        ax2_twin.plot(times, geminin_smooth_raw, color=geminin_color, linewidth=2.5, 
                      label='Geminin (raw)', zorder=3, alpha=0.9)
        ax2_twin.set_ylabel('Geminin Raw Intensity (a.u.)', fontsize=11, fontweight='bold',
                            color=geminin_color)
        ax2_twin.tick_params(axis='y', labelcolor=geminin_color)
        ax2_twin.set_ylim(bottom=0)
    
        # Phase transition lines
        if pd.notna(result.get('G1S_time')):
            ax2.axvline(result['G1S_time'], color='#3498db', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2)
        if pd.notna(result.get('SG2_time')):
            ax2.axvline(result['SG2_time'], color='#e74c3c', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2)
        
        # QC thresholds on appropriate axes
        if hasattr(self.qc, 'green_raw_threshold') and self.qc.green_raw_threshold:
            ax2.axhline(self.qc.green_raw_threshold, color=cdt1_color, 
                        linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
        if hasattr(self.qc, 'red_raw_threshold') and self.qc.red_raw_threshold:
            ax2_twin.axhline(self.qc.red_raw_threshold, color=geminin_color, 
                             linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
    
        ax2.set_xlabel('Time (h)', fontsize=11, fontweight='bold')
        ax2.set_title('Raw Signals (independent scales)', 
                      fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, 
                   loc='upper right', frameon=True, framealpha=0.9, 
                   edgecolor='gray', fontsize=9)
    
        # Overall title
        title = f'Cell {cell_id}'
        if result.get('Has_complete_S_phase'):
            title += f' | G1: {result.get("G1_duration", 0):.1f}h, S: {result.get("S_phase_duration", 0):.1f}h'
        
        qc_status = "✓ PASS" if result.get('QC_pass') else "✗ FAIL"
        title += f' | QC: {qc_status}'
        
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.00)
        
        # Signal strength annotation
        cdt1_max_raw = cdt1_raw.max()
        cdt1_min_raw = cdt1_raw.min()
        gem_max_raw = geminin_raw.max()
        
        info_text = (f'CDT1: {cdt1_max_raw:.0f} → {cdt1_min_raw:.0f}\n'
                     f'Geminin: {gem_max_raw:.0f} max')
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='gray', linewidth=1)
        ax2.text(0.98, 0.02, info_text, transform=ax2.transAxes, 
                 fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                 bbox=props, family='monospace')
    
        sns.despine(ax=ax1)
        sns.despine(ax=ax2, right=False)  # Keep right spine for twin axis
        plt.tight_layout()
    
        if save_path is None:
            save_path = self.output_dir / f'cell_{cell_id}_dual_plot.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        _logger.info(f"✓ Saved dual-axis plot: {save_path}")
        plt.show()
    
    














    
    # Overlay the traces of all QC-passed cells (split over several figures if many).
    def plot_all_qc_passed_cells(self, save_path=None, max_cells_per_plot=20, alpha=0.3):
        """
        Plot all QC-passed cells on the same graph to visualize population behavior.
        Creates multiple plots if there are many cells to avoid overcrowding.
        
        Parameters:
        -----------
        save_path : str or Path, optional
            Base path for saving plots. Will append plot number if multiple plots needed.
        max_cells_per_plot : int, default=20
            Maximum number of cells to show per plot to avoid overcrowding
        alpha : float, default=0.3
            Transparency of individual cell traces
        """
        if self.results is None or len(self.results) == 0:
            _logger.info("No results. Run analyze_all_cells() first.")
            return
        
        # Get QC-passed cells only
        qc_passed = self.results[self.results['QC_pass'] == True].copy()
        
        if len(qc_passed) == 0:
            _logger.info("No cells passed QC.")
            return
        
        _logger.info(f"\n{'='*60}")
        _logger.info(f"Plotting {len(qc_passed)} QC-passed cells")
        _logger.info(f"{'='*60}")
        
        # Determine number of plots needed
        n_plots = int(np.ceil(len(qc_passed) / max_cells_per_plot))
        
        for plot_idx in range(n_plots):
            start_idx = plot_idx * max_cells_per_plot
            end_idx = min((plot_idx + 1) * max_cells_per_plot, len(qc_passed))
            cells_subset = qc_passed.iloc[start_idx:end_idx]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            cdt1_color = '#27ae60'
            geminin_color = '#e67e22'
            
            # Store transition times for reference lines
            g1s_times = []
            sg2_times = []
            
            # Plot each cell
            for _, cell_result in cells_subset.iterrows():
                cell_id = cell_result['Cell_ID']
                cell_data = self.data[self.data['Cell_ID'] == str(cell_id)].copy()
                cell_data = cell_data.sort_values('Time').reset_index(drop=True)
                
                if len(cell_data) < 5:
                    continue
                
                times = np.array(cell_data['Time'].values, dtype=float)
                cdt1_signal = np.array(cell_data['CDT1_Signal'].values, dtype=float)
                geminin_signal = np.array(cell_data['Geminin_Signal'].values, dtype=float)
                
                # Smooth signals
                cdt1_smooth = self.smooth_signal(cdt1_signal)
                geminin_smooth = self.smooth_signal(geminin_signal)
                
                # Plot CDT1
                ax1.plot(times, cdt1_smooth, color=cdt1_color, linewidth=1.5, 
                        alpha=alpha, zorder=2)
                
                # Plot Geminin
                ax2.plot(times, geminin_smooth, color=geminin_color, linewidth=1.5, 
                        alpha=alpha, zorder=2)
                
                # Mark transitions if detected
                if pd.notna(cell_result.get('G1S_time')):
                    g1s_time = cell_result['G1S_time']
                    g1s_times.append(g1s_time)
                    ax1.scatter([g1s_time], [cell_result.get('G1S_CDT1_value', 0.5)], 
                            color='#3498db', s=30, alpha=0.6, zorder=3, marker='v')
                    ax2.scatter([g1s_time], [cell_result.get('G1S_Geminin_value', 0.5)], 
                            color='#3498db', s=30, alpha=0.6, zorder=3, marker='v')
                
                if pd.notna(cell_result.get('SG2_time')):
                    sg2_time = cell_result['SG2_time']
                    sg2_times.append(sg2_time)
                    ax1.scatter([sg2_time], [cell_result.get('SG2_CDT1_value', 0.5)], 
                            color='#e74c3c', s=30, alpha=0.6, zorder=3, marker='^')
            
            # Add reference lines for average transition times
            if g1s_times:
                mean_g1s = np.mean(g1s_times)
                ax1.axvline(mean_g1s, color='#3498db', linestyle='--', linewidth=2, 
                        alpha=0.7, label=f'Mean G1-S: {mean_g1s:.1f}h', zorder=1)
                ax2.axvline(mean_g1s, color='#3498db', linestyle='--', linewidth=2, 
                        alpha=0.7, zorder=1)
            
            if sg2_times:
                mean_sg2 = np.mean(sg2_times)
                ax1.axvline(mean_sg2, color='#e74c3c', linestyle='--', linewidth=2, 
                        alpha=0.7, label=f'Mean S-G2: {mean_sg2:.1f}h', zorder=1)
                ax2.axvline(mean_sg2, color='#e74c3c', linestyle='--', linewidth=2, 
                        alpha=0.7, zorder=1)
            
            # Add threshold lines
            ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, 
                    label='CDT1 50% threshold', zorder=1)
            
            # Formatting
            ax1.set_ylabel('CDT1 Signal Intensity (a.u.)', fontsize=12, fontweight='bold')
            ax1.set_ylim(-0.05, 1.05)
            ax1.legend(loc='upper right', frameon=True, framealpha=0.9, 
                    edgecolor='gray', fontsize=10)
            ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax1.set_title(f'All QC-Passed Cells - CDT1 Signal (Cells {start_idx+1}-{end_idx} of {len(qc_passed)})', 
                        fontsize=13, fontweight='bold', pad=10)
            
            ax2.set_ylabel('Geminin Signal Intensity (a.u.)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax2.set_title(f'All QC-Passed Cells - Geminin Signal (Cells {start_idx+1}-{end_idx} of {len(qc_passed)})', 
                        fontsize=13, fontweight='bold', pad=10)
            
            # Add text box with statistics
            stats_text = f"n = {len(cells_subset)} cells\n"
            if g1s_times:
                stats_text += f"G1-S: {np.mean(g1s_times):.1f}±{np.std(g1s_times):.1f}h\n"
            if sg2_times:
                stats_text += f"S-G2: {np.mean(sg2_times):.1f}±{np.std(sg2_times):.1f}h"
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.95, 
                        edgecolor='gray', linewidth=1)
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props)
            
            sns.despine()
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                if n_plots > 1:
                    plot_path = self.output_dir / f'all_qc_passed_cells_plot{plot_idx+1}.svg'
                else:
                    plot_path = self.output_dir / 'all_qc_passed_cells.svg'
            else:
                if n_plots > 1:
                    base = Path(save_path).stem
                    ext = Path(save_path).suffix
                    plot_path = Path(save_path).parent / f'{base}_plot{plot_idx+1}{ext}'
                else:
                    plot_path = save_path
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            _logger.info(f"✓ Saved plot {plot_idx+1}/{n_plots}: {plot_path}")
            plt.show()
        
        # Summary statistics
        _logger.info(f"\n{'='*60}")
        _logger.info("POPULATION SUMMARY")
        _logger.info(f"{'='*60}")
        
        if g1s_times:
            _logger.info(f"\nG1-S Transition (n={len(g1s_times)}):")
            _logger.info(f"  Mean ± SD: {np.mean(g1s_times):.2f} ± {np.std(g1s_times):.2f} h")
            _logger.info(f"  Median [IQR]: {np.median(g1s_times):.2f} "
                        f"[{np.percentile(g1s_times, 25):.2f}-{np.percentile(g1s_times, 75):.2f}] h")
        
        if sg2_times:
            _logger.info(f"\nS-G2 Transition (n={len(sg2_times)}):")
            _logger.info(f"  Mean ± SD: {np.mean(sg2_times):.2f} ± {np.std(sg2_times):.2f} h")
            _logger.info(f"  Median [IQR]: {np.median(sg2_times):.2f} "
                        f"[{np.percentile(sg2_times, 25):.2f}-{np.percentile(sg2_times, 75):.2f}] h")
        
        _logger.info(f"{'='*60}\n")


    
    # Four-panel population figure: G1 and S histograms, transition times, G1 vs S.
    def plot_population_summary(self, save_path=None, qc_passed_only=True):
        if self.results is None or len(self.results) == 0:
            _logger.info("No results. Run analyze_all_cells() first.")
            return

        if qc_passed_only and 'QC_pass' in self.results.columns:
            plot_data = self.results[self.results['QC_pass'] == True].copy()
            title_suffix = " (QC-passed cells)"
        else:
            plot_data = self.results.copy()
            title_suffix = " (all cells)"

        complete_cells = plot_data[plot_data['Has_complete_S_phase'] == True].copy()
        if len(complete_cells) == 0:
            _logger.info("No cells with complete S phase measurements.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        # G1 duration
        ax = axes[0, 0]
        g1_data = plot_data[plot_data['G1_duration'].notna()]
        if len(g1_data) > 0:
            sns.histplot(data=g1_data, x='G1_duration', kde=True, ax=ax, color=colors[0], bins=15, alpha=0.6, edgecolor='white')
            ax.axvline(g1_data['G1_duration'].mean(), color='#e74c3c', linestyle='--', linewidth=2,
                       label=f"Mean: {g1_data['G1_duration'].mean():.1f}h")
            ax.set_xlabel('G1 Duration (h)', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'G1 Phase{title_suffix}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)

        # S phase
        ax = axes[0, 1]
        sns.histplot(data=complete_cells, x='S_phase_duration', kde=True, ax=ax, color=colors[1], bins=15, alpha=0.6, edgecolor='white')
        mean_s = complete_cells['S_phase_duration'].mean()
        ax.axvline(mean_s, color='#34495e', linestyle='--', linewidth=2, label=f'Mean: {mean_s:.1f}h')
        ax.set_xlabel('S Duration (h)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'S Phase{title_suffix}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)

        # Transitions scatter
        ax = axes[1, 0]
        ax.scatter(complete_cells['G1S_time'], complete_cells['SG2_time'], alpha=0.5, s=50, color=colors[2], edgecolors='white', linewidth=0.5)
        ax.set_xlabel('G1-S Time (h)', fontsize=10)
        ax.set_ylabel('S-G2 Time (h)', fontsize=10)
        ax.set_title(f'Transition Timing{title_suffix}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)

        # G1 vs S
        ax = axes[1, 1]
        if 'G1_duration' in complete_cells.columns:
            valid = complete_cells[complete_cells['G1_duration'].notna()]
            if len(valid) > 0:
                ax.scatter(valid['G1_duration'], valid['S_phase_duration'], alpha=0.5, s=50, color=colors[3], edgecolors='white', linewidth=0.5)
                ax.set_xlabel('G1 Duration (h)', fontsize=10)
                ax.set_ylabel('S Duration (h)', fontsize=10)
                ax.set_title(f'G1 vs S Phase{title_suffix}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.2)

        for a in axes.flat:
            sns.despine(ax=a)
        plt.tight_layout()

        if save_path is None:
            suffix = "_qc_passed" if qc_passed_only else "_all_cells"
            save_path = self.output_dir / f'population_summary{suffix}.svg'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        _logger.info(f"Saved: {save_path}")
        plt.show()


    # Violin plot of G1, S and G2/M durations, with summary statistics printed.
    def plot_phase_durations_violin(self, save_path=None, qc_passed_only=True, show_points=True):
        if self.results is None or len(self.results) == 0:
            _logger.info("No results. Run analyze_all_cells() first.")
            return

        if qc_passed_only and 'QC_pass' in self.results.columns:
            plot_data = self.results[self.results['QC_pass'] == True].copy()
            title_suffix = " (QC-passed cells)"
        else:
            plot_data = self.results.copy()
            title_suffix = ""

        complete_cells = plot_data[plot_data['Has_complete_S_phase'] == True].copy()
        if len(complete_cells) == 0:
            _logger.info("No cells with complete S phase for plotting.")
            return

        data_list = []
        # G1
        g1_data = complete_cells[complete_cells['G1_duration'].notna()]['G1_duration']
        for val in g1_data:
            data_list.append({'Phase': 'G1', 'Duration (h)': float(val)})
        # S
        s_data = complete_cells[complete_cells['S_phase_duration'].notna()]['S_phase_duration']
        for val in s_data:
            data_list.append({'Phase': 'S', 'Duration (h)': float(val)})
        # G2/M
        complete_cells['G2M_duration'] = complete_cells['End_time'] - complete_cells['SG2_time']
        g2_valid = complete_cells[complete_cells['G2M_duration'] > 0]
        for val in g2_valid['G2M_duration']:
            data_list.append({'Phase': 'G2/M', 'Duration (h)': float(val)})

        if not data_list:
            _logger.info("No phase duration data available for plotting.")
            return

        df_plot = pd.DataFrame(data_list)
        phase_list = [p for p in ['G1', 'S', 'G2/M'] if p in df_plot['Phase'].unique()]

        if not phase_list:
            _logger.info("No phase data to plot.")
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        colors = {'G1': '#27ae60', 'S': '#e74c3c', 'G2/M': '#f39c12'}
        positions = np.arange(len(phase_list)) * 0.5
        parts = ax.violinplot([df_plot[df_plot['Phase'] == phase]['Duration (h)'].values for phase in phase_list],
                              positions=positions, widths=0.4, showmeans=False, showmedians=False, showextrema=False)

        for i, (pc, phase) in enumerate(zip(parts['bodies'], phase_list)):
            pc.set_facecolor(colors[phase])
            pc.set_alpha(0.75)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.2)

        for i, phase in enumerate(phase_list):
            phase_data = df_plot[df_plot['Phase'] == phase]['Duration (h)'].values
            q1, median, q3 = np.percentile(phase_data, [25, 50, 75])
            mean = np.mean(phase_data)
            iqr = q3 - q1
            whisker_low = np.min(phase_data[phase_data >= q1 - 1.5 * iqr])
            whisker_high = np.max(phase_data[phase_data <= q3 + 1.5 * iqr])
            pos = positions[i]
            ax.plot([pos, pos], [whisker_low, whisker_high], color='black', linewidth=1.2, alpha=0.8, zorder=3)
            cap_width = 0.06
            ax.plot([pos - cap_width, pos + cap_width], [whisker_low, whisker_low], color='black', linewidth=1.2, alpha=0.8, zorder=3)
            ax.plot([pos - cap_width, pos + cap_width], [whisker_high, whisker_high], color='black', linewidth=1.2, alpha=0.8, zorder=3)
            box_width = 0.10
            ax.plot([pos - box_width / 2, pos + box_width / 2], [q1, q1], color='black', linewidth=1.8, alpha=0.9, zorder=3)
            ax.plot([pos - box_width / 2, pos + box_width / 2], [q3, q3], color='black', linewidth=1.8, alpha=0.9, zorder=3)
            ax.plot([pos - box_width / 2, pos - box_width / 2], [q1, q3], color='black', linewidth=1.8, alpha=0.9, zorder=3)
            ax.plot([pos + box_width / 2, pos + box_width / 2], [q1, q3], color='black', linewidth=1.8, alpha=0.9, zorder=3)
            ax.plot([pos - box_width / 2, pos + box_width / 2], [median, median], color='white', linewidth=2.5, alpha=1.0, zorder=4)
            ax.scatter([pos], [mean], color='white', s=70, marker='D', edgecolors='black', linewidth=1.3, zorder=5, label='Mean' if i == 0 else '')

        if show_points:
            for i, phase in enumerate(phase_list):
                phase_data = df_plot[df_plot['Phase'] == phase]['Duration (h)'].values
                np.random.seed(42)
                jitter = np.random.normal(0, 0.025, size=len(phase_data))
                x_coords = np.full(len(phase_data), positions[i]) + jitter
                ax.scatter(x_coords, phase_data, color=colors[phase], s=15, alpha=0.35, edgecolors='black', linewidth=0.4, zorder=2)

        ax.set_xticks(positions)
        new_labels = []
        for phase in phase_list:
            n = len(df_plot[df_plot['Phase'] == phase])
            new_labels.append(f'{phase}\n(n={n})')
        ax.set_xticklabels(new_labels, fontsize=11)
        ax.set_ylabel('Duration (hours)', fontsize=12, fontweight='bold')
        ax.set_title(f'Cell Cycle Phase Durations{title_suffix}', fontsize=13, fontweight='bold', pad=15)

        # Compact stats box
        stats_text = []
        for phase in phase_list:
            phase_data = df_plot[df_plot['Phase'] == phase]['Duration (h)'].values
            stats_text.append(f'{phase}: {np.mean(phase_data):.1f}±{np.std(phase_data):.1f}h (med: {np.median(phase_data):.1f}h)')
        textstr = '\n'.join(stats_text)
        props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=0.8)
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=8.5, verticalalignment='top', horizontalalignment='right', bbox=props)

        ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='gray', fontsize=9, handletextpad=0.5)
        ax.grid(False)
        ax.set_ylim(bottom=0)
        ax.set_xlim(positions[0] - 0.45, positions[-1] + 0.45)
        sns.despine()
        ax.tick_params(axis='both', which='major', labelsize=11, length=4, width=1)
        plt.tight_layout()

        if save_path is None:
            suffix = "_qc_passed" if qc_passed_only else "_all_cells"
            save_path = self.output_dir / f'phase_durations_violin{suffix}.svg'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        _logger.info(f"Saved: {save_path}")
        plt.show()

        # Print stats in console
        _logger.info("\n" + "="*60)
        _logger.info("PHASE DURATION STATISTICS (Complete S-phase cells only)")
        _logger.info("="*60)
        for phase in phase_list:
            phase_data = df_plot[df_plot['Phase'] == phase]['Duration (h)'].values
            _logger.info(f"\n{phase} Phase (n={len(phase_data)}):")
            _logger.info(f"  Mean ± SD: {np.mean(phase_data):.2f} ± {np.std(phase_data):.2f} h")
            _logger.info(f"  Median [IQR]: {np.median(phase_data):.2f} [{np.percentile(phase_data,25):.2f}-{np.percentile(phase_data,75):.2f}] h")
            _logger.info(f"  Range: {np.min(phase_data):.2f} - {np.max(phase_data):.2f} h")
        _logger.info("="*60)

    # Grid PDF with one small trace plot per QC-passed cell.
    def plot_qc_passed_cells_grid(self, save_path=None, max_cells=None, 
                                ncols=4, figsize_per_cell=(4, 3), 
                                show_transitions=True, 
                                complete_s_phase_only=True):
        """
        Plot QC-passed cells in a grid layout with one subplot per cell.
        Clean style similar to traditional cell cycle plots.
        
        Parameters:
        -----------
        save_path : str or Path, optional
            Path for saving the plot
        max_cells : int, optional
            Maximum number of cells to plot. If None, plots all QC-passed cells
        ncols : int, default=4
            Number of columns in the grid
        figsize_per_cell : tuple, default=(4, 3)
            Size of each individual subplot (width, height) in inches
        show_transitions : bool, default=True
            Whether to mark G1-S and S-G2 transitions with vertical lines
        complete_s_phase_only : bool, default=True
            If True, only plot cells with both G1-S and S-G2 transitions detected
            If False, plot all QC-passed cells regardless of transition detection
        """
        if self.results is None or len(self.results) == 0:
            _logger.info("No results. Run analyze_all_cells() first.")
            return
        
        # Get QC-passed cells
        qc_passed = self.results[self.results['QC_pass'] == True].copy()
        
        # Optionally filter for complete S phase only
        if complete_s_phase_only:
            qc_passed = qc_passed[qc_passed['Has_complete_S_phase'] == True].copy()
            filter_desc = "QC-passed with complete S phase"
        else:
            filter_desc = "QC-passed"
        
        if len(qc_passed) == 0:
            _logger.info(f"No {filter_desc} cells found.")
            return
        
        # Limit number of cells if specified
        if max_cells is not None and len(qc_passed) > max_cells:
            qc_passed = qc_passed.iloc[:max_cells]
            _logger.info(f"Limiting to first {max_cells} {filter_desc} cells")
        
        n_cells = len(qc_passed)
        _logger.info(f"\n{'='*60}")
        _logger.info(f"Creating grid plot for {n_cells} {filter_desc} cells")
        _logger.info(f"{'='*60}")
        
        # Calculate grid dimensions
        nrows = int(np.ceil(n_cells / ncols))
        
        # Create figure with white background
        fig_width = ncols * figsize_per_cell[0]
        fig_height = nrows * figsize_per_cell[1]
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), 
                                facecolor='white')
        
        # Flatten axes array for easier indexing
        if n_cells == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        # Trace colors
        cdt1_color = '#2ca02c'  # Bright green
        geminin_color = '#d62728'  # Bright red
        
        # Plot each cell
        for idx, (_, cell_result) in enumerate(qc_passed.iterrows()):
            ax = axes[idx]
            cell_id = cell_result['Cell_ID']
            
            # Get cell data
            cell_data = self.data[self.data['Cell_ID'] == str(cell_id)].copy()
            cell_data = cell_data.sort_values('Time').reset_index(drop=True)
            
            if len(cell_data) < 5:
                ax.text(0.5, 0.5, f'Cell {cell_id}\nInsufficient data', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            times = np.array(cell_data['Time'].values, dtype=float)
            cdt1_signal = np.array(cell_data['CDT1_Signal'].values, dtype=float)
            geminin_signal = np.array(cell_data['Geminin_Signal'].values, dtype=float)
            
            # Smooth signals
            cdt1_smooth = self.smooth_signal(cdt1_signal)
            geminin_smooth = self.smooth_signal(geminin_signal)
            
            # Plot signals - clean style without backgrounds
            ax.plot(times, cdt1_smooth, color=cdt1_color, linewidth=2, 
                label='CDT1', zorder=3)
            ax.plot(times, geminin_smooth, color=geminin_color, linewidth=2, 
                label='Geminin', zorder=3)
            
            # Mark transitions with vertical lines if requested
            if show_transitions:
                if pd.notna(cell_result.get('G1S_time')):
                    ax.axvline(cell_result['G1S_time'], color='#1f77b4', 
                            linestyle='--', linewidth=1.5, alpha=0.6, zorder=2,
                            label='G1-S' if idx == 0 else '')
                if pd.notna(cell_result.get('SG2_time')):
                    ax.axvline(cell_result['SG2_time'], color='#ff7f0e', 
                            linestyle='--', linewidth=1.5, alpha=0.6, zorder=2,
                            label='S-G2' if idx == 0 else '')
            
            # Formatting - clean style
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(times[0] - 0.5, times[-1] + 0.5)
            
            # Simple title with cell ID
            title = f'Cell {cell_id}'
            # Add phase info if available
            if cell_result.get('Has_complete_S_phase'):
                g1_dur = cell_result.get('G1_duration', 0)
                s_dur = cell_result.get('S_phase_duration', 0)
                title += f' (G1={g1_dur:.1f}h, S={s_dur:.1f}h)'
            ax.set_title(title, fontsize=10, pad=8, fontweight='normal')
            
            # Only show x-label on bottom row
            if idx >= n_cells - ncols:
                ax.set_xlabel('Time (h)', fontsize=9)
            else:
                ax.set_xticklabels([])
            
            # Only show y-label on leftmost column  
            if idx % ncols == 0:
                ax.set_ylabel('Intensity (a.u.)', fontsize=9)
            else:
                ax.set_yticklabels([])
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(loc='best', fontsize=8, frameon=True, 
                        framealpha=0.9, edgecolor='black')
            
            # Clean axes style
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_linewidth(1)
            ax.spines['right'].set_linewidth(1)
            ax.spines['top'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)
            
            # Adjust tick parameters
            ax.tick_params(axis='both', which='major', labelsize=8, 
                        length=4, width=1, direction='out')
            
            # White background
            ax.set_facecolor('white')
        
        # Hide unused subplots
        for idx in range(n_cells, len(axes)):
            axes[idx].axis('off')
        
        # Add overall title
        title_text = f'{"Complete S Phase " if complete_s_phase_only else ""}QC-Passed Cells (n={n_cells})'
        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.995)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / 'qc_passed_cells_grid.pdf'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        _logger.info(f"✓ Saved grid plot: {save_path}")
        plt.show()
        
        # Print summary
        _logger.info(f"\nGrid layout: {nrows} rows × {ncols} columns")
        _logger.info(f"Total cells plotted: {n_cells} {filter_desc}")
        
        if not complete_s_phase_only:
            complete_s = qc_passed['Has_complete_S_phase'].sum()
            _logger.info(f"  - With complete S phase: {complete_s}/{n_cells} ({complete_s/n_cells*100:.1f}%)")
            _logger.info(f"  - Without complete S phase: {n_cells - complete_s}/{n_cells}")
        _logger.info(f"{'='*60}\n")



# ---------------------------
# End of module
# ---------------------------