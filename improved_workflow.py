"""
Improved PIP-FUCCI Analysis Workflow with Persistent Exclusions
Allows re-running analysis without repeating manual review
"""

from pathlib import Path
from pip_fucci_analyzer import ImprovedPIPFUCCIAnalyzer
import pandas as pd
import json
from datetime import datetime

# ============================================================================
# PERSISTENT EXCLUSION MANAGEMENT
# ============================================================================

# Add excluded cell IDs to manual_exclusions.json (with a timestamped history).
def save_exclusions(analyzer, exclusions, reason="MANUAL_REVIEW"):
    """Save exclusions to persistent JSON file"""
    
    exclusion_file = Path(analyzer.output_dir) / 'manual_exclusions.json'
    
    # Load existing exclusions if file exists
    if exclusion_file.exists():
        with open(exclusion_file, 'r') as f:
            data = json.load(f)
    else:
        data = {'exclusions': [], 'history': []}
    
    # Add new exclusions
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_entry = {
        'timestamp': timestamp,
        'reason': reason,
        'cells': exclusions
    }
    data['history'].append(new_entry)
    
    # Update master exclusion list (remove duplicates)
    all_excluded = set(data.get('exclusions', []))
    all_excluded.update(exclusions)
    data['exclusions'] = sorted(list(all_excluded))
    
    # Save
    with open(exclusion_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Saved {len(exclusions)} exclusions to: {exclusion_file}")
    print(f"  Total excluded cells: {len(data['exclusions'])}")
    
    return exclusion_file


# Read manual_exclusions.json and apply those exclusions to the analyzer.
def load_exclusions(analyzer):
    """Load persistent exclusions and apply them"""
    
    exclusion_file = Path(analyzer.output_dir) / 'manual_exclusions.json'
    
    if not exclusion_file.exists():
        print("No saved exclusions found")
        return []
    
    with open(exclusion_file, 'r') as f:
        data = json.load(f)
    
    exclusions = data.get('exclusions', [])
    
    if exclusions:
        print(f"\n{'='*70}")
        print("LOADING SAVED EXCLUSIONS")
        print(f"{'='*70}")
        print(f"Found {len(exclusions)} previously excluded cells")
        print(f"Source: {exclusion_file}")
        
        # Show history
        if 'history' in data:
            print(f"\nExclusion history:")
            for entry in data['history']:
                print(f"  {entry['timestamp']}: {len(entry['cells'])} cells ({entry['reason']})")
        
        # Apply exclusions
        analyzer.exclude_cells(exclusions, reason="LOADED_FROM_FILE", save_updated=True)
        print(f"✓ Exclusions applied")
        print(f"{'='*70}\n")
    
    return exclusions


# Back up and delete manual_exclusions.json.
def clear_exclusions(analyzer):
    """Clear all saved exclusions (use with caution!)"""
    
    exclusion_file = Path(analyzer.output_dir) / 'manual_exclusions.json'
    
    if exclusion_file.exists():
        # Backup before deleting
        backup_file = exclusion_file.parent / f"{exclusion_file.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import shutil
        shutil.copy(exclusion_file, backup_file)
        print(f"✓ Backup saved to: {backup_file}")
        
        exclusion_file.unlink()
        print(f"✓ Cleared exclusions from: {exclusion_file}")
    else:
        print("No exclusion file to clear")


# ============================================================================
# MODIFIED WORKFLOW STEPS
# ============================================================================

# Create the analyzer and run transition detection + QC on all cells with the standard settings.
def step1_initial_analysis(config, load_saved_exclusions=True):
    """
    Run initial analysis with option to load saved exclusions
    
    Parameters:
    -----------
    config : dict
        Dataset configuration
    load_saved_exclusions : bool, default=True
        If True, automatically loads and applies saved exclusions
    """
    
    print("\n" + "="*70)
    print("STEP 1: INITIAL ANALYSIS")
    print("="*70)
    
    analyzer = ImprovedPIPFUCCIAnalyzer(
        csv_file_path=config['csv_path'],
        output_dir=config['output_dir'],
        use_raw_qc=True
    )
    
    results = analyzer.analyze_all_cells(
        sg2_sensitivity=config['sg2_sensitivity'],
        min_g1_duration=config.get('min_g1_duration', 2.0),
        sustained_drop_duration=0.5,
        sustained_rise_duration=0.5,
        detect_outliers=False  # Do this AFTER manual review
    )
    
    # Load saved exclusions if requested — otherwise wipe any existing JSON
    # so that a subsequent save_exclusions() call starts from a clean slate
    if load_saved_exclusions:
        loaded_exclusions = load_exclusions(analyzer)
        if loaded_exclusions:
            print(f"✓ Using {len(loaded_exclusions)} saved exclusions")
    else:
        exclusion_file = Path(analyzer.output_dir) / 'manual_exclusions.json'
        if exclusion_file.exists():
            import shutil
            backup_file = exclusion_file.parent / f"manual_exclusions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy(exclusion_file, backup_file)
            exclusion_file.unlink()
            print(f"⚠  Fresh start: existing exclusions backed up to {backup_file.name} and cleared")
    
    # Print summary
    qc_passed = analyzer.results[analyzer.results['QC_pass'] == True]
    complete_s = qc_passed[qc_passed['Has_complete_S_phase'] == True]
    
    print(f"\n{'='*70}")
    print("STEP 1 COMPLETE")
    print(f"{'='*70}")
    print(f"Total cells: {len(results)}")
    print(f"QC passed: {len(qc_passed)} ({len(qc_passed)/len(results)*100:.1f}%)")
    print(f"Complete S phase: {len(complete_s)}")
    print(f"{'='*70}\n")
    
    return analyzer


# Run the visual cell review and save the excluded cells to manual_exclusions.json.
def step2_manual_review_with_save(analyzer, config, mode='interactive'):
    """
    Manual review that saves exclusions to persistent file
    """
    
    from cell_review_jupyter import review_cells_interactive, review_cells_grid
    
    print("\n" + "="*70)
    print("STEP 2: MANUAL REVIEW (with auto-save)")
    print(f"Mode: {mode}")
    print("="*70)
    
    if mode == 'interactive':
        figsize = config.get('plot_size', (4, 3))
        exclusions, kept = review_cells_interactive(
            analyzer, analyzer.results, analyzer.output_dir, figsize=figsize
        )
        
        # Save exclusions to persistent file
        if exclusions:
            save_exclusions(analyzer, exclusions, reason="MANUAL_VISUAL_REVIEW")
        
        print(f"\n{'='*70}")
        print("STEP 2 COMPLETE")
        print(f"{'='*70}")
        print(f"Cells excluded: {len(exclusions)}")
        print(f"Cells kept: {len(kept)}")
        print(f"✓ Exclusions saved - will be auto-loaded on next run")
        print(f"{'='*70}\n")
        
        return exclusions, kept
    
    else:  # grid mode
        review_cells_grid(analyzer, analyzer.results, analyzer.output_dir)
        return None, None


# ============================================================================
# DATA EXTRACTION FOR CUSTOM PLOTS
# ============================================================================

# Save the data behind the population summary figure (histograms, scatter points) as CSVs.
def extract_population_summary_data(analyzer):
    """
    Extract exact data used in population_summary plot
    Returns binned histogram data for each subplot
    """
    
    clean_data = analyzer.results[analyzer.results['QC_pass'] == True].copy()
    complete_s = clean_data[clean_data['Has_complete_S_phase'] == True].copy()
    
    if len(complete_s) == 0:
        return None
    
    import numpy as np
    
    # 1. G1 Duration histogram data
    g1_data = complete_s['G1_duration'].dropna()
    g1_hist, g1_bins = np.histogram(g1_data, bins=15)
    g1_bin_centers = (g1_bins[:-1] + g1_bins[1:]) / 2
    
    # 2. S Duration histogram data
    s_data = complete_s['S_phase_duration'].dropna()
    s_hist, s_bins = np.histogram(s_data, bins=15)
    s_bin_centers = (s_bins[:-1] + s_bins[1:]) / 2
    
    # 3. Transition scatter data
    transition_data = complete_s[['G1S_time', 'SG2_time']].dropna()
    
    # 4. G1 vs S scatter data
    g1_vs_s = complete_s[['G1_duration', 'S_phase_duration']].dropna()
    
    # Create DataFrames
    g1_histogram = pd.DataFrame({
        'bin_center': g1_bin_centers,
        'count': g1_hist,
        'bin_start': g1_bins[:-1],
        'bin_end': g1_bins[1:]
    })
    
    s_histogram = pd.DataFrame({
        'bin_center': s_bin_centers,
        'count': s_hist,
        'bin_start': s_bins[:-1],
        'bin_end': s_bins[1:]
    })
    
    # Save all data
    output_dir = Path(analyzer.output_dir)
    
    g1_histogram.to_csv(output_dir / 'population_g1_histogram.csv', index=False)
    s_histogram.to_csv(output_dir / 'population_s_histogram.csv', index=False)
    transition_data.to_csv(output_dir / 'population_transitions.csv', index=False)
    g1_vs_s.to_csv(output_dir / 'population_g1_vs_s.csv', index=False)
    
    print(f"\n{'='*70}")
    print("POPULATION SUMMARY DATA EXTRACTED")
    print(f"{'='*70}")
    print(f"Saved to {output_dir}:")
    print(f"  1. population_g1_histogram.csv")
    print(f"  2. population_s_histogram.csv")
    print(f"  3. population_transitions.csv")
    print(f"  4. population_g1_vs_s.csv")
    print(f"{'='*70}\n")
    
    return {
        'g1_histogram': g1_histogram,
        's_histogram': s_histogram,
        'transitions': transition_data,
        'g1_vs_s': g1_vs_s,
        'g1_raw': g1_data,
        's_raw': s_data
    }


# Compute G2/M duration and save per-cell durations and summary statistics as CSVs.
def extract_plot_data(analyzer, save_to_csv=True):
    """
    Extract raw data from analyzer for custom plotting
    
    Returns dictionary with all plot-ready data:
    - phase_durations: G1, S, G2/M for each cell
    - transitions: G1-S and S-G2 timing
    - population_stats: Summary statistics
    """
    
    # Get clean data
    clean_data = analyzer.results[analyzer.results['QC_pass'] == True].copy()
    complete_s = clean_data[clean_data['Has_complete_S_phase'] == True].copy()
    
    if len(complete_s) == 0:
        print("No clean data available")
        return None
    
    # Calculate G2/M
    complete_s['G2M_duration'] = complete_s['End_time'] - complete_s['SG2_time']
    
    # 1. Phase durations (for violin/histogram plots)
    phase_data = []
    for _, row in complete_s.iterrows():
        if pd.notna(row.get('G1_duration')):
            phase_data.append({
                'Cell_ID': row['Cell_ID'],
                'Phase': 'G1',
                'Duration_hours': row['G1_duration']
            })
        if pd.notna(row.get('S_phase_duration')):
            phase_data.append({
                'Cell_ID': row['Cell_ID'],
                'Phase': 'S',
                'Duration_hours': row['S_phase_duration']
            })
        if pd.notna(row.get('G2M_duration')) and row['G2M_duration'] > 0:
            phase_data.append({
                'Cell_ID': row['Cell_ID'],
                'Phase': 'G2M',
                'Duration_hours': row['G2M_duration']
            })
    
    phase_df = pd.DataFrame(phase_data)
    
    # 2. Transition timing
    transition_df = complete_s[['Cell_ID', 'G1S_time', 'SG2_time']].copy()
    
    # 3. Individual cell data
    cell_df = complete_s[[
        'Cell_ID', 'G1_duration', 'S_phase_duration', 'G2M_duration',
        'G1S_time', 'SG2_time', 'Start_time', 'End_time'
    ]].copy()
    
    # 4. Summary statistics
    stats = {}
    for phase in ['G1', 'S', 'G2M']:
        phase_subset = phase_df[phase_df['Phase'] == phase]['Duration_hours']
        if len(phase_subset) > 0:
            stats[phase] = {
                'n': len(phase_subset),
                'mean': phase_subset.mean(),
                'std': phase_subset.std(),
                'median': phase_subset.median(),
                'q25': phase_subset.quantile(0.25),
                'q75': phase_subset.quantile(0.75),
                'min': phase_subset.min(),
                'max': phase_subset.max()
            }
    
    stats_df = pd.DataFrame(stats).T
    
    # Save to CSV if requested
    if save_to_csv:
        output_dir = Path(analyzer.output_dir)
        
        phase_df.to_csv(output_dir / 'phase_durations_long_format.csv', index=False)
        cell_df.to_csv(output_dir / 'cell_data_wide_format.csv', index=False)
        stats_df.to_csv(output_dir / 'summary_statistics.csv')
        
        print(f"\n{'='*70}")
        print("DATA EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Saved to {output_dir}:")
        print(f"  1. phase_durations_long_format.csv - For plotting (n={len(phase_df)} rows)")
        print(f"  2. cell_data_wide_format.csv - One row per cell (n={len(cell_df)} cells)")
        print(f"  3. summary_statistics.csv - Statistics by phase")
        print(f"{'='*70}\n")
    
    return {
        'phase_durations': phase_df,
        'transitions': transition_df,
        'cells': cell_df,
        'statistics': stats_df
    }


# ============================================================================
# SMART RE-RUN FUNCTION
# ============================================================================

# Re-run the full analysis using saved exclusions (no manual review).
def rerun_analysis(config, regenerate_plots=True, force_manual_review=False):
    """
    Smart re-run that uses saved exclusions
    
    Parameters:
    -----------
    config : dict
        Dataset configuration
    regenerate_plots : bool, default=True
        Whether to regenerate all plots
    force_manual_review : bool, default=False
        If True, runs manual review even if saved exclusions exist
    
    Returns:
    --------
    analyzer : Analyzer with results
    """
    
    print("\n" + "="*70)
    print("SMART RE-RUN MODE")
    print("="*70)
    print("This will use saved exclusions (if available)")
    print("No need to manually review cells again!")
    print("="*70 + "\n")
    
    # Step 1: Load data and apply saved exclusions
    analyzer = step1_initial_analysis(config, load_saved_exclusions=True)
    
    # Step 2: Manual review only if forced or no saved exclusions
    exclusion_file = Path(analyzer.output_dir) / 'manual_exclusions.json'
    
    if force_manual_review or not exclusion_file.exists():
        print("\nRunning manual review...")
        exclusions, kept = step2_manual_review_with_save(analyzer, config)
    else:
        print("\n✓ Using saved exclusions - skipping manual review")
    
    # Step 3: Outlier detection
    print("\nDetecting outliers...")
    analyzer.detect_phase_outliers(iqr_multiplier=1.5)
    
    # Step 4: Generate plots if requested
    if regenerate_plots:
        print("\nGenerating plots...")
        analyzer.plot_phase_durations_violin(qc_passed_only=True)
        analyzer.plot_population_summary(qc_passed_only=True)
        analyzer.plot_qc_passed_cells_grid(complete_s_phase_only=True)
    
    # Step 5: Extract data
    print("\nExtracting plot data...")
    plot_data = extract_plot_data(analyzer, save_to_csv=True)
    
    print("\n" + "="*70)
    print("✓ RE-RUN COMPLETE")
    print("="*70)
    print(f"Results in: {analyzer.output_dir}")
    print(f"Extracted data CSVs ready for custom plotting")
    print("="*70 + "\n")
    
    return analyzer, plot_data


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("""
IMPROVED WORKFLOW USAGE:

1. FIRST TIME ANALYSIS:
   analyzer = step1_initial_analysis(DATASET_CONFIG)
   exclusions, kept = step2_manual_review_with_save(analyzer, DATASET_CONFIG)
   # Exclusions are automatically saved to manual_exclusions.json

2. RE-RUN AFTER TWEAKING (no manual review needed!):
   analyzer, plot_data = rerun_analysis(DATASET_CONFIG)
   # Automatically loads saved exclusions
   
3. EXTRACT DATA FOR CUSTOM PLOTS:
   plot_data = extract_plot_data(analyzer)
   # Access: plot_data['phase_durations'], plot_data['cells'], etc.

4. FORCE NEW MANUAL REVIEW:
   analyzer, plot_data = rerun_analysis(DATASET_CONFIG, force_manual_review=True)

5. CLEAR ALL EXCLUSIONS (start fresh):
   clear_exclusions(analyzer)

KEY FILES:
- manual_exclusions.json: Persistent exclusion list (auto-loaded)
- phase_durations_long_format.csv: Raw data for custom plots
- cell_data_wide_format.csv: One row per cell
- summary_statistics.csv: Mean, SD, median by phase
    """)