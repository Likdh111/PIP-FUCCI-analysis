"""
Interactive cell review tool for Jupyter notebooks
No command-line arguments needed - just modify the configuration
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pip_fucci_analyzer import ImprovedPIPFUCCIAnalyzer

# ============================================================================
# CONFIGURATION - Edit these settings
# ============================================================================

DATASET_CONFIG = {
    'name': 'dataset',  # Change this to match your dataset
    'csv_path': 'data/combined_all_conditions.csv',
    'results_dir': 'results/dataset',
    'review_mode': 'interactive',  # Options: 'interactive' or 'grid'
    'plot_size': (12, 6)  # Width, height in inches - adjust for your screen
}

# ============================================================================
# REVIEW FUNCTIONS
# ============================================================================

# Show each QC-passed cell one at a time and ask keep (y) / exclude (n) / quit (q).
def review_cells_interactive(analyzer, results, output_dir, figsize=(12, 6)):
    """Interactive review with keyboard input
    
    Parameters:
    -----------
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches
        Common options:
        - (12, 6) - Large, easy to see
        - (10, 5) - Medium
        - (8, 4) - Smaller, fits more on screen
        - (14, 7) - Extra large for detailed review
    """
    
    qc_passed = results[results['QC_pass'] == True]
    complete_s = qc_passed[qc_passed['Has_complete_S_phase'] == True]
    
    print(f"\n{'='*70}")
    print(f"REVIEWING {len(complete_s)} CELLS")
    print(f"{'='*70}")
    print("Instructions:")
    print("  - Review each plot")
    print("  - Type 'y' and press Enter to KEEP cell (good)")
    print("  - Type 'n' and press Enter to EXCLUDE cell (bad)")
    print("  - Type 'q' and press Enter to QUIT and save progress")
    print("="*70)
    
    exclusion_list = []
    kept_list = []
    
    for i, (idx, cell) in enumerate(complete_s.iterrows(), 1):
        cell_id = cell['Cell_ID']
        
        print(f"\n[{i}/{len(complete_s)}] Cell: {cell_id}")
        print(f"  G1: {cell['G1_duration']:.1f}h, S: {cell['S_phase_duration']:.1f}h")
        
        # Plot cell with custom size
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get cell data and plot
        cell_data = analyzer.data[analyzer.data['Cell_ID'] == str(cell_id)].copy()
        cell_data = cell_data.sort_values('Time').reset_index(drop=True)
        
        times = cell_data['Time'].values
        cdt1_smooth = analyzer.smooth_signal(cell_data['CDT1_Signal'].values)
        geminin_smooth = analyzer.smooth_signal(cell_data['Geminin_Signal'].values)
        
        # Colors
        cdt1_color = '#27ae60'
        geminin_color = '#e67e22'
        
        # Plot phase backgrounds if transitions detected
        if pd.notna(cell.get('G1S_time')) and pd.notna(cell.get('SG2_time')):
            ax.axvspan(times[0], cell['G1S_time'], alpha=0.08, color='green', label='G1', zorder=0)
            ax.axvspan(cell['G1S_time'], cell['SG2_time'], alpha=0.08, color='blue', label='S', zorder=0)
            ax.axvspan(cell['SG2_time'], times[-1], alpha=0.08, color='gold', label='G2', zorder=0)
        
        # Plot signals
        ax.plot(times, cdt1_smooth, color=cdt1_color, linewidth=2.5, label='CDT1', zorder=3, alpha=0.9)
        ax.plot(times, geminin_smooth, color=geminin_color, linewidth=2.5, label='Geminin', zorder=3, alpha=0.9)
        
        # Mark transitions
        if pd.notna(cell.get('G1S_time')):
            ax.axvline(cell['G1S_time'], color='#3498db', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        if pd.notna(cell.get('SG2_time')):
            ax.axvline(cell['SG2_time'], color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
        
        # Formatting
        ax.set_xlabel('Time (h)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Signal Intensity (a.u.)', fontsize=12, fontweight='bold')
        
        title = f'Cell {cell_id} | G1: {cell["G1_duration"]:.1f}h, S: {cell["S_phase_duration"]:.1f}h'
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', 
          fontsize=6, ncol=2, columnspacing=0.6, handlelength=1.0, 
          handletextpad=0.4, borderpad=0.3)
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.show()
        
        # Get user decision
        while True:
            decision = input("  Keep this cell? (y/n/q): ").lower().strip()
            if decision in ['y', 'n', 'q']:
                break
            print("  Invalid input. Enter 'y', 'n', or 'q'")
        
        if decision == 'n':
            exclusion_list.append(cell_id)
            print(f"  ✗ Excluded: {cell_id}")
        elif decision == 'y':
            kept_list.append(cell_id)
            print(f"  ✓ Keeping: {cell_id}")
        elif decision == 'q':
            print(f"\n  Quitting review. Progress saved.")
            break
        
        plt.close('all')  # Clean up
    
    # Save results
    output_dir = Path(output_dir)
    
    # Save exclusion list
    if exclusion_list:
        exclusion_file = output_dir / 'manual_exclusions.txt'
        with open(exclusion_file, 'w') as f:
            f.write("# Manually excluded cells (visual inspection)\n")
            f.write(f"# Dataset: {DATASET_CONFIG['name']}\n")
            f.write(f"# Total excluded: {len(exclusion_list)}\n\n")
            for cell_id in exclusion_list:
                f.write(f"{cell_id}\n")
        
        print(f"\n✓ Saved {len(exclusion_list)} exclusions to: {exclusion_file}")
    
    # Save review summary
    summary_file = output_dir / 'review_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"MANUAL REVIEW SUMMARY\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Dataset: {DATASET_CONFIG['name']}\n")
        f.write(f"Total cells reviewed: {i}\n")
        f.write(f"Cells kept: {len(kept_list)}\n")
        f.write(f"Cells excluded: {len(exclusion_list)}\n")
        f.write(f"Completion: {i}/{len(complete_s)} ({i/len(complete_s)*100:.1f}%)\n\n")
        
        if exclusion_list:
            f.write(f"Excluded cells:\n")
            for cell_id in exclusion_list:
                f.write(f"  - {cell_id}\n")
    
    print(f"✓ Saved review summary to: {summary_file}")
    
    # Apply exclusions to analyzer
    if exclusion_list:
        print(f"\nApplying exclusions to results...")
        analyzer.exclude_cells(exclusion_list, reason="MANUAL_VISUAL_REVIEW")
        print(f"✓ Exclusions applied and saved")
    
    return exclusion_list, kept_list


# Batch review: save a grid PDF, one PNG per cell and an exclusion template to edit.
def review_cells_grid(analyzer, results, output_dir):
    """Generate grid of all cells for visual review"""
    
    qc_passed = results[results['QC_pass'] == True]
    complete_s = qc_passed[qc_passed['Has_complete_S_phase'] == True]
    
    print(f"\nGenerating review materials for {len(complete_s)} cells...")
    
    output_dir = Path(output_dir)
    
    # 1. Create grid plot
    print("Creating grid plot...")
    analyzer.plot_qc_passed_cells_grid(
        save_path=output_dir / 'review_grid.pdf',
        complete_s_phase_only=True,
        ncols=4,
        max_cells=None  # Plot all
    )
    
    # 2. Save individual plots
    plot_dir = output_dir / 'individual_plots'
    plot_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving individual plots to {plot_dir}...")
    for i, cell_id in enumerate(complete_s['Cell_ID'], 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(complete_s)}")
        save_path = plot_dir / f"cell_{cell_id}.png"
        analyzer.plot_cell(cell_id, save_path=save_path)
        plt.close('all')
    
    # 3. Create review template
    template_file = output_dir / 'manual_exclusions_template.txt'
    with open(template_file, 'w') as f:
        f.write("# Manual exclusions for visual review\n")
        f.write(f"# Dataset: {DATASET_CONFIG['name']}\n")
        f.write("# Instructions:\n")
        f.write("#   1. Review the grid plot and individual plots\n")
        f.write("#   2. Uncomment (remove #) cell IDs you want to EXCLUDE\n")
        f.write("#   3. Save this file as 'manual_exclusions.txt'\n")
        f.write("#   4. Run the apply_exclusions() function\n\n")
        
        for cell_id in complete_s['Cell_ID']:
            f.write(f"# {cell_id}\n")
    
    print(f"\n✓ Grid review materials created:")
    print(f"  - Grid plot: {output_dir / 'review_grid.pdf'}")
    print(f"  - Individual plots: {plot_dir}")
    print(f"  - Exclusion template: {template_file}")
    print(f"\nNext steps:")
    print(f"  1. Review the plots")
    print(f"  2. Edit {template_file} to mark exclusions")
    print(f"  3. Save as 'manual_exclusions.txt' in same directory")
    print(f"  4. Run: apply_exclusions_from_file(analyzer)")


# Read the edited manual_exclusions.txt and exclude the listed cells.
def apply_exclusions_from_file(analyzer):
    """Apply exclusions from manual_exclusions.txt file"""
    
    exclusion_file = Path(analyzer.output_dir) / 'manual_exclusions.txt'
    
    if not exclusion_file.exists():
        print(f"✗ No exclusion file found: {exclusion_file}")
        return
    
    # Read exclusion file
    exclusions = []
    with open(exclusion_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                exclusions.append(line)
    
    if not exclusions:
        print("No cells marked for exclusion in file")
        return
    
    print(f"Found {len(exclusions)} cells to exclude")
    print(f"Applying exclusions...")
    
    result = analyzer.exclude_cells(exclusions, reason="MANUAL_VISUAL_REVIEW")
    
    print(f"✓ Exclusions applied and saved")
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Standalone review of a previously analyzed dataset, using DATASET_CONFIG above.
def run_review():
    """Main review function"""
    
    config = DATASET_CONFIG
    
    print(f"{'='*70}")
    print(f"PIP-FUCCI CELL REVIEW")
    print(f"{'='*70}")
    print(f"Dataset: {config['name']}")
    print(f"Mode: {config['review_mode']}")
    print(f"{'='*70}\n")
    
    # Check files exist
    csv_path = Path(config['csv_path'])
    results_dir = Path(config['results_dir'])
    results_file = results_dir / 'pip_fucci_analysis_results.csv'
    
    if not csv_path.exists():
        print(f"✗ CSV file not found: {csv_path}")
        return
    
    if not results_file.exists():
        print(f"✗ Results file not found: {results_file}")
        print(f"  Run analysis first!")
        return
    
    # Load analyzer
    print(f"Loading data...")
    analyzer = ImprovedPIPFUCCIAnalyzer(
        csv_file_path=str(csv_path),
        output_dir=str(results_dir)
    )
    
    # Load existing results
    print(f"Loading results from: {results_file}")
    analyzer.results = pd.read_csv(results_file)
    
    # Print summary
    qc_passed = analyzer.results[analyzer.results['QC_pass'] == True]
    complete_s = qc_passed[qc_passed['Has_complete_S_phase'] == True]
    
    print(f"\nCurrent status:")
    print(f"  Total cells: {len(analyzer.results)}")
    print(f"  QC passed: {len(qc_passed)}")
    print(f"  Complete S phase: {len(complete_s)}")
    print()
    
    # Run review
    if config['review_mode'] == 'interactive':
        figsize = config.get('plot_size', (12, 6))  # Default to (12, 6) if not specified
        print(f"Plot size: {figsize[0]}\" x {figsize[1]}\"")
        exclusions, kept = review_cells_interactive(
            analyzer, analyzer.results, analyzer.output_dir, figsize=figsize
        )
        return analyzer, exclusions, kept
    else:
        review_cells_grid(analyzer, analyzer.results, analyzer.output_dir)
        return analyzer


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # For Jupyter: just run the cell
    # For Python script: uncomment below
    # analyzer = run_review()
    
    print("\n" + "="*70)
    print("USAGE IN JUPYTER NOTEBOOK")
    print("="*70)
    print("\n1. INTERACTIVE MODE (one-by-one review):")
    print("   DATASET_CONFIG['review_mode'] = 'interactive'")
    print("   DATASET_CONFIG['plot_size'] = (12, 6)  # Adjust size as needed")
    print("   analyzer, exclusions, kept = run_review()")
    
    print("\n   Plot size options:")
    print("   - (14, 7) - Extra large for detailed review")
    print("   - (12, 6) - Large (default)")
    print("   - (10, 5) - Medium")
    print("   - (8, 4) - Smaller")
    
    print("\n2. GRID MODE (batch review):")
    print("   DATASET_CONFIG['review_mode'] = 'grid'")
    print("   analyzer = run_review()")
    print("   # ... review plots, edit exclusion file ...")
    print("   apply_exclusions_from_file(analyzer)")
    
    print("\n3. TO REVIEW ANOTHER DATASET:")
    print("   # Change DATASET_CONFIG at the top")
    print("   # Then run: run_review()")
    
    print("\n" + "="*70)