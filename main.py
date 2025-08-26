#!/usr/bin/env python3
# main.py - simple runner for complete analysis with visualizations

import sys
import argparse
from pathlib import Path

def main():
    """main entry point"""
    
    parser = argparse.ArgumentParser(
        description='ml analysis for temporal activity patterns with visualizations'
    )
    parser.add_argument('--data-dir', default='ml_data', help='data directory')
    parser.add_argument('--output-dir', default='ml_results', help='output directory')
    parser.add_argument('--quick', action='store_true', help='quick analysis with essential plots')
    parser.add_argument('--no-plots', action='store_true', help='skip visualizations')
    parser.add_argument('--parallel', action='store_true', help='use parallel processing')
    
    args = parser.parse_args()
    
    # check if modules exist
    try:
        from workflow_with_plots import run_complete_analysis_with_plots
        print("running analysis with comprehensive visualizations...")
        
        results = run_complete_analysis_with_plots(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_parallel=args.parallel,
            create_plots=not args.no_plots,
            n_plot_examples=12 if args.quick else 20
        )
        
    except ImportError as e:
        print(f"import error: {e}")
        print("\ntrying basic workflow...")
        
        try:
            from workflow import run_complete_ml_analysis
            results = run_complete_ml_analysis(args.data_dir, args.output_dir)
        except ImportError:
            print("error: workflow modules not found")
            print("\nrequired files:")
            print("  - ml_models.py")
            print("  - workflow.py or workflow_with_plots.py")
            print("  - visualizations.py (for plots)")
            print("  - parallel_workflow.py (for parallel processing)")
            sys.exit(1)
    
    if results:
        print("\nanalysis complete!")
        print(f"check {args.output_dir}/ for results and visualizations")
    else:
        print("\nanalysis failed - check error messages above")

if __name__ == "__main__":
    main()
