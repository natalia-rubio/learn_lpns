#!/usr/bin/env python3
"""
Wrapper script to visualize vessel boundaries for both bifurcations and EL-adjusted geometries.

This script runs plot_vessel_boundaries.py on:
1. The bifurcations split geometry (bifurcations_geometric_input.json)
2. The EL-adjusted geometry (bifurcations_EL_geometric_input.json)
"""

import os
import sys
import argparse
import subprocess


def find_centerline_file(set_name, geo_name, data_dir='data'):
    """
    Find centerline file in various possible locations.
    
    Returns:
        Path to centerline file, or None if not found
    """
    possible_paths = [
        os.path.join(data_dir, 'oneD', set_name, geo_name, 'unsteady_soln.vtp'),
        os.path.join(data_dir, 'oneD', "VMR_rigid_aortas", geo_name, 'unsteady_soln.vtp'),
        os.path.join(data_dir, 'oneD', set_name, geo_name, 'centerlines_simVascular.vtp'),
        os.path.join(data_dir, 'threeD', set_name, geo_name, 'centerlines_simVascular.vtp'),
        os.path.join(data_dir, 'threeD', set_name, geo_name, 'centerlines', 'centerlines.vtp'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def plot_geometry_variant(set_name, geo_name, variant_name, geometric_input_path, 
                          centerline_path, output_dir, data_dir='data'):
    """
    Plot vessel boundaries for a specific geometry variant.
    
    Args:
        set_name: Dataset name (e.g., 'VMR')
        geo_name: Geometry name (e.g., '0063_1001')
        variant_name: Variant name ('bifurcations' or 'EL_adjusted')
        geometric_input_path: Path to geometric input JSON
        centerline_path: Path to centerline VTP file
        output_dir: Directory to save output PNG
        data_dir: Base data directory
    
    Returns:
        Path to saved PNG, or None if failed
    """
    # Construct output path
    output_filename = f"{variant_name}_vessel_boundaries.png"
    output_path = os.path.join(output_dir, 'splitting_visualization', set_name, geo_name, output_filename)
    
    # Check if geometric input exists
    if not os.path.exists(geometric_input_path):
        print(f"  ⊘ Skipping {variant_name}: geometric input not found: {geometric_input_path}")
        return None
    
    # Create title
    title = f"{geo_name} - {variant_name.replace('_', ' ').title()} Geometry"
    
    # Run plot_vessel_boundaries.py
    script_path = os.path.join(os.path.dirname(__file__), 'plot_vessel_boundaries.py')
    
    cmd = [
        sys.executable,
        script_path,
        '--geometric-input', geometric_input_path,
        '--centerline', centerline_path,
        '--output', output_path,
        '--title', title,
    ]
    
    print(f"\n  Plotting {variant_name} geometry...")
    print(f"    Input: {geometric_input_path}")
    print(f"    Output: {output_path}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n    ✓ Success")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"\n    ✗ Failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Visualize vessel boundaries for bifurcations and EL-adjusted geometries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot vessel boundaries for a geometry
  python3 util/visualizations/plot_vessel_boundaries_wrapper.py \\
      --set-name VMR \\
      --geo-name 0063_1001
  
  # Specify custom data and output directories
  python3 util/visualizations/plot_vessel_boundaries_wrapper.py \\
      --set-name VMR \\
      --geo-name 0063_1001 \\
      --data-dir /path/to/data \\
      --output-dir /path/to/results

  # Use zeroD data under a run-config subfolder (e.g. stenosis_off)
  python3 util/visualizations/plot_vessel_boundaries_wrapper.py \\
      --set-name VMR_rigid_aorta_adults \\
      --geo-name 0094_0001 \\
      --run-config stenosis_off
        """
    )
    
    parser.add_argument('--set-name', type=str, required=True,
                       help='Dataset name (e.g., VMR)')
    parser.add_argument('--geo-name', type=str, required=True,
                       help='Geometry name (e.g., 0063_1001)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Base data directory (default: data)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Base output directory (default: results)')
    parser.add_argument('--run-config', type=str, default=None,
                       help='ZeroD run-config subfolder (e.g. stenosis_off, base). If set, geometric inputs are read from data/zeroD/<set_name>/<run-config>/<geo_name>/')
    parser.add_argument('--skip-bifurcations', action='store_true',
                       help='Skip plotting bifurcations geometry')
    parser.add_argument('--skip-el-adjusted', action='store_true',
                       help='Skip plotting EL-adjusted geometry')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Vessel Boundaries Visualization Wrapper")
    print(f"{'='*60}")
    print(f"  Set: {args.set_name}")
    print(f"  Geometry: {args.geo_name}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    if args.run_config:
        print(f"  Run config (zeroD): {args.run_config}")
    
    # Find centerline file
    print(f"\nFinding centerline file...")
    centerline_path = find_centerline_file(args.set_name, args.geo_name, args.data_dir)
    
    if centerline_path is None:
        print(f"  ✗ Error: Centerline file not found")
        print(f"    Searched in:")
        print(f"      - {os.path.join(args.data_dir, 'oneD', args.set_name, args.geo_name, 'unsteady_soln.vtp')}")
        print(f"      - {os.path.join(args.data_dir, 'oneD', args.set_name, args.geo_name, 'centerlines_simVascular.vtp')}")
        print(f"      - {os.path.join(args.data_dir, 'threeD', args.set_name, args.geo_name, 'centerlines_simVascular.vtp')}")
        sys.exit(1)
    
    print(f"  ✓ Found: {centerline_path}")
    
    # Define geometry paths (zeroD may be under set_name/run_config/geo_name when --run-config is set)
    if args.run_config:
        zerod_dir = os.path.join(args.data_dir, 'zeroD', args.set_name, args.run_config, args.geo_name)
    else:
        zerod_dir = os.path.join(args.data_dir, 'zeroD', args.set_name, args.geo_name)
    
    bifurcations_path = os.path.join(zerod_dir, 'bifurcations_geometric_input.json')
    el_adjusted_path = os.path.join(zerod_dir, 'bifurcations_EL_geometric_input.json')
    
    # Plot bifurcations geometry
    bifurcations_output = None
    if not args.skip_bifurcations:
        bifurcations_output = plot_geometry_variant(
            args.set_name,
            args.geo_name,
            'bifurcations',
            bifurcations_path,
            centerline_path,
            args.output_dir,
            args.data_dir
        )
    else:
        print(f"\n  ⊘ Skipping bifurcations geometry (--skip-bifurcations)")
    
    # Plot EL-adjusted geometry
    el_adjusted_output = None
    if not args.skip_el_adjusted:
        el_adjusted_output = plot_geometry_variant(
            args.set_name,
            args.geo_name,
            'EL_adjusted',
            el_adjusted_path,
            centerline_path,
            args.output_dir,
            args.data_dir
        )
    else:
        print(f"\n  ⊘ Skipping EL-adjusted geometry (--skip-el-adjusted)")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    if bifurcations_output:
        print(f"  ✓ Bifurcations: {bifurcations_output}")
    else:
        print(f"  ✗ Bifurcations: Failed or skipped")
    
    if el_adjusted_output:
        print(f"  ✓ EL-adjusted: {el_adjusted_output}")
    else:
        print(f"  ✗ EL-adjusted: Failed or skipped")
    
    if bifurcations_output or el_adjusted_output:
        print(f"\n  ✓ Visualization complete")
    else:
        print(f"\n  ✗ No visualizations were created")
        sys.exit(1)


if __name__ == '__main__':
    main()

