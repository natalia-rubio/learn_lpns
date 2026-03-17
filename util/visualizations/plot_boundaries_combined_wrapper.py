#!/usr/bin/env python3
"""
Wrapper to create one combined figure: original (bifurcations) schematic left, EL-adjusted right.
Uses LaTeX. Calls plot_boundaries_combined.py with both geometric inputs.
"""

import os
import sys
import argparse
import subprocess


def find_centerline_file(set_name, geo_name, data_dir='data'):
    """Find centerline file in various possible locations."""
    possible_paths = [
        os.path.join(data_dir, 'oneD', set_name, geo_name, 'unsteady_soln.vtp'),
        os.path.join(data_dir, 'oneD', 'VMR_rigid_aortas', geo_name, 'unsteady_soln.vtp'),
        os.path.join(data_dir, 'oneD', set_name, geo_name, 'centerlines_simVascular.vtp'),
        os.path.join(data_dir, 'threeD', set_name, geo_name, 'centerlines_simVascular.vtp'),
        os.path.join(data_dir, 'threeD', set_name, geo_name, 'centerlines', 'centerlines.vtp'),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Combined figure: original (bifurcations) | EL-adjusted schematics with LaTeX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 util/visualizations/plot_boundaries_combined_wrapper.py \\
      --set-name VMR_rigid_aorta_adults --geo-name 0094_0001 --run-config stenosis_off
        """
    )
    parser.add_argument('--set-name', type=str, required=True, help='Dataset name')
    parser.add_argument('--geo-name', type=str, required=True, help='Geometry name')
    parser.add_argument('--data-dir', type=str, default='data', help='Base data directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Base output directory')
    parser.add_argument('--run-config', type=str, default=None,
                        help='ZeroD run-config subfolder (e.g. stenosis_off)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Boundaries combined (original | EL-adjusted, LaTeX)")
    print(f"{'='*60}")
    print(f"  Set: {args.set_name}  Geometry: {args.geo_name}")
    if args.run_config:
        print(f"  Run config: {args.run_config}")

    centerline_path = find_centerline_file(args.set_name, args.geo_name, args.data_dir)
    if centerline_path is None:
        print("  ✗ Error: Centerline file not found")
        sys.exit(1)
    print(f"  ✓ Centerline: {centerline_path}")

    if args.run_config:
        zerod_dir = os.path.join(args.data_dir, 'zeroD', args.set_name, args.run_config, args.geo_name)
    else:
        zerod_dir = os.path.join(args.data_dir, 'zeroD', args.set_name, args.geo_name)
    bifurcations_path = os.path.join(zerod_dir, 'bifurcations_geometric_input.json')
    el_path = os.path.join(zerod_dir, 'bifurcations_EL_geometric_input.json')

    if not os.path.exists(bifurcations_path):
        print(f"  ✗ Bifurcations geometric input not found: {bifurcations_path}")
        sys.exit(1)
    if not os.path.exists(el_path):
        print(f"  ✗ EL-adjusted geometric input not found: {el_path}")
        sys.exit(1)

    output_path = os.path.join(args.output_dir, 'splitting_visualization', args.set_name, args.geo_name,
                               'boundaries_combined.png')
    title = f"{args.geo_name}"
    script_path = os.path.join(os.path.dirname(__file__), 'plot_boundaries_combined.py')
    cmd = [
        sys.executable,
        script_path,
        '--geometric-input-bifurcations', bifurcations_path,
        '--geometric-input-el', el_path,
        '--centerline', centerline_path,
        '--output', output_path,
        '--title', title,
    ]
    print(f"\n  Creating combined figure (bifurcations | EL-adjusted)...")
    print(f"    Output: {output_path}")
    print()
    try:
        subprocess.run(cmd, check=True)
        print(f"\n  ✓ Saved: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
