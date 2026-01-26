#!/usr/bin/env python3
"""
Standalone visualization script to create zero-D parameter bar charts (R, stenosis, L)
for a given dataset and geometry.

Usage:
  python util/visualizations/plot_zero_d_params.py --set-name VMR --geo-name 0063_1001 --modalities NORMAL_JUNCTION BloodVesselJunction --verbose

It searches for calibrated JSON outputs and the geometric input JSON under
<data_root>/<set_name>/<geo_name>/ and writes a PNG to
results/param_comparison/<set_name>/<geo_name>/zero_d_parameter_bars.png
"""
import os
import argparse
import glob
import sys

# Ensure package imports work relative to repository
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from util.zerod_calibration.post_processing import plot_zero_d_parameter_bars


def find_modality_jsons(data_dir):
    """Find modality JSON files in the provided directory.

    Returns a dict mapping modality name -> json path.
    """
    modality_jsons = {}

    # geometric input candidates
    geom_candidates = []
    for pat in ('geometric_input.json', '*geometric*.json', '*_geometric.json'):
        geom_candidates.extend(glob.glob(os.path.join(data_dir, pat)))

    if geom_candidates:
        # prefer exact geometric_input.json if present
        exact = os.path.join(data_dir, 'geometric_input.json')
        chosen = exact if exact in geom_candidates else geom_candidates[0]
        modality_jsons['geometric'] = os.path.abspath(chosen)

    # Also prefer a bifurcations geometric JSON if present
    bif_candidates = glob.glob(os.path.join(data_dir, '*bifurcations*geometric*.json')) + glob.glob(os.path.join(data_dir, 'bifurcations_geometric_input.json'))
    if bif_candidates:
        bif_chosen = bif_candidates[0]
        modality_jsons['bifurcations'] = os.path.abspath(bif_chosen)

    # calibrated outputs matching pattern *_calibrated_output_*.json or *_calibrated_output*.json
    calib_patterns = ['*_calibrated_output_*.json', '*_calibrated_output*.json', 'calibrated_output_*.json']
    calib_files = []
    for pat in calib_patterns:
        calib_files.extend(glob.glob(os.path.join(data_dir, pat)))

    # Also include files that contain 'calibrated_output' anywhere
    extra = glob.glob(os.path.join(data_dir, '*calibrated_output*.json'))
    for p in extra:
        if p not in calib_files:
            calib_files.append(p)

    # Build modality name from filename
    import re
    for p in sorted(calib_files):
        bn = os.path.basename(p)
        # Try to extract modality after 'calibrated_output_'
        m = re.search(r'calibrated_output_([A-Za-z0-9_]+)\.json$', bn)
        if m:
            mod = m.group(1)
        else:
            # Fallback: use filename without extension
            mod = os.path.splitext(bn)[0]
        modality_jsons[mod] = os.path.abspath(p)

    return modality_jsons


def main():
    parser = argparse.ArgumentParser(description='Generate zero-D parameter bar chart for a set/geometry')
    parser.add_argument('--set-name', required=True, help='Dataset set name (e.g., VMR)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., 0063_1001)')
    parser.add_argument('--data-root', default=None, help='Root data/zeroD directory (default: ../../data/zeroD relative to script)')
    parser.add_argument('--output-root', default=None, help='Output results root (default: results/param_comparison)')
    parser.add_argument('--modalities', nargs='*', help='Optional list of modalities to include (e.g., NORMAL_JUNCTION BloodVesselJunction). If omitted, all detected modalities are used.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Resolve data_root
    if args.data_root:
        data_root = os.path.abspath(args.data_root)
    else:
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'zeroD'))

    data_dir = os.path.join(data_root, args.set_name, args.geo_name)
    if not os.path.exists(data_dir):
        print(f"Error: data directory not found: {data_dir}")
        return 2

    modality_jsons = find_modality_jsons(data_dir)
    if not modality_jsons:
        print(f"Error: No modality JSON files found in {data_dir}")
        return 2

    # If user provided modalities, filter
    if args.modalities:
        requested = set(args.modalities)
        # Always keep geometric if present
        filtered = {}
        if 'geometric' in modality_jsons:
            filtered['geometric'] = modality_jsons['geometric']
        for m in requested:
            if m in modality_jsons:
                filtered[m] = modality_jsons[m]
            else:
                print(f"Warning: Requested modality '{m}' not found in {data_dir}")
        modality_jsons = filtered
        if not modality_jsons:
            print("Error: No valid modalities found after filtering")
            return 2

    # Output directory
    if args.output_root:
        output_root = os.path.abspath(args.output_root)
    else:
        output_root = os.path.abspath(os.path.join('results', 'param_comparison'))

    out_dir = os.path.join(output_root, args.set_name, args.geo_name)
    os.makedirs(out_dir, exist_ok=True)

    if args.verbose:
        print(f"Found modality JSONs:")
        for k, v in modality_jsons.items():
            print(f"  {k}: {v}")
        print(f"Saving plot to: {out_dir}")

    out_name = 'zero_d_parameter_bars.png'
    out_path = plot_zero_d_parameter_bars(modality_jsons, output_dir=out_dir, output_name=out_name, verbose=args.verbose)
    if out_path:
        print(f"Saved zero-D parameter bar chart: {out_path}")
        return 0
    else:
        print("Failed to generate zero-D parameter bar chart")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
