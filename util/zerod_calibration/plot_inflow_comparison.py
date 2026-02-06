#!/usr/bin/env python3
"""
Plot time vs pressure for INFLOW location comparing two CSV files.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from util.zerod_calibration.post_processing import read_zerod_csv

def find_inflow_vessel(geometric_input_path):
    """Find the vessel name with inlet boundary condition."""
    import json
    with open(geometric_input_path, 'r') as f:
        geo_input = json.load(f)
    
    vessels = geo_input.get('vessels', [])
    for vessel in vessels:
        if 'boundary_conditions' in vessel and 'inlet' in vessel.get('boundary_conditions', {}):
            return vessel.get('vessel_name')
    
    # Fallback: assume branch0_seg0 for VMR geometries
    return 'branch0_seg0'

def plot_inflow_comparison(csv1_path, csv2_path, output_path=None, geometric_input_path=None, vessel_name=None):
    """
    Plot time vs pressure for INFLOW location from two CSV files.
    
    Args:
        csv1_path: Path to first CSV file (e.g., downsampled version)
        csv2_path: Path to second CSV file (e.g., original version)
        output_path: Path to save plot (default: auto-generate)
        geometric_input_path: Path to geometric input JSON (to find INFLOW vessel)
        vessel_name: Vessel name for INFLOW (if None, will try to find from geometric input)
    """
    # Read both CSV files
    print(f"Reading CSV 1: {csv1_path}")
    results1, times1 = read_zerod_csv(csv1_path)
    
    print(f"Reading CSV 2: {csv2_path}")
    results2, times2 = read_zerod_csv(csv2_path)
    
    if not results1 or not results2:
        print("Error: Could not read one or both CSV files")
        return
    
    # Find INFLOW vessel name
    if vessel_name is None:
        if geometric_input_path and os.path.exists(geometric_input_path):
            vessel_name = find_inflow_vessel(geometric_input_path)
            print(f"Found INFLOW vessel from geometric input: {vessel_name}")
        else:
            # Try common names
            for name in ['branch0_seg0', 'branch_0_seg_0']:
                if name in results1 or name in results2:
                    vessel_name = name
                    print(f"Using vessel name: {vessel_name}")
                    break
    
    if vessel_name is None:
        print("Error: Could not determine INFLOW vessel name")
        print(f"Available vessels in CSV 1: {list(results1.keys())[:10]}")
        print(f"Available vessels in CSV 2: {list(results2.keys())[:10]}")
        return
    
    # Extract pressure_in data
    if vessel_name not in results1:
        print(f"Error: Vessel '{vessel_name}' not found in CSV 1")
        print(f"Available vessels: {list(results1.keys())[:10]}")
        return
    
    if vessel_name not in results2:
        print(f"Error: Vessel '{vessel_name}' not found in CSV 2")
        print(f"Available vessels: {list(results2.keys())[:10]}")
        return
    
    # Extract time and pressure arrays
    times1_sorted = sorted(times1)
    times2_sorted = sorted(times2)
    
    pressures1 = []
    pressures2 = []
    
    for t in times1_sorted:
        if t in results1[vessel_name] and 'pressure_in' in results1[vessel_name][t]:
            pressures1.append(results1[vessel_name][t]['pressure_in'])
        else:
            pressures1.append(np.nan)
    
    for t in times2_sorted:
        if t in results2[vessel_name] and 'pressure_in' in results2[vessel_name][t]:
            pressures2.append(results2[vessel_name][t]['pressure_in'])
        else:
            pressures2.append(np.nan)
    
    pressures1 = np.array(pressures1)
    pressures2 = np.array(pressures2)
    times1_sorted = np.array(times1_sorted)
    times2_sorted = np.array(times2_sorted)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Get base names for labels
    csv1_name = os.path.basename(csv1_path).replace('.csv', '')
    csv2_name = os.path.basename(csv2_path).replace('.csv', '')
    
    # Plot both datasets
    ax.plot(times1_sorted, pressures1, 'b-', linewidth=2, label=csv1_name, alpha=0.8)
    ax.plot(times2_sorted, pressures2, 'r--', linewidth=2, label=csv2_name, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Pressure (dynes/cm²)', fontsize=14)
    ax.set_title(f'INFLOW Pressure Comparison: {vessel_name}', fontsize=16, weight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add info text
    info_text = f"CSV 1: {len(times1_sorted)} points, range: [{times1_sorted[0]:.4f}, {times1_sorted[-1]:.4f}] s\n"
    info_text += f"CSV 2: {len(times2_sorted)} points, range: [{times2_sorted[0]:.4f}, {times2_sorted[-1]:.4f}] s"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        base_dir = os.path.dirname(csv1_path)
        output_path = os.path.join(base_dir, f'inflow_pressure_comparison_{vessel_name}.png')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot time vs pressure for INFLOW location comparing two CSV files"
    )
    parser.add_argument('csv1', help='Path to first CSV file (e.g., downsampled)')
    parser.add_argument('csv2', help='Path to second CSV file (e.g., original)')
    parser.add_argument('--output', '-o', help='Output path for plot (default: auto-generate)')
    parser.add_argument('--geometric-input', help='Path to geometric input JSON (to find INFLOW vessel)')
    parser.add_argument('--vessel-name', help='Vessel name for INFLOW (default: auto-detect)')
    
    args = parser.parse_args()
    
    plot_inflow_comparison(
        args.csv1,
        args.csv2,
        output_path=args.output,
        geometric_input_path=args.geometric_input,
        vessel_name=args.vessel_name
    )

if __name__ == '__main__':
    main()

