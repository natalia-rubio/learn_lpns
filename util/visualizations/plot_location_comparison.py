#!/usr/bin/env python3
"""
Unified script to plot pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models
at any location in the network.

Location format: "source:target" where:
  - "INFLOW:branch0_seg0" - inlet BC to vessel
  - "J0:branch1_seg0" - junction to vessel (inlet of vessel)
  - "branch0_seg0:J0" - vessel to junction (outlet of vessel)
  - "branch5_seg0:RESISTANCE_0" - vessel to outlet BC (terminal outlet)
"""

import glob
import os
import sys
import argparse
import json
import csv
import numpy as np
import warnings

from util.visualizations.cv_pressure_errors_to_latex import (
    format_modality_display_for_legend,
    MODALITY_DISPLAY,
)

# Suppress matplotlib warnings about redundant linestyle
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*linestyle.*redundantly defined.*')
warnings.filterwarnings('ignore', message='.*linestyle.*keyword argument.*')

# Check for matplotlib
HAS_MATPLOTLIB = False
plt = None
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Configure LaTeX rendering with Computer Modern font
    try:
        plt.rcParams['text.usetex'] = True
        # Test if LaTeX is available
        test_fig, test_ax = plt.subplots(figsize=(1, 1))
        test_ax.text(0.5, 0.5, r'Test $\alpha$')
        plt.close(test_fig)
        # If successful, configure LaTeX settings
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'cm'
        LATEX_AVAILABLE = True
    except Exception:
        # LaTeX not available, use mathtext with Computer Modern
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'cm'
        LATEX_AVAILABLE = False
    
    # Set font sizes
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['figure.titlesize'] = 36
    
    HAS_MATPLOTLIB = True
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# =============================================================================
# LINE STYLE CONFIGURATION
# =============================================================================
# Define colors and styles for different data sources here.
# Each source has: 'color', 'linestyle', 'linewidth', 'label'

# Colors for each junction type (same color for original and bifurcations variants)
JUNCTION_COLORS = {
    'NORMAL_JUNCTION': 'red',
    'BloodVesselJunction': 'mediumpurple',
    'DirIndepJunction': 'dodgerblue',
    'HybridJunction': 'violet',
    'BloodVesselJunction_NN': 'dodgerblue',
    'BloodVesselJunction_NN_plus_Vessel_NN': 'limegreen',
}

# Line styles for geometry variants
GEOMETRY_LINESTYLES = {
    'original': '--',      # Dashed for original geometry
    'bifurcations': ':',   # Dotted for bifurcations-only geometry
}

LINE_STYLES = {
    # Reference data (3D model)
    '3d_model': {
        'color': 'black',
        'linestyle': '-',
        'linewidth': 8,
        'label': '3D_Solution',
        'alpha': 0.75,
    },
    # Geometric 0D (uncalibrated) - original geometry
    'geometric_0d': {
        'color': 'indianred',
        'linestyle': '--',
        'linewidth': 4,
        'label': '0D Poiseuille',
        'alpha': 0.75,
    },
    # Geometric 0D (uncalibrated) - bifurcations geometry
    'bifurcations_geometric_0d': {
        'color': 'indianred',
        'linestyle': '-',
        'linewidth': 4,
        'label': '0D Poiseuille',
        'alpha': 0.75,
    },
    # Geometric 0D (uncalibrated) - bifurcations geometry
    'bifurcations_EL_geometric_0d': {
        'color': 'indianred',
        'linestyle': '--',
        'linewidth': 5,
        'label': '0D Poiseuille',
        'alpha': 0.75,
    },
    # Empirical stenosis-off forward (same network, stenosis coefficients set to zero)
    'stenosis_zero': {
        'color': '#c62828',
        'linestyle': '-',
        'linewidth': 4,
        'label': '0D (stenosis=0)',
        'alpha': 0.85,
    },
    # Original geometry calibrated results
    'original_NORMAL_JUNCTION': {
        'color': 'red',
        'linestyle': '--',
        'linewidth': 4,
        'label': '0D $\Delta P = 0$ Junction (Calibrated) (orig)',
        'alpha': 0.75,
    },
    'original_BloodVesselJunction': {
        'color': 'mediumpurple',
        'linestyle': '--',
        'linewidth': 4,
        'label': '0D RRI Junction (Calibrated) (orig)',
        'alpha': 0.75,
    },
    'original_DirIndepJunction': {
        'color': 'dodgerblue',
        'linestyle': '--',
        'linewidth': 4,
        'label': 'Dir-Indep Junction (orig)',
        'alpha': 0.75,
    },
    'original_HybridJunction': {
        'color': 'violet',
        'linestyle': '--',
        'linewidth': 4,
        'label': 'Hybrid Junction (orig)',
        'alpha': 0.75,
    },
    # Bifurcations geometry calibrated results
    'bifurcations_NORMAL_JUNCTION': {
        'color': 'red',
        'linestyle': ':',
        'linewidth': 4,
        'label': '0D $\Delta P = 0$ Junction (Calibrated) (bif)',
        'alpha': 0.75,
    },
    'bifurcations_BloodVesselJunction': {
        'color': 'mediumpurple',
        'linestyle': ':',
        'linewidth': 4,
        'label': '0D RRI Junction (Calibrated) (bif)',
        'alpha': 0.75,
    },
    'bifurcations_DirIndepJunction': {
        'color': 'dodgerblue',
        'linestyle': ':',
        'linewidth': 4,
        'label': 'Dir-Indep Junction (bif)',
        'alpha': 0.75,
    },
    'bifurcations_HybridJunction': {
        'color': 'violet',
        'linestyle': ':',
        'linewidth': 4,
        'label': 'Hybrid Junction (bif)',
        'alpha': 0.75,
    },
    # Legacy support for old naming (without geometry prefix)
    'NORMAL_JUNCTION': {
        'color': 'red',
        'linestyle': '--',
        'linewidth': 4,
        'label': '0D $\Delta P = 0$ Junction (Calibrated)',
        'alpha': 0.75,
    },
    'BloodVesselJunction': {
        'color': 'mediumpurple',
        'linestyle': '--',
        'linewidth': 4,
        'label': '0D RRI Junction (Calibrated)',
        'alpha': 0.75,
    },
    'DirIndepJunction': {
        'color': 'dodgerblue',
        'linestyle': '--',
        'linewidth': 4,
        'label': 'Dir-Indep Junction',
        'alpha': 0.75,
    },
    'HybridJunction': {
        'color': 'violet',
        'linestyle': '--',
        'linewidth': 4,
        'label': 'Hybrid Junction',
        'alpha': 0.75,
    },
    # NN-modified BloodVesselJunction (always dodger blue, dashed)
    'bifurcations_BloodVesselJunction_NN': {
        'color': 'dodgerblue',
        'linestyle': '--',
        'linewidth': 4,
        'label': 'Learned Junctions',
        'alpha': 0.75,
    },
    'original_BloodVesselJunction_NN': {
        'color': 'dodgerblue',
        'linestyle': '--',
        'linewidth': 4,
        'label': 'Learned Junctions',
        'alpha': 0.75,
    },
    'BloodVesselJunction_NN': {
        'color': 'dodgerblue',
        'linestyle': '--',
        'linewidth': 4,
        'label': 'Learned Junctions',
        'alpha': 0.75,
    },
    'BloodVesselJunction_NN_plus_Vessel_NN': {
        'color': 'limegreen',
        'linestyle': '--',
        'linewidth': 4,
        'label': 'Learned Junctions and Vessels',
        'alpha': 0.75,
    },
    'NN_vessel': {
        'color': 'gold',
        'linestyle': '-.',
        'linewidth': 4,
        'label': 'Learned Vessels',
        'alpha': 0.75,
    },
}


def _modality_key_from_plot_key(key):
    """Map plot/csv style key to MODALITY_DISPLAY key, or None if no shared bar-chart name."""
    if key == '3d_model':
        return '3d_model'
    if key in ('geometric_0d', 'bifurcations_geometric_0d', 'bifurcations_EL_geometric_0d'):
        return 'geometric'
    if 'BloodVesselJunction_NN_plus_Vessel_NN' in key:
        return 'BloodVesselJunction_NN_plus_Vessel_NN'
    if 'NN_vessel' in key:
        return 'NN_vessel'
    if 'BloodVesselJunction_NN' in key:
        return 'BloodVesselJunction_NN'
    if 'NORMAL_JUNCTION' in key:
        return 'NORMAL_JUNCTION'
    if 'BloodVesselJunction' in key:
        return 'BloodVesselJunction'
    if key.endswith('_NN'):
        return 'BloodVesselJunction_NN'
    return None


def _geometry_variant_suffix_lines(key, modality_key):
    """Optional extra legend line when multiple calibrated curves share one modality name."""
    if modality_key == 'BloodVesselJunction':
        if key.startswith('original_BloodVesselJunction'):
            return ['(orig)']
        if key.startswith('bifurcations_BloodVesselJunction') and not key.startswith('bifurcations_EL'):
            return ['(bif)']
        if key.startswith('bifurcations_EL_') and 'BloodVesselJunction' in key and 'NN' not in key:
            return ['(EL)']
    if modality_key == 'NORMAL_JUNCTION':
        if key.startswith('original_NORMAL_JUNCTION'):
            return ['(orig)']
        if key.startswith('bifurcations_NORMAL_JUNCTION') and not key.startswith('bifurcations_EL'):
            return ['(bif)']
        if key.startswith('bifurcations_EL_') and 'NORMAL_JUNCTION' in key:
            return ['(EL)']
    return None


def _legend_order_key(label):
    """
    Sort key (rank, label) for figure legend. Lower rank appears first.
    Matches modality order used with cv_pressure_max_pct_error_barchart / MODALITY_DISPLAY
    (3D, Poiseuille, Learned Vessels, Learned Junctions, Learned J+V, Optimal/Fit to 3D, then ΔP=0).
    """
    lines = [ln.strip() for ln in label.split('\n') if ln.strip()]
    if not lines:
        return (100, label)
    top = lines[0]
    if top == '3D':
        return (0, label)
    if top == 'Poiseuille':
        return (1, label)
    if top == 'Learned':
        if len(lines) >= 3 and 'and' in lines[2]:
            return (4, label)
        if len(lines) >= 2 and lines[1] == 'Vessels':
            return (2, label)
        if len(lines) >= 2 and lines[1] == 'Junctions':
            return (3, label)
        return (99, label)
    if top == 'Optimal':
        return (5, label)
    if '\\Delta' in label or (top.startswith('$') and 'Delta' in top):
        return (6, label)
    return (100, label)


def get_line_style(key):
    """
    Get line style for a given key. Handles dynamic generation for 
    {geometry}_{junction_type} patterns not explicitly defined.
    
    Args:
        key: Style key like 'original_NORMAL_JUNCTION' or 'bifurcations_BloodVesselJunction'
    
    Returns:
        Dictionary with color, linestyle, linewidth, label, alpha
    """
    if key in LINE_STYLES:
        base = LINE_STYLES[key]
    elif 'NN_vessel' in key:
        base = LINE_STYLES['NN_vessel']
    elif 'BloodVesselJunction_NN' in key or key.endswith('_NN'):
        base = {
            'color': 'dodgerblue',
            'linestyle': '--',
            'linewidth': 4,
            'label': 'Learned Junctions',
            'alpha': 0.75,
        }
    else:
        parsed = None
        for geometry in ['original', 'bifurcations']:
            if key.startswith(f'{geometry}_'):
                junction_type = key[len(f'{geometry}_'):]
                color = JUNCTION_COLORS.get(junction_type, 'gray')
                linestyle = GEOMETRY_LINESTYLES.get(geometry, '-')
                suffix = '(orig)' if geometry == 'original' else '(bif)'
                label = f'{junction_type} {suffix}'
                parsed = {
                    'color': color,
                    'linestyle': linestyle,
                    'linewidth': 4,
                    'label': label,
                    'alpha': 0.75,
                }
                break
        if parsed is not None:
            base = parsed
        else:
            base = {
                'color': 'gray',
                'linestyle': '-',
                'linewidth': 4,
                'label': key,
                'alpha': 0.75,
            }

    out = dict(base)
    mod = _modality_key_from_plot_key(key)
    if mod is not None and mod in MODALITY_DISPLAY:
        extra = _geometry_variant_suffix_lines(key, mod)
        lab = format_modality_display_for_legend(mod, extra)
        if lab is not None:
            out['label'] = lab
    return out
# =============================================================================


def parse_location(location):
    """
    Parse a location string into its components.
    
    Args:
        location: Location string like "INFLOW:branch0_seg0" or "branch0_seg0:J0"
    
    Returns:
        tuple: (source, target, vessel_name, is_inlet, location_type)
            - source: First part of location (e.g., "INFLOW", "J0", "branch0_seg0")
            - target: Second part of location (e.g., "branch0_seg0", "J0", "RESISTANCE_0")
            - vessel_name: The vessel involved
            - is_inlet: True if this is the inlet of the vessel, False if outlet
            - location_type: 'bc_inlet', 'junction_inlet', 'junction_outlet', 'bc_outlet'
    """
    parts = location.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid location format: {location}. Expected 'source:target'")
    
    source, target = parts
    
    # Determine location type based on source and target
    if source == 'INFLOW':
        # BC inlet: INFLOW:branch0_seg0
        return source, target, target, True, 'bc_inlet'
    elif source.startswith('J'):
        # Junction inlet: J0:branch1_seg0
        return source, target, target, True, 'junction_inlet'
    elif target.startswith('J'):
        # Junction outlet: branch0_seg0:J0
        return source, target, source, False, 'junction_outlet'
    elif target.startswith('RESISTANCE') or target.startswith('RCR') or target.startswith('CORONARY'):
        # BC outlet: branch5_seg0:RESISTANCE_0
        return source, target, source, False, 'bc_outlet'
    else:
        # Unknown pattern - try to infer
        if source.startswith('branch'):
            return source, target, source, False, 'outlet'
        else:
            return source, target, target, True, 'inlet'


def read_zerod_csv(csv_path):
    """
    Read 0D simulation results from CSV.
    Handles both 'location' and 'name' as the vessel identifier column.
    
    Returns:
        results: Dictionary {location: {time: {field: value}}}
        times: Sorted list of time values
    """
    results = {}
    times = set()
    
    if not os.path.exists(csv_path):
        return results, sorted(times)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Check which column name is used for vessel identifier
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return results, sorted(times)
        
        vessel_col = None
        if 'location' in fieldnames:
            vessel_col = 'location'
        elif 'name' in fieldnames:
            vessel_col = 'name'
        else:
            return results, sorted(times)
        
        for row in reader:
            location = row[vessel_col]
            time = float(row['time'])
            times.add(time)
            
            if location not in results:
                results[location] = {}
            if time not in results[location]:
                results[location][time] = {}
            
            # Extract all numeric fields
            for key, value in row.items():
                if key not in [vessel_col, 'time']:
                    try:
                        results[location][time][key] = float(value)
                    except (ValueError, TypeError):
                        continue
    
    return results, sorted(times)


def build_vessel_name_mapping(bifurcations_geometric_input_path, el_geometric_input_path):
    """
    Build a mapping from bifurcations vessel names to EL-adjusted vessel names.
    
    This handles cases where vessels were merged (e.g., branch1_seg0 -> branch1_seg0_1_2)
    or converted to connectors (e.g., branch3_seg0 -> branch3_seg0_connector).
    
    Args:
        bifurcations_geometric_input_path: Path to bifurcations geometric input
        el_geometric_input_path: Path to EL-adjusted geometric input
    
    Returns:
        Dictionary mapping bifurcations vessel names to EL vessel names
    """
    import json
    
    if not os.path.exists(bifurcations_geometric_input_path) or not os.path.exists(el_geometric_input_path):
        return {}
    
    with open(bifurcations_geometric_input_path, 'r') as f:
        bif_data = json.load(f)
    with open(el_geometric_input_path, 'r') as f:
        el_data = json.load(f)
    
    bif_vessels = {v['vessel_name']: v for v in bif_data.get('vessels', [])}
    el_vessels = {v['vessel_name']: v for v in el_data.get('vessels', [])}
    
    # Build mapping: for each bifurcations vessel, find corresponding EL vessel
    # Strategy: match by centerline_node_ids if available, otherwise by name patterns
    mapping = {}
    
    # First, try to match by centerline_node_ids (most reliable)
    el_vessels_by_inlet = {}
    el_vessels_by_outlet = {}
    for el_name, el_vessel in el_vessels.items():
        if 'centerline_node_ids' in el_vessel:
            inlet_gid = el_vessel['centerline_node_ids'].get('inlet')
            outlet_gid = el_vessel['centerline_node_ids'].get('outlet')
            if inlet_gid is not None:
                el_vessels_by_inlet[inlet_gid] = el_name
            if outlet_gid is not None:
                el_vessels_by_outlet[outlet_gid] = el_name
    
    for bif_name, bif_vessel in bif_vessels.items():
        if 'centerline_node_ids' in bif_vessel:
            bif_inlet_gid = bif_vessel['centerline_node_ids'].get('inlet')
            bif_outlet_gid = bif_vessel['centerline_node_ids'].get('outlet')
            
            # Try to match by inlet GID
            if bif_inlet_gid is not None and bif_inlet_gid in el_vessels_by_inlet:
                el_name = el_vessels_by_inlet[bif_inlet_gid]
                mapping[bif_name] = el_name
                continue
            
            # Try to match by outlet GID
            if bif_outlet_gid is not None and bif_outlet_gid in el_vessels_by_outlet:
                el_name = el_vessels_by_outlet[bif_outlet_gid]
                mapping[bif_name] = el_name
                continue
        
        # Fallback: try to match by name (exact match or merged pattern)
        if bif_name in el_vessels:
            mapping[bif_name] = bif_name
        else:
            # Check if it's part of a merged vessel name (e.g., branch1_seg0 is part of branch1_seg0_1_2)
            for el_name in el_vessels.keys():
                if el_name.startswith(bif_name + '_') or el_name == bif_name:
                    mapping[bif_name] = el_name
                    break
    
    return mapping


def extract_data_from_csv(csv_path, vessel_name, is_inlet, vessel_name_mapping=None):
    """
    Extract pressure and flow data from 0D CSV results for a specific vessel.
    
    Args:
        csv_path: Path to CSV file
        vessel_name: Name of the vessel
        is_inlet: If True, extract inlet data; if False, extract outlet data
        vessel_name_mapping: Optional dictionary mapping vessel names (for EL-adjusted geometry)
    
    Returns:
        times: Time array
        pressures: Pressure values (in dynes/cm^2)
        flows: Flow values (in cm³/s)
    """
    results, times = read_zerod_csv(csv_path)
    
    # Try original vessel name first
    mapped_vessel_name = vessel_name
    if vessel_name_mapping and vessel_name in vessel_name_mapping:
        mapped_vessel_name = vessel_name_mapping[vessel_name]
    
    # Try mapped name, then original name, then try to find partial matches
    vessel_names_to_try = [mapped_vessel_name, vessel_name]
    if vessel_name_mapping:
        # Also try reverse lookup: if mapped name contains original name
        for mapped_name in vessel_name_mapping.values():
            if vessel_name in mapped_name and mapped_name not in vessel_names_to_try:
                vessel_names_to_try.append(mapped_name)
    
    for try_name in vessel_names_to_try:
        if try_name in results:
            vessel_name = try_name
            break
    else:
        return None, None, None
    
    # Determine which fields to extract
    if is_inlet:
        pressure_field = 'pressure_in'
        flow_field = 'flow_in'
    else:
        pressure_field = 'pressure_out'
        flow_field = 'flow_out'
    
    pressures = []
    flows = []
    times_list = []
    
    for time in times:
        if time in results[vessel_name]:
            data = results[vessel_name][time]
            if pressure_field in data:
                pressures.append(data[pressure_field])
                times_list.append(time)
                if flow_field in data:
                    flows.append(data[flow_field])
                else:
                    flows.append(None)
            elif flow_field in data:
                flows.append(data[flow_field])
                times_list.append(time)
                pressures.append(None)
    
    # Filter out None values
    if pressures and all(p is not None for p in pressures):
        pressures = np.array(pressures)
    else:
        pressures = None
    
    if flows and all(f is not None for f in flows):
        flows = np.array(flows)
    else:
        flows = None
    
    times_array = np.array(times_list) if times_list else None
    
    return times_array, pressures, flows


def extract_data_from_calibration_input(calibration_input_path, location):
    """
    Extract pressure and flow data from calibration input observations for a specific location.
    
    Args:
        calibration_input_path: Path to calibration input JSON
        location: Location string like "INFLOW:branch0_seg0" or "branch0_seg0:J0"
    
    Returns:
        times: Time array (normalized [0, 1])
        pressures: Pressure values (in dynes/cm^2)
        flows: Flow values (in cm³/s)
    """
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    if 'y' not in calib_data:
        return None, None, None
    
    observations = calib_data['y']
    
    # Build observation keys from location
    pressure_key = f'pressure:{location}'
    flow_key = f'flow:{location}'
    
    if pressure_key not in observations:
        return None, None, None
    
    # Get number of observations
    num_obs = len(observations[pressure_key])
    times = np.linspace(0.0, 1.0, num_obs)
    
    # Extract pressure (in dynes/cm^2)
    pressures = np.array(observations[pressure_key])
    
    # Extract flow
    if flow_key in observations:
        flows = np.array(observations[flow_key])
    else:
        flows = None
    
    return times, pressures, flows


def get_all_locations_from_calibration_input(calibration_input_path):
    """
    Get all unique locations from calibration input observations.
    
    Returns:
        List of location strings (e.g., ["INFLOW:branch0_seg0", "branch0_seg0:J0", ...])
    """
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    if 'y' not in calib_data:
        return []
    
    observations = calib_data['y']
    locations = set()
    
    for key in observations.keys():
        if key.startswith('pressure:'):
            # Extract location from "pressure:source:target"
            parts = key.split(':')
            if len(parts) == 3:
                location = f'{parts[1]}:{parts[2]}'
                locations.add(location)
    
    return sorted(locations)


def _find_mse_comparison_csv(calibration_input_path):
    """
    Path to *_{mse_comparison}.csv next to calibration input (same file used for CV bar charts).
    """
    if not calibration_input_path:
        return None
    d = os.path.dirname(os.path.abspath(calibration_input_path))
    bn = os.path.basename(calibration_input_path)
    idx = bn.find("_calibration_input")
    if idx > 0:
        prefix = bn[:idx]
        p = os.path.join(d, f"{prefix}_mse_comparison.csv")
        if os.path.exists(p):
            return p
    cands = glob.glob(os.path.join(d, "*_mse_comparison.csv"))
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    for c in sorted(cands):
        if "bifurcations_EL" in os.path.basename(c):
            return c
    return sorted(cands)[0]


def _load_mean_pressure_max_rel_pct_from_mse_csv(path):
    """
    Parse Summary Statistics row 'Mean Pressure Max Rel Error' (fractions 0–1) into
    modality column name -> percent (0–100). Matches bifurcations_EL_mse_comparison.csv.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row:
                continue
            if row[0] == "Metric":
                header = row
            elif header and row[0] == "Mean Pressure Max Rel Error":
                out = {}
                for j, name in enumerate(header[1:], 1):
                    if j < len(row):
                        try:
                            out[name] = float(row[j]) * 100.0
                        except ValueError:
                            pass
                return out if out else None
    return None


def _slice_results_dict_for_window(results_dict, t_lo, t_hi):
    """Copy of results_dict with each series limited to times in [t_lo, t_hi]."""
    out = {}
    for k, d in results_dict.items():
        if d.get('times') is None or len(d['times']) == 0:
            continue
        t = np.asarray(d['times'], dtype=float)
        mask = (t >= t_lo) & (t <= t_hi)
        if not np.any(mask):
            continue
        out[k] = {
            'times': t[mask],
            'pressures': np.asarray(d['pressures'])[mask] if d.get('pressures') is not None else None,
            'flows': np.asarray(d['flows'])[mask] if d.get('flows') is not None else None,
        }
    return out


def _fill_rel_pct_from_mse_summary(rel_pct_by_plot_key, mse_pct_by_modality, geometric_results, calibrated_results):
    """Map MSE CSV modality columns to plot keys (same names as _modality_key_from_plot_key)."""
    for geo_key in geometric_results:
        mod = _modality_key_from_plot_key(geo_key)
        if mod == "geometric" and "geometric" in mse_pct_by_modality:
            rel_pct_by_plot_key[geo_key] = mse_pct_by_modality["geometric"]
    for jtype in calibrated_results:
        if "NORMAL_JUNCTION" in jtype:
            continue
        mod = _modality_key_from_plot_key(jtype)
        if mod and mod in mse_pct_by_modality:
            rel_pct_by_plot_key[jtype] = mse_pct_by_modality[mod]


def get_time_period(set_name, geo_name):
    """
    Try to get the actual time period from 3D simulation XML.
    Returns time period in seconds, or None if not found.
    """
    xml_paths = [
        os.path.join('data', 'threeD', set_name, geo_name, 'fluid_simulation_0-0.xml'),
        os.path.join('data', 'threeD', set_name, geo_name, 'solver.inp'),
    ]
    
    for xml_path in xml_paths:
        if os.path.exists(xml_path):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                gen_params = root.find('General_Parameters')
                if gen_params is None:
                    gen_params = root.find('GeneralSimulationParameters')
                
                if gen_params is not None:
                    num_time_steps_elem = gen_params.find('Number_of_time_steps')
                    time_step_size_elem = gen_params.find('Time_step_size')
                    
                    if num_time_steps_elem is not None and time_step_size_elem is not None:
                        num_time_steps = int(num_time_steps_elem.text)
                        time_step_size = float(time_step_size_elem.text)
                        time_period = num_time_steps * time_step_size
                        return time_period
            except Exception:
                pass
    
    return None


def plot_location_comparison(calibration_input_path, geometric_csv_path, calibrated_csv_paths,
                             location, output_path, set_name=None, geo_name=None, time_period=None,
                             geometric_input_path=None, zoom_start_idx=None, zoom_end_idx=None, 
                             verbose=False, geometric_csv_paths=None, vessel_name_mapping=None):
    """
    Plot pressure and flow comparison between 3D, geometric 0D, and calibrated 0D models
    at a specific location.
    
    Args:
        calibration_input_path: Path to calibration input JSON (for 3D observations)
        geometric_csv_path: Path to geometric 0D results CSV (legacy, single file)
        calibrated_csv_paths: Dictionary mapping junction type names to CSV paths
        location: Location string (e.g., "INFLOW:branch0_seg0" or "branch0_seg0:J0")
        output_path: Path to save plot
        set_name: Set name (for finding time period and VMR zoom defaults)
        geo_name: Geometry name (for finding time period)
        time_period: Time period in seconds (if None, will try to find from XML)
        geometric_input_path: Path to geometric input JSON (optional)
        zoom_start_idx: Start index for zoom window
        zoom_end_idx: End index for zoom window
        verbose: If True, print detailed information
        geometric_csv_paths: Dictionary mapping geometry variant names to CSV paths
                            (e.g., {'original': '...csv', 'bifurcations': '...csv'})
                            If provided, overrides geometric_csv_path
    
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"Plotting comparison for location: {location}")
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        return False
    
    # Parse location
    try:
        source, target, vessel_name, is_inlet, location_type = parse_location(location)
    except ValueError as e:
        print(f"Error: {e}")
        return False
    
    if verbose:
        print(f"  Vessel: {vessel_name}, is_inlet: {is_inlet}, type: {location_type}")
    
    # Get time period
    if time_period is None and set_name is not None and geo_name is not None:
        time_period = get_time_period(set_name, geo_name)
    
    if time_period is None:
        if 'VMR' in set_name:
            time_period = 2.0  # Default for VMR
        else:
            time_period = 1.0
        if verbose:
            print(f"  Using default time period: {time_period} s")
    else:
        if verbose:
            print(f"  Time period: {time_period:.4f} s")
    
    # Extract geometric 0D results (support both single path and dictionary of paths)
    geometric_results = {}
    times_geo = None
    
    # Build dictionary of geometric CSV paths
    geo_csv_dict = {}
    if geometric_csv_paths and isinstance(geometric_csv_paths, dict):
        geo_csv_dict = geometric_csv_paths
    elif geometric_csv_path:
        # Legacy: single path treated as 'original'
        geo_csv_dict = {'original': geometric_csv_path}
    
    for geo_variant, csv_path in geo_csv_dict.items():
        if csv_path and os.path.exists(csv_path):
            times_var, pressures_var, flows_var = extract_data_from_csv(
                csv_path, vessel_name, is_inlet, vessel_name_mapping=vessel_name_mapping)
            if times_var is not None:
                pressures_var_mmhg = pressures_var / 1333.0 if pressures_var is not None else None
                # Use style key format: geometric_0d or bifurcations_geometric_0d
                if geo_variant == 'original':
                    style_key = 'geometric_0d'
                else:
                    style_key = f'{geo_variant}_geometric_0d'
                geometric_results[style_key] = {
                    'times': times_var,
                    'pressures': pressures_var_mmhg,
                    'flows': flows_var
                }
                if times_geo is None:
                    times_geo = times_var  # Use first one as reference
                if verbose:
                    print(f"  Found {len(times_var)} {geo_variant} geometric 0D time points")
    
    if times_geo is None:
        print(f"  Error: Could not extract any geometric 0D results for {vessel_name}")
        return False
    
    # For backward compatibility, also set these variables
    pressures_geo_mmhg = geometric_results.get('geometric_0d', {}).get('pressures')
    flows_geo = geometric_results.get('geometric_0d', {}).get('flows')
    
    # Extract 3D observations from calibration input
    times_3d_norm, pressures_3d, flows_3d = extract_data_from_calibration_input(
        calibration_input_path, location)
    
    if times_3d_norm is not None:
        # Interpolate 3D observations to match geometric 0D time points
        times_3d_sec = times_geo
        time_min = times_geo[0]
        time_max = times_geo[-1]
        times_3d_original = time_min + times_3d_norm * (time_max - time_min)
        
        if pressures_3d is not None:
            pressures_3d_interp = np.interp(times_geo, times_3d_original, pressures_3d)
            pressures_3d_mmhg = pressures_3d_interp / 1333.0
        else:
            pressures_3d_mmhg = None
        
        if flows_3d is not None:
            flows_3d = np.interp(times_geo, times_3d_original, flows_3d)
        
        if verbose:
            print(f"  Found {len(times_3d_norm)} 3D observation points")
    else:
        if verbose:
            print(f"  Warning: Could not extract 3D observations for {location}")
        pressures_3d_mmhg = None
        flows_3d = None
        times_3d_sec = times_geo
    
    # Extract calibrated 0D results
    calibrated_results = {}
    if calibrated_csv_paths and isinstance(calibrated_csv_paths, dict):
        for jtype, csv_path in calibrated_csv_paths.items():
            if csv_path and os.path.exists(csv_path):
                times_cal, pressures_cal, flows_cal = extract_data_from_csv(
                    csv_path, vessel_name, is_inlet, vessel_name_mapping=vessel_name_mapping)
                if times_cal is not None:
                    pressures_cal_mmhg = pressures_cal / 1333.0 if pressures_cal is not None else None
                    calibrated_results[jtype] = {
                        'times': times_cal,
                        'pressures': pressures_cal_mmhg,
                        'flows': flows_cal
                    }
                    if verbose:
                        print(f"  {jtype}: Found {len(times_cal)} time points")
    
    # Create plot with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
    axes[0].sharex(axes[1])
    axes[2].sharex(axes[3])
    axes[1].tick_params(labelbottom=True)
    axes[3].tick_params(labelbottom=True)
    
    # Get styles from configuration
    style_3d = get_line_style('3d_model')
    
    # Determine zoom window - automatically set to last 20% of time period
    num_time_steps = len(times_geo)
    
    if zoom_start_idx is None or zoom_end_idx is None:
        # Calculate total time span
        time_start = times_geo[0]
        time_end = times_geo[-1]
        total_time_span = time_end - time_start
        
        # Last 20% of time period
        zoom_time_start = time_start + 0.6 * total_time_span
        zoom_time_end = time_start + 0.8 * total_time_span
        
        # Find indices corresponding to these times
        if zoom_start_idx is None:
            # Find index where time >= zoom_time_start
            zoom_start_idx = np.searchsorted(times_geo, zoom_time_start)
        if zoom_end_idx is None:
            # Find index where time >= zoom_time_end (or use last index)
            zoom_end_idx = min(np.searchsorted(times_geo, zoom_time_end, side='right'), num_time_steps)
    
    # Validate zoom window (end index exclusive, same convention as MSE post_processing)
    zoom_start_idx = max(0, min(zoom_start_idx, num_time_steps - 1))
    zoom_end_idx = min(int(zoom_end_idx), num_time_steps)
    zoom_end_idx = max(zoom_start_idx + 1, zoom_end_idx)

    if zoom_start_idx >= zoom_end_idx:
        if num_time_steps >= 2:
            if num_time_steps == 2:
                zoom_start_idx, zoom_end_idx = 1, 2
            else:
                zoom_start_idx, zoom_end_idx = num_time_steps - 1, num_time_steps
        else:
            raise ValueError(
                f"Zoom window invalid and fewer than 2 time steps ({num_time_steps}). "
                f"Got zoom_start_idx={zoom_start_idx}, zoom_end_idx={zoom_end_idx}"
            )
    zoom_times = times_geo[zoom_start_idx:zoom_end_idx]
    time_zoom_start = zoom_times[0] if len(zoom_times) > 0 else times_geo[0]
    time_zoom_end = zoom_times[-1] if len(zoom_times) > 0 else times_geo[-1]
    # Exact x-axis limits for zoom panels (first/last time sample in window; no extra margin)
    zoom_xlim_lo = float(times_geo[zoom_start_idx])
    zoom_xlim_hi = float(times_geo[zoom_end_idx - 1]) if zoom_end_idx > zoom_start_idx else zoom_xlim_lo

    # Legend max %: only when *_{mse_comparison}.csv exists (same Mean Pressure Max Rel Error as CV bar charts).
    rel_pct_by_plot_key = {}
    mse_csv = _find_mse_comparison_csv(calibration_input_path)
    mse_by_mod = _load_mean_pressure_max_rel_pct_from_mse_csv(mse_csv) if mse_csv else None
    if mse_by_mod:
        _fill_rel_pct_from_mse_summary(rel_pct_by_plot_key, mse_by_mod, geometric_results, calibrated_results)
    
    # Prepare zoomed data
    pressures_3d_zoom = pressures_3d_mmhg[zoom_start_idx:zoom_end_idx] if pressures_3d_mmhg is not None else None
    flows_3d_zoom = flows_3d[zoom_start_idx:zoom_end_idx] if flows_3d is not None else None
    
    # Prepare zoomed geometric results
    geometric_results_zoom = {}
    for geo_key, geo_data in geometric_results.items():
        if geo_data['times'] is not None and len(geo_data['times']) > 0:
            times_geo_arr = np.array(geo_data['times'])
            zoom_mask = (times_geo_arr >= time_zoom_start) & (times_geo_arr <= time_zoom_end)
            if np.any(zoom_mask):
                geometric_results_zoom[geo_key] = {
                    'times': times_geo_arr[zoom_mask],
                    'pressures': np.array(geo_data['pressures'])[zoom_mask] if geo_data['pressures'] is not None else None,
                    'flows': np.array(geo_data['flows'])[zoom_mask] if geo_data['flows'] is not None else None
                }
    
    calibrated_results_zoom = {}
    for jtype, data in calibrated_results.items():
        if data['times'] is not None and len(data['times']) > 0:
            times_cal = np.array(data['times'])
            zoom_mask = (times_cal >= time_zoom_start) & (times_cal <= time_zoom_end)
            if np.any(zoom_mask):
                calibrated_results_zoom[jtype] = {
                    'times': times_cal[zoom_mask],
                    'pressures': np.array(data['pressures'])[zoom_mask] if data['pressures'] is not None else None,
                    'flows': np.array(data['flows'])[zoom_mask] if data['flows'] is not None else None
                }
    
    # Helper functions for plotting
    def _label_with_rel_pct(base_label, plot_key):
        pct = rel_pct_by_plot_key.get(plot_key)
        if pct is None or not np.isfinite(pct):
            return base_label
        return f"{base_label}\nMPE: {pct:.1f}\%%"

    def plot_pressure_data(ax, times_data, pressures_3d_data, geometric_data_dict, 
                          calibrated_data_dict, set_ylim_from_3d=False):
        if pressures_3d_data is not None:
            ax.plot(times_data, pressures_3d_data, 
                   color=style_3d['color'], linestyle=style_3d['linestyle'],
                   linewidth=style_3d['linewidth'], label=style_3d['label'], 
                   alpha=style_3d['alpha'])
        
        # Plot geometric 0D results (can be multiple: original, bifurcations)
        for geo_key, geo_data in geometric_data_dict.items():
            if geo_data.get('pressures') is not None:
                style = get_line_style(geo_key)
                lab = _label_with_rel_pct(style['label'], geo_key)
                ax.plot(geo_data['times'], geo_data['pressures'], 
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], label=lab, 
                       alpha=style['alpha'])
        
        for jtype, data in calibrated_data_dict.items():
            if 'NORMAL_JUNCTION' in jtype:
                continue
            if data['pressures'] is not None:
                style = get_line_style(jtype)
                lab = _label_with_rel_pct(style['label'], jtype)
                ax.plot(data['times'], data['pressures'], 
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], label=lab, 
                       alpha=style['alpha'])
        
        # Set y-limits
        all_pressures = []
        if pressures_3d_data is not None:
            all_pressures.extend(pressures_3d_data)
        for geo_data in geometric_data_dict.values():
            if geo_data.get('pressures') is not None:
                all_pressures.extend(geo_data['pressures'])
        for jtype, data in calibrated_data_dict.items():
            if 'NORMAL_JUNCTION' in jtype:
                continue
            if data['pressures'] is not None:
                all_pressures.extend(data['pressures'])
        
        if all_pressures:
            pressure_min = np.min(all_pressures)
            pressure_max = np.max(all_pressures)
            pressure_range = pressure_max - pressure_min
            # if pressure_range > 0:
            #     ax.set_ylim(pressure_min - 0.1 * pressure_range, pressure_max + 0.1 * pressure_range)
    
    def plot_flow_data(ax, times_data, flows_3d_data, geometric_data_dict, 
                      calibrated_data_dict, set_ylim_from_3d=False):
        if flows_3d_data is not None:
            ax.plot(times_data, flows_3d_data, 
                   color=style_3d['color'], linestyle=style_3d['linestyle'],
                   linewidth=style_3d['linewidth'], label=style_3d['label'], 
                   alpha=style_3d['alpha'])
        
        # Plot geometric 0D results (can be multiple: original, bifurcations)
        for geo_key, geo_data in geometric_data_dict.items():
            if geo_data.get('flows') is not None:
                style = get_line_style(geo_key)
                lab = _label_with_rel_pct(style['label'], geo_key)
                ax.plot(geo_data['times'], geo_data['flows'], 
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], label=lab, 
                       alpha=style['alpha'])
        
        for jtype, data in calibrated_data_dict.items():
            if 'NORMAL_JUNCTION' in jtype:
                continue
            if data['flows'] is not None:
                style = get_line_style(jtype)
                lab = _label_with_rel_pct(style['label'], jtype)
                ax.plot(data['times'], data['flows'], 
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], label=lab, 
                       alpha=style['alpha'])
        
        # Set y-limits
        all_flows = []
        if flows_3d_data is not None:
            all_flows.extend(flows_3d_data)
        for geo_data in geometric_data_dict.values():
            if geo_data.get('flows') is not None:
                all_flows.extend(geo_data['flows'])
        for jtype, data in calibrated_data_dict.items():
            if 'NORMAL_JUNCTION' in jtype:
                continue
            if data['flows'] is not None:
                all_flows.extend(data['flows'])
        
        if all_flows:
            flow_min = np.min(all_flows)
            flow_max = np.max(all_flows)
            flow_range = flow_max - flow_min
            # if flow_range > 0:
            #     ax.set_ylim(flow_min - 0.1 * flow_range, flow_max + 0.1 * flow_range)
    
    # Top “full cycle” panels: only the last 80% of the cycle (drop first 20%). Zoom band & zoom rows unchanged.
    _t0 = float(times_geo[0])
    _t1 = float(times_geo[-1])
    _span = _t1 - _t0
    full_view_t0 = _t0 + 0.2 * _span
    full_view_t1 = _t1

    # Top row: plot only the last 80% of samples so autoscale / y-range uses that window (not full cycle).
    tgeo = np.asarray(times_geo, dtype=float)
    mask_top = (tgeo >= full_view_t0) & (tgeo <= full_view_t1)
    times_3d_top = np.asarray(times_3d_sec)[mask_top] if times_3d_sec is not None else None
    pressures_3d_top = np.asarray(pressures_3d_mmhg)[mask_top] if pressures_3d_mmhg is not None else None
    flows_3d_top = np.asarray(flows_3d)[mask_top] if flows_3d is not None else None
    geometric_results_top = _slice_results_dict_for_window(geometric_results, full_view_t0, full_view_t1)
    calibrated_results_top = _slice_results_dict_for_window(calibrated_results, full_view_t0, full_view_t1)
    
    # Row 1–2: full cycle (top) — x view = last 80% of cycle
    ax = axes[0]
    ax.axvspan(zoom_xlim_lo, zoom_xlim_hi, alpha=0.4, color='gray', zorder=0)
    plot_pressure_data(ax, times_3d_top, pressures_3d_top, geometric_results_top, 
                      calibrated_results_top, set_ylim_from_3d=False)
    ax.set_ylabel(r'Pressure (mmHg)', fontsize=24)
    ax.set_xlim(full_view_t0, full_view_t1)
    ax.margins(x=0)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    ax = axes[1]
    ax.axvspan(zoom_xlim_lo, zoom_xlim_hi, alpha=0.4, color='gray', zorder=0)
    plot_flow_data(ax, times_3d_top, flows_3d_top, geometric_results_top, 
                  calibrated_results_top, set_ylim_from_3d=False)
    ax.set_ylabel(r'Flow (cm$^3$/s)', fontsize=24)
    ax.set_xlim(full_view_t0, full_view_t1)
    ax.margins(x=0)
    ax.tick_params(axis='x', labelbottom=True, bottom=True, labelsize=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Row 3–4: zoom window (bottom); xlims match window exactly (no horizontal margin)
    ax = axes[2]
    plot_pressure_data(ax, zoom_times, pressures_3d_zoom, geometric_results_zoom, 
                      calibrated_results_zoom, set_ylim_from_3d=False)
    ax.set_ylabel(r'Pressure (mmHg)', fontsize=24)
    ax.set_xlim(zoom_xlim_lo, zoom_xlim_hi)
    ax.margins(x=0)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    ax = axes[3]
    plot_flow_data(ax, zoom_times, flows_3d_zoom, geometric_results_zoom, 
                   calibrated_results_zoom, set_ylim_from_3d=False)
    ax.set_xlabel(r'Time (s)', fontsize=24)
    ax.set_ylabel(r'Flow (cm$^3$/s)', fontsize=24)
    ax.set_xlim(zoom_xlim_lo, zoom_xlim_hi)
    ax.margins(x=0)
    ax.tick_params(axis='x', labelbottom=True, bottom=True, labelsize=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Create legend (exclude spurious entries like "-location INFLOW:branch0_seg0")
    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)

    legend_pairs = list(zip(unique_handles, unique_labels))
    legend_pairs.sort(key=lambda hl: _legend_order_key(hl[1]))
    unique_handles = [h for h, _ in legend_pairs]
    unique_labels = [lb for _, lb in legend_pairs]
    
    # Title
    title_text = location
    if location == "INFLOW:branch0_seg0":
        title_text = "3D vs 0D Solutions at Vasculature Inlet"
    
    plt.tight_layout(rect=[0, 0, 1, 0.76])
    fig.suptitle(title_text, fontsize=28, weight='bold', y=0.95)
    # Two-row legend: 3D centered on top; remaining modalities in one row below (same order as before).
    center_x = 0.5
    handles_3d, labels_3d = [], []
    handles_rest, labels_rest = [], []
    for h, lb in zip(unique_handles, unique_labels):
        if lb.split("\n")[0].strip() == "3D":
            handles_3d.append(h)
            labels_3d.append(lb)
        else:
            handles_rest.append(h)
            labels_rest.append(lb)

    if handles_3d and handles_rest:
        fig.legend(
            handles_3d,
            labels_3d,
            loc="lower center",
            bbox_to_anchor=(center_x, 0.85),
            bbox_transform=fig.transFigure,
            ncol=1,
            fontsize=22,
            frameon=False,
        )
        fig.legend(
            handles_rest,
            labels_rest,
            loc="upper center",
            bbox_to_anchor=(center_x, 0.87),
            bbox_transform=fig.transFigure,
            ncol=len(handles_rest),
            fontsize=22,
            frameon=False,
        )
    else:
        fig.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            bbox_to_anchor=(center_x, 0.62),
            bbox_transform=fig.transFigure,
            ncol=min(3, max(1, len(unique_handles))),
            fontsize=22,
            frameon=False,
        )
    
    # Ensure x-axis labels are visible
    for ax_idx in [1, 3]:
        axes[ax_idx].tick_params(axis='x', labelbottom=True, bottom=True, labelsize=20)
        locs = axes[ax_idx].xaxis.get_majorticklocs()
        if len(locs) > 0:
            axes[ax_idx].xaxis.set_ticks(locs)
            axes[ax_idx].xaxis.set_ticklabels([f'{loc:.2f}' for loc in locs], fontsize=20)
    
    # Save plot (PNG and PDF)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_path = os.path.splitext(output_path)[0]
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    other_ext = '.pdf' if not output_path.lower().endswith('.pdf') else '.png'
    plt.savefig(base_path + other_ext, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"✓ Plot saved to: {output_path} and {base_path}{other_ext}")
    
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot pressure and flow comparison at any location in the network"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_3, VMR)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_007, 0063_1001)')
    parser.add_argument('--location', help='Specific location to plot (e.g., "INFLOW:branch0_seg0" or "branch0_seg0:J0")')
    parser.add_argument('--calibration-input', help='Path to calibration input JSON (default: auto-detect)')
    parser.add_argument('--geometric-csv', help='Path to geometric 0D results CSV (default: auto-detect)')
    parser.add_argument('--junction-types', type=lambda s: [x.strip() for x in s.split(',') if x.strip()],
                       default='original_NORMAL_JUNCTION,bifurcations_NORMAL_JUNCTION,original_BloodVesselJunction,bifurcations_BloodVesselJunction',
                       help='Comma-separated junction types to plot (e.g., original_NORMAL_JUNCTION,bifurcations_BloodVesselJunction)')
    parser.add_argument('--run-config', default='base',
                        help='Run config name for output subfolder (e.g., base, stenosis_off)')
    parser.add_argument('--output-dir', default='results/location_comparison', 
                        help='Output directory for plots (run-config subfolder is appended)')
    parser.add_argument('--data-dir', default='data/zeroD', 
                        help='Data directory for input files')
    parser.add_argument('--geometry-variant', default=None,
                        help='Geometry variant (e.g. bifurcations_EL). When set, also load NN modalities from {variant}_NN_*_results.csv if present.')
    parser.add_argument('--time-period', type=float, default=None,
                        help='Time period in seconds (default: auto-detect)')
    parser.add_argument('--zoom-start', type=int, default=None,
                        help='Start index for zoom window')
    parser.add_argument('--zoom-end', type=int, default=None,
                        help='End index for zoom window')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required but not available.")
        sys.exit(1)
    
    if not args.junction_types:
        print("Error: No valid junction types (--junction-types must be a non-empty comma-separated list).")
        sys.exit(1)
    
    # Auto-detect file paths
    data_dir = os.path.join(args.data_dir, args.set_name, args.geo_name)
    
    calibration_input_path = args.calibration_input or os.path.join(data_dir, 'calibration_input.json')
    geometric_csv_path = args.geometric_csv or os.path.join(data_dir, 'geometric_results.csv')
    
    # Set up calibrated CSV paths
    calibrated_csv_paths = {}
    for jtype in args.junction_types:
        csv_path = os.path.join(data_dir, f'calibrated_results_{jtype}.csv')
        calibrated_csv_paths[jtype] = csv_path
    # Add NN modalities (Learned Junctions, Learned Junctions and Vessels, Learned Vessels) when geometry-variant is set
    if getattr(args, 'geometry_variant', None):
        for nn_jtype, nn_basename in [
            ('BloodVesselJunction_NN', f"{args.geometry_variant}_NN_BloodVesselJunction_results.csv"),
            ('BloodVesselJunction_NN_plus_Vessel_NN', f"{args.geometry_variant}_NN_JunctionAndVessel_results.csv"),
            ('NN_vessel', f"{args.geometry_variant}_NN_VesselOnly_results.csv"),
        ]:
            nn_path = os.path.join(data_dir, nn_basename)
            if os.path.exists(nn_path):
                calibrated_csv_paths[nn_jtype] = nn_path
    
    if not os.path.exists(calibration_input_path):
        print(f"Error: Calibration input file not found: {calibration_input_path}")
        sys.exit(1)
    
    if not os.path.exists(geometric_csv_path):
        print(f"Error: Geometric CSV file not found: {geometric_csv_path}")
        sys.exit(1)
    
    # Get list of locations
    if args.location:
        locations = [args.location]
    else:
        locations = get_all_locations_from_calibration_input(calibration_input_path)
        print(f"\nFound {len(locations)} locations")
        # Build ordered list of unique vessels as they appear in locations
        vessel_order = []
        for loc in locations:
            try:
                _, _, vname, _, _ = parse_location(loc)
            except Exception:
                vname = None
            if vname and vname not in vessel_order:
                vessel_order.append(vname)

        # If there are more than 10 unique vessels, limit to first 10 and filter locations
        if len(vessel_order) > 10:
            selected_vessels = vessel_order[:10]
            if args.verbose:
                print(f"  ⚠ More than 10 unique vessels ({len(vessel_order)}). Limiting to first 10 vessels for plotting: {selected_vessels}")
            # Keep only locations that involve the selected vessels
            filtered_locations = []
            for loc in locations:
                try:
                    _, _, vname, _, _ = parse_location(loc)
                except Exception:
                    vname = None
                if vname in selected_vessels:
                    filtered_locations.append(loc)
            locations = filtered_locations
            if args.verbose:
                print(f"  Using {len(locations)} locations after filtering to first 10 vessels")
    
    # Create output directory: <output_dir>/<run_config>/<set_name>/<geo_name>
    output_dir = os.path.join(args.output_dir, args.run_config, args.set_name, args.geo_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find geometric input path
    geometric_input_path = os.path.join(data_dir, 'geometric_input.json')
    if not os.path.exists(geometric_input_path):
        geometric_input_path = None
    
    # Plot for each location
    success_count = 0
    for location in locations:
        # Generate safe filename from location
        safe_location = location.replace(':', '_')
        output_path = os.path.join(output_dir, f"{safe_location}_comparison.png")
        
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Processing location: {location}")
            print(f"{'='*60}")
        
        success = plot_location_comparison(
            calibration_input_path, geometric_csv_path, calibrated_csv_paths,
            location, output_path, set_name=args.set_name, geo_name=args.geo_name,
            time_period=args.time_period, geometric_input_path=geometric_input_path,
            zoom_start_idx=args.zoom_start, zoom_end_idx=args.zoom_end,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: Created {success_count}/{len(locations)} location comparison plots")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    sys.exit(0 if success_count == len(locations) else 1)


if __name__ == '__main__':
    main()

