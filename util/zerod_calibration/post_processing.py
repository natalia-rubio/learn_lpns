import os
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from calibration_helpers import HAS_SCIPY_INTERP, get_time_period
HAS_SCIPY_INTERP = False
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


def downsample_csv_file_to_times(input_csv_path, output_csv_path, target_times, method='linear', verbose=False):
    """
    Read a 0D CSV file and write a new CSV downsampled/interpolated to the provided target_times.

    Args:
        input_csv_path: existing CSV with columns including 'time' and a vessel id column ('location' or 'name')
        output_csv_path: path to write the downsampled CSV
        target_times: iterable of target time values (floats)
        method: 'linear' or 'nearest'

    Returns:
        True on success, False on failure
    """
    if not os.path.exists(input_csv_path):
        if verbose:
            print(f"    ✗ Input CSV for downsampling not found: {input_csv_path}")
        return False

    try:
        # Read original CSV into memory
        with open(input_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                if verbose:
                    print(f"    ✗ Input CSV has no header: {input_csv_path}")
                return False

            vessel_col = 'location' if 'location' in fieldnames else ('name' if 'name' in fieldnames else None)
            if vessel_col is None:
                if verbose:
                    print(f"    ✗ Could not find 'location' or 'name' column in: {input_csv_path}")
                return False

            # Collect data by vessel
            data_by_vessel = {}
            numeric_fields = [fn for fn in fieldnames if fn not in [vessel_col, 'time']]

            for row in reader:
                vessel = row[vessel_col]
                try:
                    t = float(row['time'])
                except Exception:
                    continue

                if vessel not in data_by_vessel:
                    data_by_vessel[vessel] = {'times': [], 'fields': {f: [] for f in numeric_fields}}

                data_by_vessel[vessel]['times'].append(t)
                for f in numeric_fields:
                    try:
                        data_by_vessel[vessel]['fields'][f].append(float(row.get(f, 'nan')))
                    except Exception:
                        data_by_vessel[vessel]['fields'][f].append(np.nan)

        # Prepare output directory
        out_dir = os.path.dirname(output_csv_path)
        os.makedirs(out_dir, exist_ok=True)

        # Write interpolated CSV: for each vessel and each target time write a row
        with open(output_csv_path, 'w', newline='') as fout:
            writer_fieldnames = [vessel_col, 'time'] + numeric_fields
            writer = csv.DictWriter(fout, fieldnames=writer_fieldnames)
            writer.writeheader()

            for vessel, dat in data_by_vessel.items():
                times = np.array(dat['times'], dtype=float)
                if times.size == 0:
                    continue

                # Ensure sorting by time
                sort_idx = np.argsort(times)
                times = times[sort_idx]
                field_arrays = {}
                for f in numeric_fields:
                    arr = np.array(dat['fields'][f], dtype=float)
                    if arr.size == 0:
                        arr = np.full(times.shape, np.nan)
                    else:
                        arr = arr[sort_idx]
                    field_arrays[f] = arr

                # For each target time compute interpolated values
                for tt in target_times:
                    row = {vessel_col: vessel, 'time': float(tt)}
                    for f in numeric_fields:
                        arr = field_arrays[f]
                        if method == 'nearest':
                            # Find nearest index
                            idx = np.abs(times - tt).argmin()
                            val = float(arr[idx]) if idx < len(arr) else float(arr[-1])
                        else:
                            # Linear interpolation with numpy (handles edge values by clipping)
                            try:
                                val = float(np.interp(tt, times, arr))
                            except Exception:
                                val = float('nan')
                        row[f] = val
                    writer.writerow(row)

        if verbose:
            print(f"    ✓ Downsampled CSV written: {output_csv_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"    ✗ Exception while downsampling CSV {input_csv_path} -> {output_csv_path}: {e}")
        return False


def plot_junction_pressure_differences(calibration_input_path, geometric_input_path, output_dir, zoom_start_idx=None, zoom_end_idx=None, set_name=None, geo_name=None, verbose=False):
    """
    Plot pressure differences at junctions from 3D solution.
    For each junction, plots pressure difference (inlet - outlet) vs time, one line per outlet.
    
    Args:
        calibration_input_path: Path to calibration input JSON (contains 3D observations)
        geometric_input_path: Path to geometric input JSON (contains junction structure)
        output_dir: Directory to save plots
        zoom_start_idx: Start index for zoom window (default: 599)
        zoom_end_idx: End index for zoom window (default: 699)
        set_name: Set name (for extracting time period from XML)
        geo_name: Geometry name (for extracting time period from XML)
        verbose: If True, print detailed information
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("  ✗ matplotlib not available, skipping junction pressure difference plots")
        return
    
    # Read calibration input for 3D observations
    if not os.path.exists(calibration_input_path):
        if verbose:
            print(f"  ✗ Calibration input not found: {calibration_input_path}")
        return
    
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    # Get 3D observations
    if '_full_observations' in calib_data and 'y' in calib_data['_full_observations']:
        obs_3d = calib_data['_full_observations']['y']
    elif 'y' in calib_data:
        obs_3d = calib_data['y']
    else:
        if verbose:
            print("  ✗ No 3D observations found in calibration input")
        return
    
    # Read geometric input for junction structure
    if not os.path.exists(geometric_input_path):
        if verbose:
            print(f"  ✗ Geometric input not found: {geometric_input_path}")
        return
    
    with open(geometric_input_path, 'r') as f:
        geo_input = json.load(f)
    
    junctions = geo_input.get('junctions', [])
    if not junctions:
        if verbose:
            print("  ✗ No junctions found in geometric input")
        return
    
    # Create output directory
    junction_plots_dir = os.path.join(output_dir, 'junction_pressure_differences')
    os.makedirs(junction_plots_dir, exist_ok=True)
    
    # Extract time array from observations (use first available observation)
    num_obs = None
    for obs_key, obs_values in obs_3d.items():
        if isinstance(obs_values, list) and len(obs_values) > 0:
            num_obs = len(obs_values)
            break
    
    if num_obs is None:
        if verbose:
            print("  ✗ Could not determine number of observations")
        return
    
    # Get time period in seconds (for converting to actual time)
    time_period = None
    if set_name and geo_name:
        time_period = get_time_period(set_name, geo_name)
    
    if time_period is None:
        if verbose:
            print("  ⚠ Could not determine time period, using default 1.0 s")
        time_period = 1.0
    else:
        if verbose:
            print(f"  Time period: {time_period:.4f} s")
    
    # Create time array in seconds (same as inlet comparison plots)
    time_array = np.linspace(0.0, time_period, num_obs)
    time_label = 'Time (s)'
    
    # Set default zoom window if not provided
    max_length = len(time_array)
    if zoom_start_idx is None or zoom_end_idx is None:
        if set_name == 'VMR':
            # For VMR: zoom to 1-2 seconds
            zoom_start_time = 1.0
            zoom_end_time = 2.0
            # Find indices corresponding to these times
            dt = time_period / (num_obs - 1) if num_obs > 1 else 1.0
            zoom_start_idx = int(zoom_start_time / dt) if zoom_start_idx is None else zoom_start_idx
            zoom_end_idx = min(int(zoom_end_time / dt) + 1, max_length) if zoom_end_idx is None else zoom_end_idx
            if verbose:
                print(f"  VMR zoom window: {zoom_start_time:.1f}s to {zoom_end_time:.1f}s (indices {zoom_start_idx} to {zoom_end_idx})")
        else:
            # Default for other sets: indices 599-699
            if zoom_start_idx is None:
                zoom_start_idx = 599
            if zoom_end_idx is None:
                zoom_end_idx = 699
    
    # Validate and adjust zoom window
    if zoom_start_idx >= max_length:
        zoom_start_idx = max(0, max_length - 100)
        zoom_end_idx = max_length
    elif zoom_end_idx > max_length:
        zoom_end_idx = max_length
    
    if zoom_start_idx >= zoom_end_idx:
        zoom_start_idx = max(0, max_length - 100)
        zoom_end_idx = max_length
    
    # Process each junction
    plot_count = 0
    for junc in junctions:
        junc_name = junc.get('junction_name', '')
        if not junc_name:
            continue
        
        inlet_vessels = junc.get('inlet_vessels', [])
        outlet_vessels = junc.get('outlet_vessels', [])
        
        if not inlet_vessels or not outlet_vessels:
            if verbose:
                print(f"  ⚠ Skipping junction {junc_name}: missing inlet or outlet vessels")
            continue
        
        # Get inlet vessel (assuming single inlet for now)
        inlet_vessel_idx = inlet_vessels[0] if inlet_vessels else None
        if inlet_vessel_idx is None:
            continue
        
        # Find inlet vessel name
        vessels = geo_input.get('vessels', [])
        if inlet_vessel_idx >= len(vessels):
            continue
        
        inlet_vessel = vessels[inlet_vessel_idx]
        inlet_vessel_name = inlet_vessel.get('vessel_name', '')
        
        if not inlet_vessel_name:
            continue
        
        # Get inlet pressure at junction: "pressure:inlet_vessel_name:junction_name"
        inlet_key = f"pressure:{inlet_vessel_name}:{junc_name}"
        if inlet_key not in obs_3d:
            if verbose:
                print(f"  ⚠ Junction {junc_name}: inlet pressure observation not found ({inlet_key})")
            continue
        
        inlet_pressure = np.array(obs_3d[inlet_key])
        if len(inlet_pressure) != len(time_array):
            # Interpolate to match time array if needed
            if HAS_SCIPY_INTERP:
                from scipy.interpolate import interp1d
                inlet_times = np.arange(len(inlet_pressure))
                interp_func = interp1d(inlet_times, inlet_pressure, kind='linear', 
                                     bounds_error=False, fill_value=(inlet_pressure[0], inlet_pressure[-1]))
                inlet_pressure = interp_func(time_array)
            else:
                # Simple resampling
                indices = np.linspace(0, len(inlet_pressure) - 1, len(time_array)).astype(int)
                inlet_pressure = inlet_pressure[indices]
        
        # Get inlet flow at junction: "flow:inlet_vessel_name:junction_name"
        inlet_flow_key = f"flow:{inlet_vessel_name}:{junc_name}"
        inlet_flow = None
        if inlet_flow_key in obs_3d:
            inlet_flow = np.array(obs_3d[inlet_flow_key])
            if len(inlet_flow) != len(time_array):
                # Interpolate to match time array if needed
                if HAS_SCIPY_INTERP:
                    from scipy.interpolate import interp1d
                    inlet_flow_times = np.arange(len(inlet_flow))
                    interp_func = interp1d(inlet_flow_times, inlet_flow, kind='linear', 
                                         bounds_error=False, fill_value=(inlet_flow[0], inlet_flow[-1]))
                    inlet_flow = interp_func(time_array)
                else:
                    # Simple resampling
                    indices = np.linspace(0, len(inlet_flow) - 1, len(time_array)).astype(int)
                    inlet_flow = inlet_flow[indices]
        
        # Get outlet pressures and calculate differences
        outlet_pressures = {}
        outlet_vessel_names = []
        
        for outlet_vessel_idx in outlet_vessels:
            if outlet_vessel_idx >= len(vessels):
                continue
            
            outlet_vessel = vessels[outlet_vessel_idx]
            outlet_vessel_name = outlet_vessel.get('vessel_name', '')
            
            if not outlet_vessel_name:
                continue
            
            # Get outlet pressure at junction: "pressure:junction_name:outlet_vessel_name"
            outlet_key = f"pressure:{junc_name}:{outlet_vessel_name}"
            if outlet_key not in obs_3d:
                if verbose:
                    print(f"  ⚠ Junction {junc_name}: outlet pressure observation not found for {outlet_vessel_name} ({outlet_key})")
                continue
            
            outlet_pressure = np.array(obs_3d[outlet_key])
            if len(outlet_pressure) != len(time_array):
                # Interpolate to match time array if needed
                if HAS_SCIPY_INTERP:
                    from scipy.interpolate import interp1d
                    outlet_times = np.arange(len(outlet_pressure))
                    interp_func = interp1d(outlet_times, outlet_pressure, kind='linear', 
                                         bounds_error=False, fill_value=(outlet_pressure[0], outlet_pressure[-1]))
                    outlet_pressure = interp_func(time_array)
                else:
                    # Simple resampling
                    indices = np.linspace(0, len(outlet_pressure) - 1, len(time_array)).astype(int)
                    outlet_pressure = outlet_pressure[indices]
            
            # Calculate pressure difference: inlet - outlet (convert to mmHg by dividing by 1333)
            pressure_diff = (inlet_pressure - outlet_pressure) / 1333.0
            outlet_pressures[outlet_vessel_name] = pressure_diff
            outlet_vessel_names.append(outlet_vessel_name)
        
        if not outlet_pressures:
            if verbose:
                print(f"  ⚠ Junction {junc_name}: no valid outlet pressures found")
            continue
        
        # Create plot with two subplots (zoomed on top, full on bottom)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax1_zoom = axes[0]  # Top: zoomed view
        ax1_full = axes[1]   # Bottom: full view
        
        # Define colors: red, blue, green (cycle if more than 3 outlets)
        color_list = ['red', 'blue', 'green']
        
        # Extract zoom window data (use indices, not time values)
        zoom_mask = np.zeros(len(time_array), dtype=bool)
        zoom_mask[zoom_start_idx:zoom_end_idx] = True
        zoom_times = time_array[zoom_mask]
        
        # Plot zoomed view (top subplot)
        for i, (outlet_name, pressure_diff) in enumerate(outlet_pressures.items()):
            color = color_list[i % len(color_list)]
            zoom_pressure_diff = pressure_diff[zoom_mask]
            ax1_zoom.plot(zoom_times, zoom_pressure_diff, label=outlet_name, linewidth=2, color=color)
        
        ax1_zoom.set_ylabel('Pressure Difference (mmHg)', fontsize=20)
        ax1_zoom.set_title(f'Junction {junc_name}: Pressure Difference (Inlet - Outlets)', fontsize=20)
        ax1_zoom.legend(loc='best', fontsize=20)
        ax1_zoom.grid(True, alpha=0.3)
        ax1_zoom.tick_params(axis='both', labelsize=20)
        ax1_zoom.set_xlim(zoom_times[0], zoom_times[-1])
        
        # Add second y-axis for inlet flow in zoomed view (if available)
        if inlet_flow is not None:
            ax2_zoom = ax1_zoom.twinx()
            zoom_inlet_flow = inlet_flow[zoom_mask]
            ax2_zoom.plot(zoom_times, zoom_inlet_flow, label='Inlet Flow', linewidth=2, color='black', linestyle='--')
            ax2_zoom.set_ylabel('Flow (cm³/s)', fontsize=20, color='black')
            ax2_zoom.tick_params(axis='y', labelsize=20, labelcolor='black')
            ax2_zoom.legend(loc='upper center', fontsize=20)
        
        # Plot full view (bottom subplot)
        for i, (outlet_name, pressure_diff) in enumerate(outlet_pressures.items()):
            color = color_list[i % len(color_list)]
            ax1_full.plot(time_array, pressure_diff, label=outlet_name, linewidth=2, color=color)
        
        ax1_full.set_xlabel(time_label, fontsize=20)
        ax1_full.set_ylabel('Pressure Difference (mmHg)', fontsize=20)
        ax1_full.grid(True, alpha=0.3)
        ax1_full.tick_params(axis='both', labelsize=20)
        ax1_full.set_xlim(time_array[0], time_array[-1])
        
        # Add shaded region to indicate zoom window (use time values, not indices)
        zoom_time_start = time_array[zoom_start_idx]
        zoom_time_end = time_array[min(zoom_end_idx, len(time_array)-1)]
        ax1_full.axvspan(zoom_time_start, zoom_time_end, alpha=0.3, color='gray', label='Zoom region')
        
        # Add second y-axis for inlet flow in full view (if available)
        if inlet_flow is not None:
            ax2_full = ax1_full.twinx()
            ax2_full.plot(time_array, inlet_flow, label='Inlet Flow', linewidth=2, color='black', linestyle='--')
            ax2_full.set_ylabel('Flow (cm³/s)', fontsize=20, color='black')
            ax2_full.tick_params(axis='y', labelsize=20, labelcolor='black')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{junc_name}_pressure_difference.png"
        plot_path = os.path.join(junction_plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        if verbose:
            print(f"  ✓ Created plot for junction {junc_name}: {plot_path}")
    
    if verbose:
        print(f"  Created {plot_count} junction pressure difference plots in: {junction_plots_dir}")
    else:
        print(f"  Created {plot_count} junction pressure difference plots")


def calculate_mse_between_3d_and_0d(calibration_input_path, csv_results_dict, geometric_input_path=None, zoom_start_idx=None, zoom_end_idx=None, output_csv_path=None, verbose=False, set_name=None, downsample_0d=True, downsample_method='linear', downsample_save_dir=None):
    """
    Calculate and print Mean Squared Error (MSE) between 3D observations and 0D solutions.
    Optionally saves results to a CSV file.
    
    Args:
        calibration_input_path: Path to calibration input JSON (contains 3D observations)
        csv_results_dict: Dictionary mapping modality names to CSV file paths
                         e.g., {'geometric': 'path/to/geometric_results.csv',
                                'NORMAL_JUNCTION': 'path/to/normal_junction_results.csv', ...}
        geometric_input_path: Optional path to geometric input JSON (for understanding structure)
        zoom_start_idx: Start index for zoom window (default: 599, or 1-2s for VMR)
        zoom_end_idx: End index for zoom window (default: 699, or 1-2s for VMR)
        output_csv_path: Optional path to save MSE results CSV (default: auto-generate based on calibration_input_path)
        verbose: If True, print detailed comparison table. If False, only print summary.
        set_name: Optional set name (e.g., 'VMR') for set-specific defaults
    
    Returns:
        Dictionary mapping modality names to MSE results
    """
    print("\n" + "="*80)
    print("Calculating Mean Squared Error (MSE) between 3D and 0D solutions")
    print("="*80)
    
    # Read 3D observations from calibration input
    # Report downsampling mode
    if downsample_0d:
        print("  Downsampling of 0D CSVs is ENABLED")
        if downsample_save_dir:
            print(f"  Downsampled CSVs will be written to: {downsample_save_dir}")
    else:
        print("  Downsampling of 0D CSVs is disabled")
    if not os.path.exists(calibration_input_path):
        print(f"  ✗ Calibration input not found: {calibration_input_path}")
        return {}
    
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    # Get 3D observations (use full observations if available, otherwise use y)
    if '_full_observations' in calib_data and 'y' in calib_data['_full_observations']:
        print(f"  Using observed_inflow_bc times for downsampling")
        obs_3d = calib_data['y']
    else:
        print("  ✗ No 3D observations found in calibration input")
        return {}
    print(f"  Using normalized time grid for downsampling")
    
    # Read geometric input to understand vessel/junction structure

    vessels = []
    junctions = []
    if geometric_input_path and os.path.exists(geometric_input_path):
        with open(geometric_input_path, 'r') as f:
            geo_input = json.load(f)
        vessels = geo_input.get('vessels', [])
        junctions = geo_input.get('junctions', [])
    
    # Set default zoom window if not provided (before entering the loop)
    # Find max_3d_length first for VMR time-based calculation
    max_3d_length = 0
    for obs_values_3d in obs_3d.values():
        if isinstance(obs_values_3d, list):
            max_3d_length = max(max_3d_length, len(obs_values_3d))
        elif hasattr(obs_values_3d, '__len__'):
            max_3d_length = max(max_3d_length, len(obs_values_3d))
    
    zoom_start_idx = int(max_3d_length/2); zoom_end_idx = int(max_3d_length)
    # if zoom_start_idx is None or zoom_end_idx is None:
    #     if set_name == 'VMR':
    #         # For VMR: zoom to 1-2 seconds
    #         # Assume period is 2.0s for VMR data (standard cardiac cycle)
    #         time_period = 2.0
    #         dt = time_period / (max_3d_length - 1) if max_3d_length > 1 else 1.0
    #         zoom_start_idx = int(1.0 / dt) if zoom_start_idx is None else zoom_start_idx
    #         zoom_end_idx = min(int(2.0 / dt) + 1, max_3d_length) if zoom_end_idx is None else zoom_end_idx
    #         print(f"  VMR zoom window: 1.0s to 2.0s (indices {zoom_start_idx} to {zoom_end_idx})")
    #     else:
    #         # Default for other sets: indices 599-699
    #         if zoom_start_idx is None:
    #             zoom_start_idx = 599
    #         if zoom_end_idx is None:
    #             zoom_end_idx = 699

    # Prepare target times for optional downsampling of 0D CSVs
    target_times = None
    if downsample_0d:
        # Prefer explicit observed inflow BC times from calibration input
        if isinstance(calib_data, dict) and 'observed_inflow_bc' in calib_data and isinstance(calib_data['observed_inflow_bc'].get('t'), list):
            target_times = calib_data['observed_inflow_bc']['t']
            if verbose:
                print(f"  Using observed_inflow_bc times for downsampling ({len(target_times)} points)")
        else:
            # Fallback: use normalized time grid matching max 3D observation length
            if max_3d_length > 0:
                target_times = np.linspace(0.0, 1.0, max_3d_length).tolist()
                if verbose:
                    print(f"  Using normalized time grid for downsampling ({len(target_times)} points)")
            else:
                target_times = None
    
    # Calculate MSE for each modality
    mse_results = {}
    
    for modality_name, csv_path in csv_results_dict.items():
        original_csv_path = csv_path
        if not original_csv_path or not os.path.exists(original_csv_path):
            print(f"\n  {modality_name}: ✗ CSV file not found: {original_csv_path}")
            continue

        # Optionally downsample the 0D CSV to match 3D observation times
        if downsample_0d and target_times is not None:
            # Determine output path for downsampled CSV
            csv_dir = downsample_save_dir if downsample_save_dir else os.path.dirname(original_csv_path)
            os.makedirs(csv_dir, exist_ok=True)
            base = os.path.basename(original_csv_path).replace('.csv', '')
            downsampled_name = f"{base}_downsampled_to_3d_times.csv"
            downsampled_path = os.path.join(csv_dir, downsampled_name)
            if not os.path.exists(downsampled_path):
                if verbose:
                    print(f"\n  {modality_name}: downsampling 0D CSV to match 3D times -> {downsampled_path}")
                ok = downsample_csv_file_to_times(original_csv_path, downsampled_path, target_times, method=downsample_method, verbose=verbose)
                if not ok:
                    print(f"    ✗ Downsampling failed for: {original_csv_path}")
                    # Fall back to original CSV
                    csv_path = original_csv_path
                else:
                    csv_path = downsampled_path
                    if verbose:
                        print(f"    ✓ Using downsampled CSV: {downsampled_path}")
            else:
                # Use existing downsampled file
                csv_path = downsampled_path
                if verbose:
                    print(f"    ✓ Re-using existing downsampled CSV: {downsampled_path}")
        else:
            csv_path = original_csv_path

        print(f"\n  {modality_name}:")
        print(f"    Reading 0D results from: {csv_path}")

        # Read 0D CSV results
        results_0d, times_0d = read_zerod_csv(csv_path)
        
        if not results_0d or not times_0d:
            print(f"    ✗ No data found in CSV file")
            continue
        
        print(f"    Found {len(results_0d)} vessels, {len(times_0d)} time points")
        
        # Determine zoom window - matching the shaded region in plots
        # This corresponds to the zoom window used in plot_inlet_comparison.py and plot_outlet_comparison.py
        
        # Find the maximum length of 3D observations to validate zoom window
        max_3d_length = 0
        for obs_values_3d in obs_3d.values():
            if isinstance(obs_values_3d, list):
                max_3d_length = max(max_3d_length, len(obs_values_3d))
            elif hasattr(obs_values_3d, '__len__'):
                max_3d_length = max(max_3d_length, len(obs_values_3d))
        
        # Validate zoom window
        assert zoom_start_idx <= max_3d_length
        assert zoom_end_idx <= max_3d_length
        assert zoom_start_idx <= zoom_end_idx
            
        
        num_zoom_timesteps = zoom_end_idx - zoom_start_idx
        print(f"    Using zoom window: timesteps {zoom_start_idx} to {zoom_end_idx-1} ({num_zoom_timesteps} timesteps)")
        
        # Calculate MSE for each observation
        modality_mse = {}
        total_mse_pressure = []
        total_mse_flow = []
        
        for obs_key, obs_values_3d in obs_3d.items():
            if not isinstance(obs_values_3d, list):
                obs_values_3d = obs_values_3d.tolist() if hasattr(obs_values_3d, 'tolist') else list(obs_values_3d)
            
            # Apply zoom window filter: use timesteps in range [zoom_start_idx:zoom_end_idx] (matching shaded region in plots)
            if len(obs_values_3d) > zoom_start_idx:
                obs_values_3d = obs_values_3d[zoom_start_idx:zoom_end_idx]
            else:
                # If 3D data is shorter than zoom window, skip this observation
                continue
            
            # Parse observation key: "pressure:INFLOW:branch0_seg0" or "flow:branch0_seg0:J0"
            parts = obs_key.split(':')
            if len(parts) != 3:
                continue
            
            obs_type = parts[0]  # 'pressure' or 'flow'
            part1 = parts[1]     # e.g., 'INFLOW', 'branch0_seg0', or 'J0'
            part2 = parts[2]     # e.g., 'branch0_seg0' or 'J0'
            
            # Determine vessel name and field (pressure_in/out, flow_in/out)
            vessel_name = None
            field_name = None
            
            # Handle inlet observations: "pressure:INFLOW:branch0_seg0" or "flow:INFLOW:branch0_seg0"
            if part1 == 'INFLOW':
                vessel_name = part2
                field_name = f"{obs_type}_in"
            # Handle outlet observations at junctions: "pressure:branch0_seg0:J0" or "flow:branch0_seg0:J0"
            elif part1.startswith('branch') and part2.startswith('J'):
                vessel_name = part1
                field_name = f"{obs_type}_out"
            # Handle inlet observations at junctions: "pressure:J0:branch1_seg0" or "flow:J0:branch1_seg0"
            elif part1.startswith('J') and part2.startswith('branch'):
                vessel_name = part2
                field_name = f"{obs_type}_in"
            else:
                # Try to find vessel name in parts
                for p in [part1, part2]:
                    if p.startswith('branch'):
                        vessel_name = p
                        # Guess field based on position
                        if part1 == vessel_name:
                            field_name = f"{obs_type}_out"
                        else:
                            field_name = f"{obs_type}_in"
                        break
            
            if not vessel_name or not field_name:
                continue

            # FILTER: only include vessels that exist in the original 0D geometry
            # If geometric input was provided, use its `vessels` list to decide which
            # observation locations are valid for MSE. Also exclude any vessel name
            # that looks like a connector (contains the substring 'connector').
            try:
                if vessels:
                    vessel_name_set = set([v.get('vessel_name') for v in vessels if v.get('vessel_name')])
                    #if vessel_name not in vessel_name_set:
                    if 'connector' in vessel_name.lower():
                        if verbose:
                            print(f"    Skipping observation {obs_key}: '{vessel_name}' is a connector or not in original 0D vessels")
                        continue
                    #else:
                    #    if verbose:
                    #        print(f"    Skipping observation {obs_key}: '{vessel_name}' not found in original 0D vessels")
                    #continue
            except Exception:
                # If anything goes wrong while checking geometry, fall back to previous behavior
                pass
            
            # Extract 0D data for this vessel
            if vessel_name not in results_0d:
                continue
            
            # Extract 0D values at available time points
            times_0d_valid = []
            obs_values_0d_raw = []
            for time in sorted(times_0d):
                if time in results_0d[vessel_name] and field_name in results_0d[vessel_name][time]:
                    times_0d_valid.append(time)
                    obs_values_0d_raw.append(results_0d[vessel_name][time][field_name])
            
            if len(times_0d_valid) < 2:
                continue
            
            # Apply zoom window filter to 0D data: use same index range [zoom_start_idx:zoom_end_idx]
            if len(obs_values_0d_raw) > zoom_start_idx:
                obs_values_0d_zoomed = obs_values_0d_raw[zoom_start_idx:zoom_end_idx]
                times_0d_zoomed = times_0d_valid[zoom_start_idx:zoom_end_idx]
            else:
                # If 0D data is shorter than zoom window, skip this observation
                continue
            
            if len(obs_values_0d_zoomed) == 0:
                continue
            
            # Interpolate 0D data to match 3D observation time points (now both are in zoom window)
            if len(obs_values_3d) != len(obs_values_0d_zoomed):
                # Interpolate 0D data to match 3D observation count
                # Use linear interpolation
                times_0d_array = np.array(times_0d_zoomed)
                obs_values_0d_array = np.array(obs_values_0d_zoomed)
                
                # Create time points matching 3D observation count
                time_min = times_0d_array[0]
                time_max = times_0d_array[-1]
                times_3d_interp = np.linspace(time_min, time_max, len(obs_values_3d))
                
                # Interpolate 0D values
                if HAS_SCIPY_INTERP:
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(times_0d_array, obs_values_0d_array, kind='linear', 
                                          bounds_error=False, fill_value=(obs_values_0d_array[0], obs_values_0d_array[-1]))
                    obs_values_0d = interp_func(times_3d_interp)
                else:
                    # Use numpy interpolation
                    obs_values_0d = np.interp(times_3d_interp, times_0d_array, obs_values_0d_array)
            else:
                # Same length, use directly (both are already in zoom window)
                obs_values_0d = np.array(obs_values_0d_zoomed)
            
            # Convert to numpy arrays
            obs_values_0d = np.array(obs_values_0d)
            obs_values_3d = np.array(obs_values_3d)
            
            # Ensure 0D data matches the length of filtered 3D data (zoom window)
            # If 0D data is longer, take the last portion matching 3D length
            if len(obs_values_0d) > len(obs_values_3d):
                obs_values_0d = obs_values_0d[-len(obs_values_3d):]
            elif len(obs_values_0d) < len(obs_values_3d):
                # If 0D data is shorter, pad or truncate 3D to match
                obs_values_3d = obs_values_3d[:len(obs_values_0d)]
            
            # Remove NaN and inf values
            valid_mask = np.isfinite(obs_values_0d) & np.isfinite(obs_values_3d)
            if not np.any(valid_mask):
                continue
            
            obs_values_0d_clean = obs_values_0d[valid_mask]
            obs_values_3d_clean = obs_values_3d[valid_mask]
            
            # Ensure same length (take minimum)
            min_len = min(len(obs_values_0d_clean), len(obs_values_3d_clean))
            if min_len == 0:
                continue
            
            obs_values_0d_clean = obs_values_0d_clean[:min_len]
            obs_values_3d_clean = obs_values_3d_clean[:min_len]
            
            # Calculate MSE
            diff = obs_values_0d_clean - obs_values_3d_clean
            
            mse = np.mean(diff**2)
            mae = np.mean(np.abs(diff))
            
            # Save comparison plot for debugging
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                
                # Create output directory for debug plots
                base_dir = os.path.dirname(calibration_input_path)
                debug_plots_dir = os.path.join(base_dir, 'mse_debug_plots')
                os.makedirs(debug_plots_dir, exist_ok=True)
                
                # Create safe filename from observation key
                safe_obs_key = obs_key.replace(':', '_').replace('/', '_')
                plot_filename = f"{modality_name}_{safe_obs_key}_comparison.png"
                plot_path = os.path.join(debug_plots_dir, plot_filename)
                
                # Create comparison plot
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                # Plot 0D vs 3D as scatter or line plot
                if len(obs_values_0d_clean) > 50:
                    # For many points, use scatter
                    ax.scatter(obs_values_3d_clean, obs_values_0d_clean, alpha=0.5, s=10, label='Data points')
                else:
                    # For fewer points, use line plot
                    ax.plot(obs_values_3d_clean, obs_values_0d_clean, 'o-', alpha=0.7, markersize=4, label='Data points')
                
                # Add diagonal line (perfect match)
                min_val = min(np.min(obs_values_3d_clean), np.min(obs_values_0d_clean))
                max_val = max(np.max(obs_values_3d_clean), np.max(obs_values_0d_clean))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match', alpha=0.5)
                
                ax.set_xlabel('3D Observations', fontsize=12)
                ax.set_ylabel('0D Results', fontsize=12)
                # Format MSE and MAE in engineering notation
                ax.set_title(f'{modality_name}: {obs_key}\nMSE={mse:.3E}, MAE={mae:.3E}, n={min_len}', fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                pass  # Silently fail if plotting is not available
            
            modality_mse[obs_key] = {
                'mse': mse,
                'type': obs_type,
                'vessel': vessel_name,
                'n_points': min_len
            }
            
            if obs_type == 'pressure':
                total_mse_pressure.append(mse)
            else:
                total_mse_flow.append(mse)
        
        # Store results
        mse_results[modality_name] = {
            'individual': modality_mse,
            'mean_pressure_mse': np.mean(total_mse_pressure) if total_mse_pressure else np.nan,
            'mean_flow_mse': np.mean(total_mse_flow) if total_mse_flow else np.nan,
            'overall_mse': np.mean(total_mse_pressure + total_mse_flow) if (total_mse_pressure or total_mse_flow) else np.nan
        }
        
        # Print summary (only in verbose mode)
        if verbose:
            print(f"    Calculated MSE for {len(modality_mse)} observation series")
            if total_mse_pressure:
                print(f"    Mean Pressure MSE: {np.mean(total_mse_pressure):.3E}")
            if total_mse_flow:
                print(f"    Mean Flow MSE: {np.mean(total_mse_flow):.3E}")
            if total_mse_pressure or total_mse_flow:
                print(f"    Overall MSE: {np.mean(total_mse_pressure + total_mse_flow):.3E}")
    
    # Get all observation keys and modalities for summary
    all_obs_keys = set()
    for modality_results in mse_results.values():
        all_obs_keys.update(modality_results['individual'].keys())
    
    if not all_obs_keys:
        if verbose:
            print("  No observations found for comparison")
        return mse_results
    
    modalities = list(csv_results_dict.keys())
    
    # Print detailed comparison table (only in verbose mode)
    if verbose:
        print("\n" + "="*80)
        print("Detailed MSE Comparison")
        print("="*80)
        
        # Print header
        print(f"\n{'Observation':<40} {'Type':<10} ", end="")
        for mod in modalities:
            if mod in mse_results:
                print(f"{mod:<15} ", end="")
        print()
        print("-" * (50 + 15 * len([m for m in modalities if m in mse_results])))
        
        # Print each observation
        for obs_key in sorted(all_obs_keys):
            # Truncate long keys for display
            display_key = obs_key[:38] + ".." if len(obs_key) > 40 else obs_key
            
            # Get type from first available result
            obs_type = "unknown"
            for modality_results in mse_results.values():
                if obs_key in modality_results['individual']:
                    obs_type = modality_results['individual'][obs_key]['type']
                    break
            
            print(f"{display_key:<40} {obs_type:<10} ", end="")
            for mod in modalities:
                if mod in mse_results and obs_key in mse_results[mod]['individual']:
                    mse_val = mse_results[mod]['individual'][obs_key]['mse']
                    print(f"{mse_val:>13.3E}  ", end="")
                else:
                    print(f"{'N/A':>13}  ", end="")
            print()
        
        print("\n" + "-" * (50 + 15 * len([m for m in modalities if m in mse_results])))
    
    # Always print summary statistics with column headers
    # Print header row
    print(f"{'':<40} {'':<10} ", end="")
    for mod in modalities:
        if mod in mse_results:
            print(f"{mod:<15} ", end="")
    print()
    print("-" * (50 + 15 * len([m for m in modalities if m in mse_results])))
    
    # Print summary rows
    print(f"{'SUMMARY':<40} {'':<10} ", end="")
    for mod in modalities:
        if mod in mse_results:
            overall = mse_results[mod]['overall_mse']
            if not np.isnan(overall):
                print(f"{overall:>13.3E}  ", end="")
            else:
                print(f"{'N/A':>13}  ", end="")
    print()
    
    print(f"{'Mean Pressure MSE':<40} {'':<10} ", end="")
    for mod in modalities:
        if mod in mse_results:
            mse_p = mse_results[mod]['mean_pressure_mse']
            if not np.isnan(mse_p):
                print(f"{mse_p:>13.3E}  ", end="")
            else:
                print(f"{'N/A':>13}  ", end="")
    print()
    
    print(f"{'Mean Flow MSE':<40} {'':<10} ", end="")
    for mod in modalities:
        if mod in mse_results:
            mse_f = mse_results[mod]['mean_flow_mse']
            if not np.isnan(mse_f):
                print(f"{mse_f:>13.3E}  ", end="")
            else:
                print(f"{'N/A':>13}  ", end="")
    print()
    
    # Save results to CSV file
    if output_csv_path is None:
        # Auto-generate CSV path based on calibration input path
        base_dir = os.path.dirname(calibration_input_path)
        base_name = os.path.basename(calibration_input_path).replace('.json', '')
        output_csv_path = os.path.join(base_dir, f'{base_name}_mse_comparison.csv')
    
    try:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Write CSV with two sections: summary and detailed
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['MSE Comparison Results'])
            writer.writerow(['Zoom Window', f'{zoom_start_idx} to {zoom_end_idx-1}'])
            writer.writerow([])
            
            # Write summary statistics
            writer.writerow(['Summary Statistics'])
            writer.writerow(['Metric'] + modalities)
            
            # Overall MSE
            row = ['Overall MSE']
            for mod in modalities:
                if mod in mse_results:
                    overall = mse_results[mod]['overall_mse']
                    if not np.isnan(overall):
                        row.append(f'{overall:.3E}')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            writer.writerow(row)
            
            # Mean Pressure MSE
            row = ['Mean Pressure MSE']
            for mod in modalities:
                if mod in mse_results:
                    mse_p = mse_results[mod]['mean_pressure_mse']
                    if not np.isnan(mse_p):
                        row.append(f'{mse_p:.3E}')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            writer.writerow(row)
            
            # Mean Flow MSE
            row = ['Mean Flow MSE']
            for mod in modalities:
                if mod in mse_results:
                    mse_f = mse_results[mod]['mean_flow_mse']
                    if not np.isnan(mse_f):
                        row.append(f'{mse_f:.3E}')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            writer.writerow(row)
            
            writer.writerow([])
            writer.writerow(['Detailed Results'])
            writer.writerow(['Observation', 'Type', 'Vessel'] + modalities)
            
            # Write individual observation results
            for obs_key in sorted(all_obs_keys):
                # Get type and vessel from first available result
                obs_type = "unknown"
                vessel_name = "unknown"
                for modality_results in mse_results.values():
                    if obs_key in modality_results['individual']:
                        obs_type = modality_results['individual'][obs_key]['type']
                        vessel_name = modality_results['individual'][obs_key].get('vessel', 'unknown')
                        break
                
                row = [obs_key, obs_type, vessel_name]
                for mod in modalities:
                    if mod in mse_results and obs_key in mse_results[mod]['individual']:
                        mse_val = mse_results[mod]['individual'][obs_key]['mse']
                        row.append(f'{mse_val:.3E}')
                    else:
                        row.append('N/A')
                writer.writerow(row)
        
        print(f"\n  ✓ MSE results saved to: {output_csv_path}")
        
    except Exception as e:
        print(f"  ✗ Warning: Could not save MSE results to CSV: {e}")
        import traceback
        traceback.print_exc()
    
    return mse_results
