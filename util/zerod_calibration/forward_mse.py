import os
import json
import numpy as np
import csv
from util.zerod_calibration.tools.file_io import read_zerod_csv

try:
    from scipy.interpolate import interp1d
    HAS_SCIPY_INTERP = True
except ImportError:
    interp1d = None
    HAS_SCIPY_INTERP = False

_PRESSURE_DYNES_TO_MMHG = 1333.322


def _downsample_csv_file_to_times(input_csv_path, output_csv_path, target_times, verbose=False):
    """
    Read a 0D CSV file and write a new CSV linearly interpolated to target_times.

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


_MSE_SUMMARY_ROWS = (
    ('Overall MSE', 'overall_mse', '.3E'),
    ('Mean Pressure MSE', 'mean_pressure_mse', '.3E'),
    ('Mean Flow MSE', 'mean_flow_mse', '.3E'),
    ('Overall Max Error', 'overall_max_error', '.3E'),
    ('Mean Pressure Max Error', 'mean_pressure_max_error', '.3E'),
    ('Mean Flow Max Error', 'mean_flow_max_error', '.3E'),
    ('Overall Max Rel Error', 'overall_max_rel_error', '.4f'),
    ('Mean Pressure Max Rel Error', 'mean_pressure_max_rel_error', '.4f'),
    ('Mean Flow Max Rel Error', 'mean_flow_max_rel_error', '.4f'),
)

MSE_METRIC_KEYS = tuple(key for _, key, _ in _MSE_SUMMARY_ROWS)
_MSE_ROW_NAME_TO_KEY = {label: key for label, key, _ in _MSE_SUMMARY_ROWS}


def parse_mse_comparison_csv(csv_path):
    """Parse MSE comparison CSV; return dict modality -> dict metric_key -> float."""
    result = {}
    if not os.path.exists(csv_path):
        return result
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        in_summary = False
        header = None
        for row in reader:
            if not row:
                continue
            if row[0] == "Summary Statistics":
                in_summary = True
                continue
            if in_summary and header is None:
                header = row
                continue
            if in_summary and header is not None:
                metric_key = _MSE_ROW_NAME_TO_KEY.get(row[0])
                if metric_key is not None:
                    for i, mod in enumerate(header[1:], start=1):
                        if mod not in result:
                            result[mod] = {}
                        if i < len(row) and row[i].strip() not in ("", "N/A"):
                            try:
                                result[mod][metric_key] = float(row[i])
                            except ValueError:
                                result[mod][metric_key] = np.nan
                        else:
                            result[mod][metric_key] = np.nan
                if row[0] == "Detailed Results":
                    break
    return result


def _load_mse_obs_3d(calib_data):
    full = calib_data.get('_full_observations', {})
    if isinstance(full, dict) and 'y' in full:
        return full['y']
    if 'y' in calib_data:
        return calib_data['y']
    return None


def _max_3d_obs_length(obs_3d):
    max_len = 0
    for obs_values in obs_3d.values():
        if isinstance(obs_values, list):
            max_len = max(max_len, len(obs_values))
        elif hasattr(obs_values, '__len__'):
            max_len = max(max_len, len(obs_values))
    return max_len


def _build_mse_target_times(calib_data, max_3d_length, verbose=False):
    inflow = calib_data.get('observed_inflow_bc', {}) if isinstance(calib_data, dict) else {}
    if isinstance(inflow.get('t'), list):
        if verbose:
            print(f"  Using observed_inflow_bc times for downsampling ({len(inflow['t'])} points)")
        return inflow['t']
    if max_3d_length > 0:
        if verbose:
            print(f"  Using normalized time grid for downsampling ({max_3d_length} points)")
        return np.linspace(0.0, 1.0, max_3d_length).tolist()
    return None


def _downsample_0d_csv_for_mse(original_csv_path, modality_name, target_times, verbose):
    csv_dir = os.path.dirname(original_csv_path)
    os.makedirs(csv_dir, exist_ok=True)
    base = os.path.basename(original_csv_path).replace('.csv', '')
    downsampled_path = os.path.join(csv_dir, f"{base}_downsampled_to_3d_times.csv")
    print(f"\n  {modality_name}: downsampling 0D CSV to match 3D times")
    print(f"    Source file: {original_csv_path}")
    print(f"    Target file: {downsampled_path}")
    ok = _downsample_csv_file_to_times(
        original_csv_path, downsampled_path, target_times, verbose=verbose,
    )
    if not ok:
        print(f"    ✗ Downsampling failed for: {original_csv_path}")
        return original_csv_path
    if verbose:
        print(f"    ✓ Using downsampled CSV: {downsampled_path}")
    return downsampled_path


def _auto_zoom_indices(times_0d, zoom_start_idx, zoom_end_idx):
    """Return zoom indices from the 60–80% time span of a 0D CSV."""
    times_sorted = sorted(times_0d)
    if not times_sorted:
        start = zoom_start_idx if zoom_start_idx is not None else 0
        end = zoom_end_idx if zoom_end_idx is not None else 100
        return start, end, None, None

    time_start = times_sorted[0]
    time_end = times_sorted[-1]
    total_span = time_end - time_start
    zoom_time_start = time_start + 0.6 * total_span
    zoom_time_end = time_start + 0.8 * total_span

    if zoom_start_idx is None:
        zoom_start_idx = int(np.searchsorted(times_sorted, zoom_time_start))
    if zoom_end_idx is None:
        zoom_end_idx = min(
            int(np.searchsorted(times_sorted, zoom_time_end, side='right')),
            len(times_sorted),
        )
    zoom_start_idx = max(0, min(zoom_start_idx, len(times_sorted) - 1))
    zoom_end_idx = min(zoom_end_idx, len(times_sorted))
    if zoom_start_idx >= zoom_end_idx:
        zoom_start_idx = int(0.6 * len(times_sorted))
        zoom_end_idx = int(0.8 * len(times_sorted))
    return zoom_start_idx, zoom_end_idx, zoom_time_start, zoom_time_end


def _clamp_zoom_to_3d_length(zoom_start_idx, zoom_end_idx, max_3d_length):
    zoom_start_idx = min(zoom_start_idx, max_3d_length)
    zoom_end_idx = min(zoom_end_idx, max_3d_length)
    if zoom_start_idx >= zoom_end_idx:
        zoom_start_idx = int(0.6 * max_3d_length)
        zoom_end_idx = int(0.8 * max_3d_length)
    return zoom_start_idx, zoom_end_idx


def _parse_inflow_observation_key(obs_key):
    parts = obs_key.split(':')
    if len(parts) != 3:
        return None
    obs_type, part1, vessel_name = parts
    if part1 != 'INFLOW':
        return None
    return obs_type, vessel_name, f"{obs_type}_in"


def _extract_0d_series(results_0d, times_0d, vessel_name, field_name):
    times_valid, values = [], []
    for time in sorted(times_0d):
        if time in results_0d.get(vessel_name, {}) and field_name in results_0d[vessel_name][time]:
            times_valid.append(time)
            values.append(results_0d[vessel_name][time][field_name])
    return times_valid, values


def _interpolate_0d_to_3d_length(obs_values_3d, obs_values_0d_zoomed, times_0d_zoomed):
    if len(obs_values_3d) == len(obs_values_0d_zoomed):
        return np.array(obs_values_0d_zoomed)
    times_0d_array = np.array(times_0d_zoomed)
    obs_values_0d_array = np.array(obs_values_0d_zoomed)
    times_3d_interp = np.linspace(times_0d_array[0], times_0d_array[-1], len(obs_values_3d))
    if HAS_SCIPY_INTERP:
        fn = interp1d(
            times_0d_array, obs_values_0d_array, kind='linear',
            bounds_error=False,
            fill_value=(obs_values_0d_array[0], obs_values_0d_array[-1]),
        )
        return fn(times_3d_interp)
    return np.interp(times_3d_interp, times_0d_array, obs_values_0d_array)


def _compute_aligned_mse(obs_values_0d, obs_values_3d, obs_type):
    obs_values_0d = np.asarray(obs_values_0d)
    obs_values_3d = np.asarray(obs_values_3d)
    if len(obs_values_0d) > len(obs_values_3d):
        obs_values_0d = obs_values_0d[-len(obs_values_3d):]
    elif len(obs_values_0d) < len(obs_values_3d):
        obs_values_3d = obs_values_3d[:len(obs_values_0d)]

    valid_mask = np.isfinite(obs_values_0d) & np.isfinite(obs_values_3d)
    if not np.any(valid_mask):
        return None
    obs_values_0d = obs_values_0d[valid_mask]
    obs_values_3d = obs_values_3d[valid_mask]
    min_len = min(len(obs_values_0d), len(obs_values_3d))
    if min_len == 0:
        return None
    obs_values_0d = obs_values_0d[:min_len]
    obs_values_3d = obs_values_3d[:min_len]

    diff = obs_values_0d - obs_values_3d
    if obs_type == 'pressure':
        diff = diff / _PRESSURE_DYNES_TO_MMHG
    idx_max = int(np.argmax(np.abs(diff)))
    x3d_at_max = obs_values_3d[idx_max]
    rel_err_at_max = (
        np.abs((x3d_at_max - obs_values_0d[idx_max]) / x3d_at_max)
        if x3d_at_max != 0 else np.nan
    )
    return {
        'mse': float(np.mean(diff ** 2)),
        'max_error': float(np.max(np.abs(diff))),
        'rel_error_at_max': rel_err_at_max,
    }


def _aggregate_modality_mse(modality_mse, total_mse_pressure, total_mse_flow,
                            total_max_pressure, total_max_flow,
                            total_max_pressure_rel, total_max_flow_rel):
    return {
        'individual': modality_mse,
        'mean_pressure_mse': np.mean(total_mse_pressure) if total_mse_pressure else np.nan,
        'mean_flow_mse': np.mean(total_mse_flow) if total_mse_flow else np.nan,
        'overall_mse': np.mean(total_mse_pressure + total_mse_flow) if (total_mse_pressure or total_mse_flow) else np.nan,
        'mean_pressure_max_error': np.mean(total_max_pressure) if total_max_pressure else np.nan,
        'mean_flow_max_error': np.mean(total_max_flow) if total_max_flow else np.nan,
        'overall_max_error': np.mean(total_max_pressure + total_max_flow) if (total_max_pressure or total_max_flow) else np.nan,
        'mean_pressure_max_rel_error': np.nanmean(total_max_pressure_rel) if total_max_pressure_rel else np.nan,
        'mean_flow_max_rel_error': np.nanmean(total_max_flow_rel) if total_max_flow_rel else np.nan,
        'overall_max_rel_error': np.nanmean(total_max_pressure_rel + total_max_flow_rel) if (total_max_pressure_rel or total_max_flow_rel) else np.nan,
    }


def _mse_metric_cell(mse_results, mod, key, fmt):
    if mod not in mse_results:
        return 'N/A'
    val = mse_results[mod].get(key, np.nan)
    if np.isnan(val):
        return 'N/A'
    return f'{val:{fmt}}'


def _write_mse_comparison_csv(output_csv_path, modalities, mse_results, all_obs_keys,
                              zoom_start_idx, zoom_end_idx):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['MSE Comparison Results'])
        writer.writerow(['Zoom Window', f'{zoom_start_idx} to {zoom_end_idx - 1}'])
        writer.writerow([])
        writer.writerow(['Summary Statistics'])
        writer.writerow(['Metric'] + modalities)
        for row_name, key, fmt in _MSE_SUMMARY_ROWS:
            writer.writerow([row_name] + [_mse_metric_cell(mse_results, mod, key, fmt) for mod in modalities])
        writer.writerow([])
        writer.writerow(['Detailed Results'])
        writer.writerow(['Observation', 'Type', 'Vessel'] + modalities)
        for obs_key in sorted(all_obs_keys):
            obs_type, vessel_name = 'unknown', 'unknown'
            for modality_results in mse_results.values():
                if obs_key in modality_results['individual']:
                    obs_type = modality_results['individual'][obs_key]['type']
                    vessel_name = modality_results['individual'][obs_key].get('vessel', 'unknown')
                    break
            row = [obs_key, obs_type, vessel_name]
            for mod in modalities:
                if mod in mse_results and obs_key in mse_results[mod]['individual']:
                    row.append(f"{mse_results[mod]['individual'][obs_key]['mse']:.3E}")
                else:
                    row.append('N/A')
            writer.writerow(row)


def _print_mse_summary_table(modalities, mse_results, all_obs_keys, verbose):
    if verbose:
        print("\n" + "=" * 80)
        print("Detailed MSE Comparison")
        print("=" * 80)
        print(f"\n{'Observation':<40} {'Type':<10} ", end="")
        for mod in modalities:
            if mod in mse_results:
                print(f"{mod:<15} ", end="")
        print()
        sep = "-" * (50 + 15 * len([m for m in modalities if m in mse_results]))
        print(sep)
        for obs_key in sorted(all_obs_keys):
            display_key = obs_key[:38] + ".." if len(obs_key) > 40 else obs_key
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
        print("\n" + sep)

    print(f"{'':<40} {'':<10} ", end="")
    for mod in modalities:
        if mod in mse_results:
            print(f"{mod:<15} ", end="")
    print()
    sep = "-" * (50 + 15 * len([m for m in modalities if m in mse_results]))
    print(sep)
    for label, key in (
        ('SUMMARY', 'overall_mse'),
        ('Mean Pressure MSE', 'mean_pressure_mse'),
        ('Mean Flow MSE', 'mean_flow_mse'),
    ):
        print(f"{label:<40} {'':<10} ", end="")
        for mod in modalities:
            if mod in mse_results:
                val = mse_results[mod][key]
                print(f"{val:>13.3E}  " if not np.isnan(val) else f"{'N/A':>13}  ", end="")
        print()


def _compute_modality_mse(csv_path, obs_3d, zoom_start_idx, zoom_end_idx,
                          zoom_window_calculated, max_3d_length, verbose):
    """Compute per-observation MSE for one 0D results CSV."""
    results_0d, times_0d = read_zerod_csv(csv_path)
    if not results_0d or not times_0d:
        print(f"    ✗ No data found in CSV file")
        return None, zoom_start_idx, zoom_end_idx, zoom_window_calculated

    print(f"    Found {len(results_0d)} vessels, {len(times_0d)} time points")

    if not zoom_window_calculated and (zoom_start_idx is None or zoom_end_idx is None):
        zoom_start_idx, zoom_end_idx, zoom_time_start, zoom_time_end = _auto_zoom_indices(
            times_0d, zoom_start_idx, zoom_end_idx,
        )
        zoom_window_calculated = True
        if zoom_time_start is not None:
            print(
                f"    Auto-calculated zoom window: {zoom_time_start:.4f}s to {zoom_time_end:.4f}s "
                f"(indices {zoom_start_idx} to {zoom_end_idx})"
            )

    zoom_start_idx, zoom_end_idx = _clamp_zoom_to_3d_length(zoom_start_idx, zoom_end_idx, max_3d_length)
    print(
        f"    Using zoom window: timesteps {zoom_start_idx} to {zoom_end_idx - 1} "
        f"({zoom_end_idx - zoom_start_idx} timesteps)"
    )

    all_obs_keys = list(obs_3d.keys())
    print(f"\n    Available observation keys: {len(all_obs_keys)}")
    if verbose:
        for key in sorted(all_obs_keys):
            print(f"      - {key}")
    available_vessels = sorted(results_0d.keys())
    print(f"    Available vessels in 0D results: {len(available_vessels)}")
    if verbose:
        for v in available_vessels[:20]:
            print(f"      - {v}")
        if len(available_vessels) > 20:
            print(f"      ... and {len(available_vessels) - 20} more")

    modality_mse = {}
    total_mse_pressure, total_mse_flow = [], []
    total_max_pressure, total_max_flow = [], []
    total_max_pressure_rel, total_max_flow_rel = [], []
    locations_processed, locations_skipped = [], []

    for obs_key, obs_values_3d in obs_3d.items():
        if not isinstance(obs_values_3d, list):
            obs_values_3d = obs_values_3d.tolist() if hasattr(obs_values_3d, 'tolist') else list(obs_values_3d)

        if len(obs_values_3d) <= zoom_start_idx:
            locations_skipped.append(f"{obs_key} (3D data too short: {len(obs_values_3d)} <= {zoom_start_idx})")
            continue
        obs_values_3d = obs_values_3d[zoom_start_idx:zoom_end_idx]

        parsed = _parse_inflow_observation_key(obs_key)
        if parsed is None:
            parts = obs_key.split(':')
            reason = f"{obs_key} (not INFLOW)" if len(parts) == 3 else f"{obs_key} (invalid format)"
            locations_skipped.append(reason)
            continue
        obs_type, vessel_name, field_name = parsed

        if 'connector' in vessel_name.lower():
            locations_skipped.append(f"{obs_key} (vessel {vessel_name} is a connector)")
            if verbose:
                print(f"    Skipping observation {obs_key}: '{vessel_name}' is a connector")
            continue
        if vessel_name not in results_0d:
            locations_skipped.append(f"{obs_key} (vessel {vessel_name} not in 0D results)")
            continue

        times_0d_valid, obs_values_0d_raw = _extract_0d_series(
            results_0d, times_0d, vessel_name, field_name,
        )
        if len(times_0d_valid) < 2:
            locations_skipped.append(f"{obs_key} (insufficient 0D time points: {len(times_0d_valid)})")
            continue
        if len(obs_values_0d_raw) <= zoom_start_idx:
            locations_skipped.append(
                f"{obs_key} (0D data too short: {len(obs_values_0d_raw)} <= {zoom_start_idx})"
            )
            continue

        obs_values_0d_zoomed = obs_values_0d_raw[zoom_start_idx:zoom_end_idx]
        times_0d_zoomed = times_0d_valid[zoom_start_idx:zoom_end_idx]
        if not obs_values_0d_zoomed:
            locations_skipped.append(f"{obs_key} (no data in zoom window)")
            continue

        obs_values_0d = _interpolate_0d_to_3d_length(obs_values_3d, obs_values_0d_zoomed, times_0d_zoomed)
        metrics = _compute_aligned_mse(obs_values_0d, obs_values_3d, obs_type)
        if metrics is None:
            locations_skipped.append(f"{obs_key} (no valid finite values)")
            continue

        locations_processed.append(obs_key)
        if verbose:
            print(
                f"      ✓ Processed {obs_key}: MSE={metrics['mse']:.3E}, "
                f"max_err={metrics['max_error']:.3E}"
            )

        modality_mse[obs_key] = {
            'mse': metrics['mse'],
            'max_error': metrics['max_error'],
            'rel_error_at_max': metrics['rel_error_at_max'],
            'type': obs_type,
            'vessel': vessel_name,
        }
        if obs_type == 'pressure':
            total_mse_pressure.append(metrics['mse'])
            total_max_pressure.append(metrics['max_error'])
            total_max_pressure_rel.append(metrics['rel_error_at_max'] if np.isfinite(metrics['rel_error_at_max']) else np.nan)
        else:
            total_mse_flow.append(metrics['mse'])
            total_max_flow.append(metrics['max_error'])
            total_max_flow_rel.append(metrics['rel_error_at_max'] if np.isfinite(metrics['rel_error_at_max']) else np.nan)

    print(f"\n    Locations used for MSE calculation: {len(locations_processed)}")
    if locations_processed:
        for loc in sorted(locations_processed):
            print(f"      ✓ {loc}")
    if locations_skipped:
        print(f"\n    Locations skipped: {len(locations_skipped)}")
        if verbose:
            for loc in sorted(locations_skipped):
                print(f"      ⊘ {loc}")
    if verbose:
        print(f"\n    Calculated MSE for {len(modality_mse)} observation series")
        if total_mse_pressure:
            print(f"    Mean Pressure MSE: {np.mean(total_mse_pressure):.3E}")
        if total_mse_flow:
            print(f"    Mean Flow MSE: {np.mean(total_mse_flow):.3E}")
        if total_mse_pressure or total_mse_flow:
            print(f"    Overall MSE: {np.mean(total_mse_pressure + total_mse_flow):.3E}")

    return (
        _aggregate_modality_mse(
            modality_mse, total_mse_pressure, total_mse_flow,
            total_max_pressure, total_max_flow, total_max_pressure_rel, total_max_flow_rel,
        ),
        zoom_start_idx, zoom_end_idx, zoom_window_calculated,
    )


def calculate_mse_between_3d_and_0d(calibration_input_path, csv_results_dict, output_csv_path=None, verbose=False, downsample_0d=True):
    """
    Calculate and print Mean Squared Error (MSE) between 3D observations and 0D solutions.
    Optionally saves results to a CSV file.

    Args:
        calibration_input_path: Path to calibration input JSON (contains 3D observations)
        csv_results_dict: Dictionary mapping modality names to CSV file paths
        output_csv_path: Optional path to save MSE results CSV
        verbose: If True, print detailed comparison tables
        downsample_0d: If True, downsample 0D CSVs to 3D observation times

    Returns:
        Dictionary mapping modality names to MSE results
    """
    print("\n" + "=" * 80)
    print("Calculating Mean Squared Error (MSE) between 3D and 0D solutions")
    print("=" * 80)
    if downsample_0d:
        print("  Downsampling of 0D CSVs is ENABLED")
    else:
        print("  Downsampling of 0D CSVs is disabled")

    if not os.path.exists(calibration_input_path):
        print(f"  ✗ Calibration input not found: {calibration_input_path}")
        return {}

    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)

    obs_3d = _load_mse_obs_3d(calib_data)
    if not obs_3d:
        print("  ✗ No 3D observations found in calibration input")
        return {}

    max_3d_length = _max_3d_obs_length(obs_3d)
    target_times = _build_mse_target_times(calib_data, max_3d_length, verbose=verbose) if downsample_0d else None

    mse_results = {}
    zoom_start_idx = None
    zoom_end_idx = None
    zoom_window_calculated = False

    for modality_name, original_csv_path in csv_results_dict.items():
        if not original_csv_path or not os.path.exists(original_csv_path):
            print(f"\n  {modality_name}: ✗ CSV file not found: {original_csv_path}")
            continue

        csv_path = original_csv_path
        if downsample_0d and target_times is not None:
            csv_path = _downsample_0d_csv_for_mse(
                original_csv_path, modality_name, target_times, verbose,
            )

        print(f"\n  {modality_name}:")
        print(f"    Reading 0D results from: {csv_path}")

        modality_result, zoom_start_idx, zoom_end_idx, zoom_window_calculated = _compute_modality_mse(
            csv_path, obs_3d,
            zoom_start_idx, zoom_end_idx, zoom_window_calculated, max_3d_length, verbose,
        )
        if modality_result is not None:
            mse_results[modality_name] = modality_result

    all_obs_keys = set()
    for modality_results in mse_results.values():
        all_obs_keys.update(modality_results['individual'].keys())
    if not all_obs_keys:
        if verbose:
            print("  No observations found for comparison")
        return mse_results

    modalities = list(csv_results_dict.keys())
    _print_mse_summary_table(modalities, mse_results, all_obs_keys, verbose)

    if output_csv_path is None:
        base_dir = os.path.dirname(calibration_input_path)
        base_name = os.path.basename(calibration_input_path).replace('.json', '')
        output_csv_path = os.path.join(base_dir, f'{base_name}_mse_comparison.csv')

    try:
        _write_mse_comparison_csv(
            output_csv_path, modalities, mse_results, all_obs_keys,
            zoom_start_idx, zoom_end_idx,
        )
        print(f"\n  ✓ MSE results saved to: {output_csv_path}")
    except Exception as e:
        print(f"  ✗ Warning: Could not save MSE results to CSV: {e}")
        import traceback
        traceback.print_exc()

    return mse_results
