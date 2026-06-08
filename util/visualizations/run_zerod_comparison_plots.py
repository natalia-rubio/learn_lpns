"""Orchestrate location comparison and zero-D parameter plots for one geometry variant."""

import os
import traceback

from util.zerod_calibration.modality_paths import (
    modality_csv_paths,
    modality_json_paths,
    split_location_plot_csv_paths,
)
from util.zerod_calibration.tools.file_io import get_time_period
from util.visualizations.plot_location_comparison import (
    build_vessel_name_mapping,
    get_all_locations_from_calibration_input,
    plot_location_comparison,
)
from util.visualizations.plot_zero_d_parameter_bars import plot_zero_d_parameter_bars


def run_location_comparison_plots(
    geometry_variant,
    geo_variant_paths,
    geometry_variants,
    base_dir,
    junction_type,
    *,
    set_name,
    geo_name,
    run_config,
    trial_id=None,
    nn_vessel=False,
    inflow_only=True,
    verbose=False,
):
    """Generate per-location 3D vs 0D comparison plots for one geometry variant."""
    variant_calibration_input = geo_variant_paths['calibration_input']
    variant_geometric_results = geo_variant_paths['geometric_results']
    variant_geometric_input = geo_variant_paths['geometric_input']

    if not os.path.exists(variant_calibration_input) or not os.path.exists(variant_geometric_results):
        print(f"    Skipping location plots for {geometry_variant} (missing calibration or geometric CSV)")
        return 0

    locations = get_all_locations_from_calibration_input(str(variant_calibration_input))
    if inflow_only:
        locations = [loc for loc in locations if loc.startswith('INFLOW:')]
    if not locations:
        print(f"    Skipping location plots for {geometry_variant} (no locations in calibration input)")
        return 0

    all_csv_paths = modality_csv_paths(
        geo_variant_paths, base_dir, geometry_variant, junction_type, nn_vessel,
    )
    geometric_csv_path, calibrated_csv_paths, geometric_csv_paths = split_location_plot_csv_paths(
        all_csv_paths, geometry_variant, geo_variant_paths,
    )
    if not calibrated_csv_paths and not geometric_csv_paths:
        print(f"    Skipping location plots for {geometry_variant} (no result CSVs found)")
        return 0

    trial_suffix = f"_trial_{trial_id}" if trial_id is not None else ""
    output_dir = os.path.join(
        'results', 'location_comparison', run_config, set_name, geo_name,
        f'{geometry_variant}{trial_suffix}',
    )
    os.makedirs(output_dir, exist_ok=True)

    time_period = get_time_period(set_name, geo_name)

    vessel_name_mapping = None
    if geometry_variant == 'bifurcations_EL':
        bifurcations_input = geometry_variants['bifurcations']['geometric_input']
        if os.path.exists(bifurcations_input) and os.path.exists(variant_geometric_input):
            vessel_name_mapping = build_vessel_name_mapping(
                bifurcations_input, variant_geometric_input,
            )

    print(f"\n  Creating {geometry_variant} location comparison plots...")
    success_count = 0
    for location in locations:
        safe_location = location.replace(':', '_')
        plot_path = os.path.join(output_dir, f"{safe_location}{trial_suffix}_comparison.png")
        try:
            if plot_location_comparison(
                str(variant_calibration_input),
                str(geometric_csv_path or variant_geometric_results),
                calibrated_csv_paths,
                location,
                plot_path,
                set_name=set_name,
                geo_name=geo_name,
                time_period=time_period,
                geometric_input_path=str(variant_geometric_input),
                verbose=False,
                geometric_csv_paths=geometric_csv_paths or None,
                vessel_name_mapping=vessel_name_mapping,
            ):
                success_count += 1
        except Exception as e:
            print(f"      ✗ Failed to create plot for {location}: {e}")
            if verbose:
                traceback.print_exc()

    print(f"    Created {success_count}/{len(locations)} location comparison plots")
    print(f"    Output directory: {output_dir}")
    return success_count


def run_zero_d_parameter_bar_plot(
    geometry_variant,
    geo_variant_paths,
    base_dir,
    junction_type,
    *,
    set_name,
    geo_name,
    run_config,
    trial_id=None,
    nn_vessel=False,
    verbose=False,
):
    """Generate R/S/L parameter bar chart for one geometry variant."""
    trial_suffix = f"_trial_{trial_id}" if trial_id is not None else ""
    prefix = '' if geometry_variant == 'original' else f'{geometry_variant}_'
    output_dir = os.path.join(
        'results', 'param_comparison', run_config, set_name, geo_name,
        f'{geometry_variant}{trial_suffix}',
    )
    os.makedirs(output_dir, exist_ok=True)

    modality_jsons = modality_json_paths(
        geo_variant_paths, base_dir, geometry_variant, junction_type, nn_vessel,
    )
    if not modality_jsons:
        print(f"    Skipping parameter bar chart for {geometry_variant} (no modality JSONs found)")
        return None

    if trial_id is not None:
        out_name = f"{prefix}trial_{trial_id}_zero_d_parameter_bars.png"
    elif prefix:
        out_name = f"{prefix}zero_d_parameter_bars.png"
    else:
        out_name = 'zero_d_parameter_bars.png'

    print(f"\n  Creating zero-D parameter bar chart for {geometry_variant}...")
    out_path = plot_zero_d_parameter_bars(
        modality_jsons, output_dir=output_dir, output_name=out_name, verbose=verbose,
    )
    if out_path:
        print(f"    ✓ Saved zero-D parameter bar chart: {out_path}")
    else:
        print(f"    ✗ Failed to create zero-D parameter bar chart for {geometry_variant}")
    return out_path


def run_zerod_comparison_plots(
    geometry_variant,
    geometry_variants,
    base_dir,
    junction_type,
    *,
    set_name,
    geo_name,
    run_config,
    trial_id=None,
    nn_vessel=False,
    verbose=False,
):
    """Run location comparison plots and parameter bar chart for --geometry_variant."""
    if geometry_variant not in geometry_variants:
        print(f"  ✗ Unknown geometry variant for plots: {geometry_variant}")
        return

    geo_variant_paths = geometry_variants[geometry_variant]
    run_location_comparison_plots(
        geometry_variant,
        geo_variant_paths,
        geometry_variants,
        base_dir,
        junction_type,
        set_name=set_name,
        geo_name=geo_name,
        run_config=run_config,
        trial_id=trial_id,
        nn_vessel=nn_vessel,
        verbose=verbose,
    )
    run_zero_d_parameter_bar_plot(
        geometry_variant,
        geo_variant_paths,
        base_dir,
        junction_type,
        set_name=set_name,
        geo_name=geo_name,
        run_config=run_config,
        trial_id=trial_id,
        nn_vessel=nn_vessel,
        verbose=verbose,
    )
