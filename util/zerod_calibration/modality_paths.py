"""Shared paths for forward-sim CSV/JSON outputs keyed by plot/MSE modality name."""

import os


def nn_forward_sim_specs(base_dir, geo_variant_name, junction_type, nn_vessel):
    """Return (modality_key, sim_input_json, results_csv) for each NN forward sim."""
    specs = [
        (
            'BloodVesselJunction_NN',
            os.path.join(base_dir, f'{geo_variant_name}_NN_{junction_type}.json'),
            os.path.join(base_dir, f'{geo_variant_name}_NN_{junction_type}_results.csv'),
        ),
    ]
    if nn_vessel:
        specs.extend([
            (
                'BloodVesselJunction_NN_plus_Vessel_NN',
                os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel.json'),
                os.path.join(base_dir, f'{geo_variant_name}_NN_JunctionAndVessel_results.csv'),
            ),
            (
                'NN_vessel',
                os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly.json'),
                os.path.join(base_dir, f'{geo_variant_name}_NN_VesselOnly_results.csv'),
            ),
        ])
    return specs


def modality_csv_paths(geo_variant_paths, base_dir, geo_variant_name, junction_type, nn_vessel):
    """Modality name -> forward results CSV (for MSE and location comparison plots)."""
    results = {}
    geometric_results = geo_variant_paths['geometric_results']
    if os.path.exists(geometric_results):
        results['geometric'] = str(geometric_results)
    calibrated_results = geo_variant_paths['junction_types'][junction_type]['calibrated_results']
    if os.path.exists(calibrated_results):
        results[junction_type] = str(calibrated_results)
    if geo_variant_name in ('bifurcations', 'bifurcations_EL'):
        for modality_key, _, results_csv in nn_forward_sim_specs(
            base_dir, geo_variant_name, junction_type, nn_vessel,
        ):
            if os.path.exists(results_csv):
                results[modality_key] = str(results_csv)
    return results


def modality_json_paths(geo_variant_paths, base_dir, geo_variant_name, junction_type, nn_vessel):
    """Modality name -> calibrated/geometric JSON (for zero-D parameter bar charts)."""
    modality_jsons = {}
    geom_json = geo_variant_paths.get('geometric_input')
    if geom_json and os.path.exists(geom_json):
        modality_jsons['geometric'] = str(geom_json)
    calib_json = geo_variant_paths['junction_types'][junction_type]['calibrated_output']
    if calib_json and os.path.exists(calib_json):
        modality_jsons[junction_type] = str(calib_json)
    if geo_variant_name in ('bifurcations', 'bifurcations_EL'):
        for modality_key, sim_input, _ in nn_forward_sim_specs(
            base_dir, geo_variant_name, junction_type, nn_vessel,
        ):
            if os.path.exists(sim_input):
                modality_jsons[modality_key] = str(sim_input)
    return modality_jsons


def split_location_plot_csv_paths(all_csv_paths, geo_variant_name, geo_variant_paths):
    """Split modality CSV dict into plot_location_comparison arguments."""
    geometric_csv_path = geo_variant_paths['geometric_results']
    if not os.path.exists(geometric_csv_path):
        geometric_csv_path = None
    calibrated_csv_paths = {
        k: v for k, v in all_csv_paths.items() if k != 'geometric'
    }
    geometric_csv_paths = {}
    if 'geometric' in all_csv_paths:
        geometric_csv_paths[geo_variant_name] = all_csv_paths['geometric']
    return geometric_csv_path, calibrated_csv_paths, geometric_csv_paths
