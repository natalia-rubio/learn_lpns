import os
import json
import numpy as np
from scipy.interpolate import interp1d
from util.zerod_calibration.file_io import read_centerline_vtp
from util.zerod_calibration.file_io import parse_simulation_xml


# def replace_inlet_bc_in_calibrated_output(calibrated_output_path, calibration_input_path):
#     """
#     Replace the inlet boundary condition in calibrated output with the original observed BC from calibration input.
    
#     Args:
#         calibrated_output_path: Path to calibrated output JSON file
#         calibration_input_path: Path to calibration input JSON file (contains observed_inflow_bc)
        
#     Returns:
#         True if BC was replaced, False otherwise
#     """
#     # Read calibration input to get original observed BC
#     if not os.path.exists(calibration_input_path):
#         print(f"  Warning: Calibration input not found at {calibration_input_path}")
#         return False
    
#     with open(calibration_input_path, 'r') as f:
#         calib_input = json.load(f)
    
#     original_bc_values = calib_input.get('observed_inflow_bc')
#     if original_bc_values is None or 't' not in original_bc_values or 'Q' not in original_bc_values:
#         print("  Warning: No observed_inflow_bc found in calibration input, keeping calibrated output BC")
#         return False
    
#     # Read calibrated output
#     if not os.path.exists(calibrated_output_path):
#         print(f"  Warning: Calibrated output not found at {calibrated_output_path}")
#         return False
    
#     with open(calibrated_output_path, 'r') as f:
#         calibrated_output = json.load(f)
    
#     # Find and update the INFLOW BC
#     bc_found = False
#     for bc in calibrated_output.get('boundary_conditions', []):
#         if bc.get('bc_name') == 'INFLOW':
#             bc['bc_values'] = {
#                 't': original_bc_values['t'].copy() if isinstance(original_bc_values['t'], list) else original_bc_values['t'].tolist(),
#                 'Q': original_bc_values['Q'].copy() if isinstance(original_bc_values['Q'], list) else original_bc_values['Q'].tolist()
#             }
#             bc_found = True
#             print(f"  Replaced INFLOW BC with original observed BC")
#             print(f"    Time points: {len(bc['bc_values']['t'])}")
#             print(f"    Time range: [{bc['bc_values']['t'][0]:.6f}, {bc['bc_values']['t'][-1]:.6f}] s")
#             break
    
#     if not bc_found:
#         # Add inflow BC if it doesn't exist
#         if 'boundary_conditions' not in calibrated_output:
#             calibrated_output['boundary_conditions'] = []
#         calibrated_output['boundary_conditions'].append({
#             'bc_name': 'INFLOW',
#             'bc_type': 'FLOW',
#             'bc_values': {
#                 't': original_bc_values['t'].copy() if isinstance(original_bc_values['t'], list) else original_bc_values['t'].tolist(),
#                 'Q': original_bc_values['Q'].copy() if isinstance(original_bc_values['Q'], list) else original_bc_values['Q'].tolist()
#             }
#         })
#         print(f"  Added INFLOW BC with original observed BC")
#         print(f"    Time points: {len(calibrated_output['boundary_conditions'][-1]['bc_values']['t'])}")
    

#     # Write updated calibrated output
#     with open(calibrated_output_path, 'w') as f:
#         json.dump(calibrated_output, f, indent=4)
    
#     return True

def update_geometric_input_with_calibration_bc(geometric_input_path, calibration_input_path):
    """
    Update geometric_input.json with inflow BC from calibration_input.json.
    Uses the full time frame (not the second half used for calibration).
    
    Args:
        geometric_input_path: Path to geometric input JSON
        calibration_input_path: Path to calibration input JSON
    """
    print(f"\nUpdating geometric input with inflow BC from calibration input...")
    
    if not os.path.exists(calibration_input_path):
        print(f"  Warning: Calibration input not found at {calibration_input_path}")
        return False
    
    # Read calibration input
    with open(calibration_input_path, 'r') as f:
        calib_data = json.load(f)
    
    # Use full BC if available (stored for forward simulations), otherwise use calibration BC
    if '_full_bc_for_forward_sim' in calib_data:
        # Use full time frame for forward simulations
        calib_inflow_bc = calib_data['_full_bc_for_forward_sim'].copy()
        print(f"  Using full time frame from calibration input (for forward simulations)")
    else:
        # Fallback: extract from boundary conditions (this would be second half)
        calib_inflow_bc = None
        for bc in calib_data.get('boundary_conditions', []):
            if bc.get('bc_name') == 'INFLOW':
                calib_inflow_bc = bc.get('bc_values', {})
                break
        if not calib_inflow_bc:
            print(f"  Warning: Could not find INFLOW BC in calibration input")
            return False
        print(f"  Warning: Full BC not found, using calibration BC (may be second half)")
    
    # Read geometric input
    with open(geometric_input_path, 'r') as f:
        geo_input = json.load(f)
    
    # Refine inlet BC for forward simulation (halve timestep size, interpolate flow)
    refined_bc = calib_inflow_bc #  refine_inlet_bc_for_forward_simulation(calib_inflow_bc)
    
    # Update inflow BC
    geo_updated = False
    n_pts_inflow = len(refined_bc.get('t', []))
    for bc in geo_input.get('boundary_conditions', []):
        if bc.get('bc_name') == 'INFLOW':
            bc['bc_values'] = refined_bc.copy()
            geo_updated = True
            print(f"  Updated geometric input INFLOW BC (refined for forward simulation):")
            print(f"    Original number of time points: {len(calib_inflow_bc.get('t', []))}")
            print(f"    Refined number of time points: {n_pts_inflow}")
            if refined_bc.get('t'):
                print(f"    Time range: [{refined_bc['t'][0]:.6f}, {refined_bc['t'][-1]:.6f}]")
            if refined_bc.get('Q'):
                print(f"    Flow range: [{min(refined_bc['Q']):.3f}, {max(refined_bc['Q']):.3f}]")
            break

    # Ensure simulation_parameters.number_of_time_pts_per_cardiac_cycle
    # matches the length of the refined inflow BC time series
    if geo_updated and 'simulation_parameters' in geo_input:
        geo_input['simulation_parameters']['number_of_time_pts_per_cardiac_cycle'] = n_pts_inflow
        
        print(f"  Updated simulation_parameters.number_of_time_pts_per_cardiac_cycle to {n_pts_inflow}")
        
        # Update cardiac_cycle_period to match the BC time array
        if refined_bc.get('t') and len(refined_bc['t']) > 1:
            t_array = refined_bc['t']
            if t_array[0] == 0.0:
                cardiac_period = t_array[-1]
            else:
                dt = t_array[1] - t_array[0] if len(t_array) > 1 else 0.0
                cardiac_period = t_array[-1] - t_array[0] + dt
            geo_input['simulation_parameters']['cardiac_cycle_period'] = cardiac_period
            print(f"  Updated simulation_parameters.cardiac_cycle_period to {cardiac_period:.6f} s (from BC time array)")
    
    if geo_updated:
        # Write updated geometric input
        with open(geometric_input_path, 'w') as f:
            json.dump(geo_input, f, indent=4)
        print(f"  Saved updated geometric input to: {geometric_input_path}")
        return True
    else:
        print(f"  Warning: Could not find INFLOW BC in geometric input to update")
        return False

def update_outlet_bcs_in_file(file_path, outlet_params, file_type="calibration input"):
    """
    Update outlet boundary conditions (R and Pd) in a JSON file using fitted parameters.
    
    Args:
        file_path: Path to JSON file to update (calibration input or calibrated output)
        outlet_params: Dictionary mapping BC names to (R, Pd) tuples
        file_type: String describing file type (for logging)
    
    Returns:
        True if file was updated, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"  Warning: {file_type} file not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        updated_count = 0
        # Find and update outlet BCs
        for bc in data.get('boundary_conditions', []):
            if bc.get('bc_type') == 'RESISTANCE':
                bc_name = bc.get('bc_name')
                if bc_name in outlet_params:
                    resistance, pd = outlet_params[bc_name]
                    old_r = bc['bc_values'].get('R', 1.0)
                    old_pd = bc['bc_values'].get('Pd', 0.0)
                    bc['bc_values']['R'] = resistance
                    bc['bc_values']['Pd'] = pd
                    updated_count += 1
                    print(f"    {bc_name}: R {old_r:.4f} -> {resistance:.4f}, Pd {old_pd:.4f} -> {pd:.4f}")
        
        if updated_count > 0:
            # Write updated file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"  ✓ Updated {updated_count} outlet BC(s) in {file_type}: {file_path}")
            return True
        else:
            print(f"  No outlet BCs found to update in {file_type}: {file_path}")
            return False
            
    except Exception as e:
        print(f"  Warning: Could not update {file_type} file {file_path}: {e}")
        return False

def refine_inlet_bc_for_forward_simulation(input_path, refinement_factor=2):
    """
    Refine the inlet boundary condition for forward simulation.
    """
    with open(input_path, 'r') as f:
        cali = json.load(f)

    bc_time = cali['boundary_conditions'][0]['bc_values']['t']
    bc_flow = cali['boundary_conditions'][0]['bc_values']['Q']
    bc_time_refined = np.linspace(0, bc_time[-1], len(bc_time) * refinement_factor, endpoint=True).tolist()
    bc_flow_refined = interp1d(bc_time, bc_flow, kind='cubic')(bc_time_refined)
    # plot the original and refined boundary condition
    if False:
        plt.scatter(bc_time, bc_flow, label='Original')
        plt.scatter(bc_time_refined, bc_flow_refined, label='Refined')
        plt.legend()
        plt.show()
    cali['boundary_conditions'][0]['bc_values']['t'] = list(bc_time_refined)
    cali['boundary_conditions'][0]['bc_values']['Q'] = list(bc_flow_refined)
    cali['simulation_parameters']['number_of_time_pts_per_cardiac_cycle'] = len(bc_time_refined)

    cali['simulation_parameters']['number_of_cardiac_cycles'] = 10
    cali['simulation_parameters']['output_all_cycles'] = False

    with open(input_path, 'w') as f:
        json.dump(cali, f, indent=4)
    print(f"Refined inlet boundary condition for forward simulation to: {input_path}")
    return