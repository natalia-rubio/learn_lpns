#!/usr/bin/env python3
"""
Helper functions for calibration workflow.
"""

import os
import vtk
import numpy as np
import xml.etree.ElementTree as ET
from vtk.util.numpy_support import vtk_to_numpy as v2n
import json
try:
    from scipy.interpolate import CubicSpline, interp1d
    HAS_SCIPY_INTERP = True
except (ImportError, ValueError, AttributeError):
    # Handle import errors and version incompatibility issues
    HAS_SCIPY_INTERP = False
    try:
        from scipy.interpolate import CubicSpline
    except (ImportError, ValueError, AttributeError):
        CubicSpline = None

def fit_outlet_resistances_from_3d(geometric_input_path, observations):
    """
    Fit outlet boundary condition resistances and distal pressures from 3D solution observations.
    Fits linear relationship: P = R*Q + Pd using least squares regression.
    
    Args:
        geometric_input_path: Path to geometric 0D input JSON (will be updated)
        observations: Dictionary with observation data (y, dy) containing outlet pressure and flow
    
    Returns:
        Dictionary mapping outlet BC names to fitted (R, Pd) tuples
    """
    print(f"\nFitting outlet resistances and distal pressures from 3D solution...")
    
    # Load geometric input
    with open(geometric_input_path, 'r') as f:
        inp = json.load(f)
    
    # Extract outlet resistances and distal pressures
    outlet_params = {}
    
    # Find all outlet BCs in geometric input
    outlet_bcs = {}
    for bc in inp.get('boundary_conditions', []):
        if bc.get('bc_type') == 'RESISTANCE':
            bc_name = bc.get('bc_name')
            if bc_name:
                outlet_bcs[bc_name] = bc
    
    # Find vessels with outlet BCs
    vessels = inp.get('vessels', [])
    vessel_to_bc = {}
    for vessel in vessels:
        if 'boundary_conditions' in vessel and 'outlet' in vessel['boundary_conditions']:
            bc_name = vessel['boundary_conditions']['outlet']
            vessel_name = vessel['vessel_name']
            vessel_to_bc[vessel_name] = bc_name
    
    print(f"  Found {len(outlet_bcs)} outlet boundary conditions")
    
    # Extract pressure and flow for each outlet
    obs_y = observations.get('y', {})
    
    for vessel_name, bc_name in vessel_to_bc.items():
        if bc_name not in outlet_bcs:
            print(f"  Warning: BC {bc_name} not found in boundary_conditions")
            continue
        
        # Look for pressure and flow observations
        # Pattern: "pressure:{vessel_name}:{bc_name}" and "flow:{vessel_name}:{bc_name}"
        pressure_key = f"pressure:{vessel_name}:{bc_name}"
        flow_key = f"flow:{vessel_name}:{bc_name}"
        
        if pressure_key not in obs_y:
            print(f"  Warning: No pressure observation found for {vessel_name}:{bc_name}")
            continue
        
        if flow_key not in obs_y:
            print(f"  Warning: No flow observation found for {vessel_name}:{bc_name}")
            continue
        
        pressures = np.array(obs_y[pressure_key])
        flows = np.array(obs_y[flow_key])
        
        # Convert to numpy arrays if needed
        if isinstance(pressures, list):
            pressures = np.array(pressures)
        if isinstance(flows, list):
            flows = np.array(flows)
        
        # Filter out invalid values (inf, nan)
        valid_mask = np.isfinite(pressures) & np.isfinite(flows)
        
        if not np.any(valid_mask):
            print(f"  Warning: No valid data points for {vessel_name}:{bc_name}")
            print(f"    Pressure range: [{np.min(pressures):.2f}, {np.max(pressures):.2f}]")
            print(f"    Flow range: [{np.min(flows):.2f}, {np.max(flows):.2f}]")
            continue
        
        valid_pressures = pressures[valid_mask]
        valid_flows = flows[valid_mask]
        
        if len(valid_pressures) < 2:
            print(f"  Warning: Insufficient data points for {vessel_name}:{bc_name} (need at least 2)")
            continue
        
        # Fit linear relationship: P = R*Q + Pd
        # Using least squares: [R, Pd] = (Q^T * Q)^(-1) * Q^T * P
        # Where Q is the design matrix: [flows, ones]
        try:
            # Create design matrix: [flows, ones] for [R, Pd]
            A = np.vstack([valid_flows, np.ones(len(valid_flows))]).T
            b = valid_pressures
            
            # Solve least squares: [R, Pd] = (A^T * A)^(-1) * A^T * b
            params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            fitted_resistance = float(params[0])
            fitted_pd = float(params[1])
            
            # Check if fit is reasonable
            if not np.isfinite(fitted_resistance) or not np.isfinite(fitted_pd):
                print(f"  Warning: Invalid fit parameters for {vessel_name}:{bc_name}")
                continue
            
            # Check if resistance is positive (should be for physical validity)
            if fitted_resistance < 0:
                print(f"  Warning: Negative resistance fitted for {vessel_name}:{bc_name} ({fitted_resistance:.4f}), using absolute value")
                fitted_resistance = abs(fitted_resistance)
            
            outlet_params[bc_name] = (fitted_resistance, fitted_pd)
            
            # Calculate R-squared for quality assessment
            predicted_pressures = fitted_resistance * valid_flows + fitted_pd
            ss_res = np.sum((valid_pressures - predicted_pressures) ** 2)
            ss_tot = np.sum((valid_pressures - np.mean(valid_pressures)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            print(f"  {bc_name} ({vessel_name}):")
            print(f"    Pressure range: [{np.min(valid_pressures):.2f}, {np.max(valid_pressures):.2f}] dynes/cm²")
            print(f"    Flow range: [{np.min(valid_flows):.2f}, {np.max(valid_flows):.2f}] cm³/s")
            print(f"    Fitted R: {fitted_resistance:.4f}")
            print(f"    Fitted Pd: {fitted_pd:.4f}")
            print(f"    R²: {r_squared:.4f}")
            
        except np.linalg.LinAlgError as e:
            print(f"  Warning: Linear regression failed for {vessel_name}:{bc_name}: {e}")
            continue
    
    # Update geometric input with fitted resistances and distal pressures
    print(f"\n  Updating geometric input with fitted parameters...")
    for bc_name, (resistance, pd) in outlet_params.items():
        if bc_name in outlet_bcs:
            old_resistance = outlet_bcs[bc_name]['bc_values'].get('R', 1.0)
            old_pd = outlet_bcs[bc_name]['bc_values'].get('Pd', 0.0)
            outlet_bcs[bc_name]['bc_values']['R'] = resistance
            outlet_bcs[bc_name]['bc_values']['Pd'] = pd
            print(f"    {bc_name}: R {old_resistance:.4f} -> {resistance:.4f}, Pd {old_pd:.4f} -> {pd:.4f}")
    
    # Save updated geometric input
    with open(geometric_input_path, 'w') as f:
        json.dump(inp, f, indent=4)
    
    print(f"  ✓ Updated geometric input saved to: {geometric_input_path}")
    
    return outlet_params

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

def load_from_json(json_path):
    """
    Load JSON file from path.
    """
    with open(json_path, 'r') as f:
        print(f"  Loading JSON file from: {json_path}")
        geometric_input = json.load(f)
    return geometric_input

def save_to_json(geometric_input, json_path):
    """
    Save geometric input to JSON file.
    """
    with open(json_path, 'w') as f:
        json.dump(geometric_input, f, indent=4)
    print(f"  Saved geometric input to: {json_path}")
    return


def get_time_period(set_name, geo_name):
    """
    Try to get the actual time period from 3D simulation XML.
    Returns time period in seconds, or None if not found.
    """
    # Try to find XML file
    xml_paths = [
        os.path.join('data', 'threeD', set_name, geo_name, 'fluid_simulation_0-0.xml'),
        os.path.join('data', 'threeD', set_name, geo_name, 'solver.inp'),
    ]
    
    for xml_path in xml_paths:
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Look for time step size and number of time steps
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
            except Exception as e:
                pass
    
    return None

def replace_inlet_bc_in_calibrated_output(calibrated_output_path, calibration_input_path):
    """
    Replace the inlet boundary condition in calibrated output with the original observed BC from calibration input.
    
    Args:
        calibrated_output_path: Path to calibrated output JSON file
        calibration_input_path: Path to calibration input JSON file (contains observed_inflow_bc)
        
    Returns:
        True if BC was replaced, False otherwise
    """
    # Read calibration input to get original observed BC
    if not os.path.exists(calibration_input_path):
        print(f"  Warning: Calibration input not found at {calibration_input_path}")
        return False
    
    with open(calibration_input_path, 'r') as f:
        calib_input = json.load(f)
    
    original_bc_values = calib_input.get('observed_inflow_bc')
    if original_bc_values is None or 't' not in original_bc_values or 'Q' not in original_bc_values:
        print("  Warning: No observed_inflow_bc found in calibration input, keeping calibrated output BC")
        return False
    
    # Read calibrated output
    if not os.path.exists(calibrated_output_path):
        print(f"  Warning: Calibrated output not found at {calibrated_output_path}")
        return False
    
    with open(calibrated_output_path, 'r') as f:
        calibrated_output = json.load(f)
    
    # Find and update the INFLOW BC
    bc_found = False
    for bc in calibrated_output.get('boundary_conditions', []):
        if bc.get('bc_name') == 'INFLOW':
            bc['bc_values'] = {
                't': original_bc_values['t'].copy() if isinstance(original_bc_values['t'], list) else original_bc_values['t'].tolist(),
                'Q': original_bc_values['Q'].copy() if isinstance(original_bc_values['Q'], list) else original_bc_values['Q'].tolist()
            }
            bc_found = True
            print(f"  Replaced INFLOW BC with original observed BC")
            print(f"    Time points: {len(bc['bc_values']['t'])}")
            print(f"    Time range: [{bc['bc_values']['t'][0]:.6f}, {bc['bc_values']['t'][-1]:.6f}] s")
            break
    
    if not bc_found:
        # Add inflow BC if it doesn't exist
        if 'boundary_conditions' not in calibrated_output:
            calibrated_output['boundary_conditions'] = []
        calibrated_output['boundary_conditions'].append({
            'bc_name': 'INFLOW',
            'bc_type': 'FLOW',
            'bc_values': {
                't': original_bc_values['t'].copy() if isinstance(original_bc_values['t'], list) else original_bc_values['t'].tolist(),
                'Q': original_bc_values['Q'].copy() if isinstance(original_bc_values['Q'], list) else original_bc_values['Q'].tolist()
            }
        })
        print(f"  Added INFLOW BC with original observed BC")
        print(f"    Time points: {len(calibrated_output['boundary_conditions'][-1]['bc_values']['t'])}")
    

    # Write updated calibrated output
    with open(calibrated_output_path, 'w') as f:
        json.dump(calibrated_output, f, indent=4)
    
    return True

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

def extract_observations_from_1d(centerline_soln_path, geometric_input_path, geo_dir=None, start_idx=0, derivative_method='central'):
    """
    Extract observation data from 1D centerline solution VTP file.
    Extracts observations at boundaries and junctions following the format expected by svZeroDCalibrator.
    
    Args:
        centerline_soln_path: Path to centerline solution VTP (with pressure/velocity arrays)
        geometric_input_path: Path to geometric 0D input JSON (to understand vessel/junction structure)
        geo_dir: Geometry directory containing XML file (optional, will try to infer from paths)
        start_idx: Starting index for observations (default: 0). Observations will be sliced from this index.
        derivative_method: Method for computing derivatives ('central', 'forward', or 'backward', default: 'forward').
                          'central' uses central differences (np.gradient), 
                          'forward' uses forward differences (f[i+1] - f[i]) / dt,
                          'backward' uses backward differences (f[i] - f[i-1]) / dt.
        
    Returns:
        Dictionary with observation data (y, dy) for calibration
    """
    print(f"Reading 1D solution from: {centerline_soln_path}")
    centerline_data, _ = read_centerline_vtp(centerline_soln_path)
    
    # print the start index
    print(f"  Start index for observations: {start_idx}")
    # Read geometric input to understand vessel/junction structure
    with open(geometric_input_path, 'r') as f:
        geometric_input = json.load(f)
    
    vessels = geometric_input.get('vessels', [])
    junctions = geometric_input.get('junctions', [])
    
    # Create vessel name to index mapping
    vessel_name_to_idx = {v['vessel_name']: i for i, v in enumerate(vessels)}
    
    # Find all timestep arrays
    pressure_timesteps = []
    flow_timesteps = []
    end_idx = -1
    
    for key in centerline_data.keys():
        if key.startswith('pressure_'):
            pressure_timesteps.append(key)
        elif key.startswith('velocity_') or key.startswith('flow_'):
            flow_timesteps.append(key)
    
    # Sort timesteps
    def extract_timestep(name):
        try:
            return int(name.split('_')[-1])
        except:
            return 0
    
    pressure_timesteps.sort(key=extract_timestep)
    flow_timesteps.sort(key=extract_timestep)
    
    if not pressure_timesteps or not flow_timesteps:
        raise ValueError("No pressure or flow timesteps found in centerline solution")
    
    # Calculate timestep increment from the solution
    timestep_indices = [extract_timestep(name) for name in pressure_timesteps]
    if len(timestep_indices) > 1:
        # Calculate increment between consecutive timesteps
        increments = [timestep_indices[i+1] - timestep_indices[i] for i in range(len(timestep_indices)-1)]
        # Use the most common increment (in case there are variations)
        timestep_increment = max(set(increments), key=increments.count) if increments else 1
    else:
        timestep_increment = 1
    
    # Try to get time_step_size from XML
    time_step_size = None
    if geo_dir is None:
        # Try to infer geo_dir from geometric_input_path
        # Typically: data/zeroD/set_X/tree_Y/geometric_input.json
        # XML would be at: data/threeD/set_X/tree_Y/fluid_simulation_0-0.xml
        geo_dir = os.path.dirname(geometric_input_path)
        # Try to find threeD directory
        parts = geo_dir.split(os.sep)
        if 'zeroD' in parts:
            idx = parts.index('zeroD')
            parts[idx] = 'threeD'
            geo_dir = os.sep.join(parts)
    
    xml_path = os.path.join(geo_dir, 'fluid_simulation_0-0.xml')
    if os.path.exists(xml_path):
        sim_params = parse_simulation_xml(xml_path)
        if sim_params and 'time_step_size' in sim_params:
            time_step_size = sim_params['time_step_size']
            print(f"  Found XML time_step_size: {time_step_size:.6f} s")
            print(f"  Timestep increment in solution: {timestep_increment}")
    # If we are using a VMR type geometry, get the dt from dictionary
    elif 'VMR' in centerline_soln_path:
        geometry_name = centerline_soln_path.split('/')[-2]
        VMR_time_step_dict = {'0063_1001': 0.00041666875}
        time_step_size = VMR_time_step_dict[geometry_name]
        print(f"  Found time_step_size in VMR dictionary: {time_step_size:.6f} s")
    else:
        raise ValueError("Could not find time_step_size in XML or VMR dictionary")
    
    # Extract time array (assume uniform time steps)
    num_timesteps = len(pressure_timesteps)
    times = np.linspace(0.0, 1.0, num_timesteps)  # Normalized time
    obs_len = len(times)
    
    # Extract data at boundary points and junctions
    branch_id = centerline_data.get('BranchId', None)
    gid = centerline_data.get('GlobalNodeId', None)
    
    # Find inlet (GID == 0) and outlets
    inlet_idx = None
    outlet_indices = []
    
    if gid is not None:
        for i in range(len(gid)):
            if gid[i] == 0:
                inlet_idx = i
            elif gid[i] > 0:  # Outlet
                outlet_indices.append(i)
    
    # Calculate dt from XML timestep size and solution increment
    if time_step_size is not None:
        dt = time_step_size * timestep_increment
        print(f"  Calculated dt for derivatives: {dt:.6f} s (time_step_size * increment = {time_step_size:.6f} * {timestep_increment})")

    else:
        # Fallback: use normalized time difference
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        print(f"  Warning: Could not find XML time_step_size, using normalized time difference: {dt:.6f}")
    
    # Extract observations
    observations = {"y": {}, "dy": {}}
    
    # Helper function to extract and refine data at a point
    def extract_at_point(point_idx, times, dt, deriv_method='backward', verbose=False):
        if point_idx is None or point_idx >= len(branch_id):
            return None, None, None, None
        
        pressure_data = np.array([centerline_data[pt][point_idx] 
                               for pt in pressure_timesteps])
        flow_data = np.array([centerline_data[ft][point_idx] 
                           for ft in flow_timesteps])
        
        # Keep pressure in original units (dynes/cm^2) - conversion to mmHg done in visualization
        # Use data directly without refinement
        pressure_refined = pressure_data.tolist()
        flow_refined = flow_data.tolist()
        
        # Compute derivatives
        if len(times) > 1:
            if deriv_method == 'forward':
                # Forward differences: df/dt ≈ (f[i+1] - f[i]) / dt
                # For last point, use backward difference
                if verbose:
                    print(f"  Using forward difference method for derivatives")
                pressure_der = np.zeros_like(pressure_data)
                flow_der = np.zeros_like(flow_data)
                
                # Forward differences for all but last point
                pressure_der[:-1] = (pressure_data[1:] - pressure_data[:-1]) / dt
                flow_der[:-1] = (flow_data[1:] - flow_data[:-1]) / dt
                
                # Backward difference for last point
                if len(pressure_data) > 1:
                    pressure_der[-1] = (pressure_data[-1] - pressure_data[-2]) / dt
                    flow_der[-1] = (flow_data[-1] - flow_data[-2]) / dt
                
                pressure_der = pressure_der.tolist()
                flow_der = flow_der.tolist()
            elif deriv_method == 'backward':
                # Backward differences: df/dt ≈ (f[i] - f[i-1]) / dt
                # For first point, use forward difference
                if verbose:
                    print(f"  Using backward difference method for derivatives")
                pressure_der = np.zeros_like(pressure_data)
                flow_der = np.zeros_like(flow_data)
                
                # Backward differences for all but first point
                pressure_der[1:] = (pressure_data[1:] - pressure_data[:-1]) / dt
                flow_der[1:] = (flow_data[1:] - flow_data[:-1]) / dt
                
                # Forward difference for first point
                if len(pressure_data) > 1:
                    pressure_der[0] = (pressure_data[1] - pressure_data[0]) / dt
                    flow_der[0] = (flow_data[1] - flow_data[0]) / dt
                
                pressure_der = pressure_der.tolist()
                flow_der = flow_der.tolist()
            else:
                # Central differences (default): uses np.gradient
                if verbose:
                    print(f"  Using central difference method for derivatives")
                pressure_der = np.gradient(pressure_data, dt).tolist()
                flow_der = np.gradient(flow_data, dt).tolist()
        else:
            pressure_der = [0.0] * len(pressure_refined)
            flow_der = [0.0] * len(flow_refined)
        
        return pressure_refined, pressure_der, flow_refined, flow_der
    
    # Extract observations at boundaries (inlet and outlets)
    # Inflow BC
    if inlet_idx is not None:
        p_ref, p_der, f_ref, f_der = extract_at_point(inlet_idx, times, dt, derivative_method, verbose=True)
        if p_ref is not None:
            observations["y"][f"pressure:INFLOW:branch0_seg0"] = p_ref[start_idx:end_idx]
            observations["dy"][f"pressure:INFLOW:branch0_seg0"] = p_der[start_idx:end_idx]
            observations["y"][f"flow:INFLOW:branch0_seg0"] = f_ref[start_idx:end_idx]
            observations["dy"][f"flow:INFLOW:branch0_seg0"] = f_der[start_idx:end_idx]
    
    # Outlet BCs - find vessels connected to outlets
    for vessel in vessels:
        if 'boundary_conditions' in vessel:
            bc_outlet = vessel['boundary_conditions'].get('outlet')
            if bc_outlet:
                # Find outlet point for this vessel's branch
                vessel_branch = int(vessel['vessel_name'].split('_')[0].replace('branch', ''))
                
                # Find the last point (outlet) of this branch
                outlet_point_idx = None
                # First try to find by matching branch_id and checking if it's an outlet
                for i in range(len(branch_id) - 1, -1, -1):  # Search backwards to find last point
                    if branch_id[i] == vessel_branch:
                        # Check if this is an outlet (either in outlet_indices or last point of branch)
                        if i in outlet_indices or gid is None:
                            outlet_point_idx = i
                            break
                
                # If not found, use the last point of the branch as fallback
                if outlet_point_idx is None:
                    for i in range(len(branch_id) - 1, -1, -1):
                        if branch_id[i] == vessel_branch:
                            outlet_point_idx = i
                            break
                
                if outlet_point_idx is not None:
                    p_ref, p_der, f_ref, f_der = extract_at_point(outlet_point_idx, times, dt, derivative_method, verbose=False)
                    if p_ref is not None:
                        observations["y"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = p_ref[start_idx:end_idx]
                        observations["dy"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = p_der[start_idx:end_idx]
                        observations["y"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = f_ref[start_idx:end_idx]
                        observations["dy"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = f_der[start_idx:end_idx]
                    else:
                        # Create zero observations if extraction failed
                        print(f"  Warning: Could not extract observations for {vessel['vessel_name']}:{bc_outlet}, using zeros")
                        raise ValueError(f"Could not extract observations for {vessel['vessel_name']}:{bc_outlet}")
                        # zero_obs = [0.0] * obs_len
                        # observations["y"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                        # observations["dy"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                        # observations["y"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                        # observations["dy"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                else:
                    # Create zero observations if outlet point not found
                    print(f"  Warning: Could not find outlet point for {vessel['vessel_name']}:{bc_outlet}, using zeros")
                    raise ValueError(f"Could not find outlet point for {vessel['vessel_name']}:{bc_outlet}")
                    # zero_obs = [0.0] * obs_len
                    # observations["y"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                    # observations["dy"][f"pressure:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                    # observations["y"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
                    # observations["dy"][f"flow:{vessel['vessel_name']}:{bc_outlet}"] = zero_obs
    
    # Extract observations at junctions
    # For each junction, extract data for vessels connected to it
    for junc in junctions:
        junc_name = junc.get('junction_name', '')
        inlet_vessel_ids = junc.get('inlet_vessels', [])
        outlet_vessel_ids = junc.get('outlet_vessels', [])
        
        # For inlet vessels: format is "flow:vessel_name:junction_name"
        for vessel_id in inlet_vessel_ids:
            if vessel_id < len(vessels):
                vessel = vessels[vessel_id]
                vessel_name = vessel['vessel_name']
                vessel_branch = int(vessel_name.split('_')[0].replace('branch', ''))
                
                # Find a point on this vessel near the junction (use last point of branch)
                # This is approximate - ideally we'd find the exact junction point
                for i in range(len(branch_id) - 1, -1, -1):
                    if branch_id[i] == vessel_branch:
                        p_ref, p_der, f_ref, f_der = extract_at_point(i, times, dt, derivative_method, verbose=False)
                        if p_ref is not None:
                            observations["y"][f"pressure:{vessel_name}:{junc_name}"] = p_ref[start_idx:end_idx]
                            observations["dy"][f"pressure:{vessel_name}:{junc_name}"] = p_der[start_idx:end_idx]
                            observations["y"][f"flow:{vessel_name}:{junc_name}"] = f_ref[start_idx:end_idx]
                            observations["dy"][f"flow:{vessel_name}:{junc_name}"] = f_der[start_idx:end_idx]
                        break
        
        # For outlet vessels: format is "flow:junction_name:vessel_name"
        for vessel_id in outlet_vessel_ids:
            if vessel_id < len(vessels):
                vessel = vessels[vessel_id]
                vessel_name = vessel['vessel_name']
                vessel_branch = int(vessel_name.split('_')[0].replace('branch', ''))
                
                # Find a point on this vessel near the junction (use first point of branch)
                for i in range(len(branch_id)):
                    if branch_id[i] == vessel_branch:
                        p_ref, p_der, f_ref, f_der = extract_at_point(i, times, dt, derivative_method, verbose=False)
                        if p_ref is not None:
                            observations["y"][f"pressure:{junc_name}:{vessel_name}"] = p_ref[start_idx:end_idx]
                            observations["dy"][f"pressure:{junc_name}:{vessel_name}"] = p_der[start_idx:end_idx]
                            observations["y"][f"flow:{junc_name}:{vessel_name}"] = f_ref[start_idx:end_idx]
                            observations["dy"][f"flow:{junc_name}:{vessel_name}"] = f_der[start_idx:end_idx]
                            
                        break

    return observations

def find_inlet_outlet_caps(geo_dir):
    """
    Find inlet and outlet cap files from geometry directory.
    Looks in both svVascularize format (mesh/fluid_msh_0/mesh-surfaces) 
    and SimVascular format (mesh-complete/mesh-surfaces).
    The largest cap is the inlet, all others are outlets.
    
    Args:
        geo_dir: Path to geometry directory
        
    Returns:
        inlet_cap: Inlet cap filename
        outlet_caps: List of outlet cap filenames
        mesh_surfaces_dir: Path to mesh-surfaces directory
    """
    import vtk
    
    # Try svVascularize format first (mesh/fluid_msh_0/mesh-surfaces)
    mesh_dir = os.path.join(geo_dir, 'mesh', 'fluid_msh_0')
    mesh_surfaces_dir = os.path.join(mesh_dir, 'mesh-surfaces')
    
    # If not found, try SimVascular format (mesh-complete/mesh-surfaces)
    if not os.path.exists(mesh_surfaces_dir):
        mesh_complete_dir = os.path.join(geo_dir, 'mesh-complete')
        mesh_surfaces_dir = os.path.join(mesh_complete_dir, 'mesh-surfaces')
        
        # Check for inlet_cap directory in SimVascular format
        inlet_cap_dir = os.path.join(mesh_complete_dir, 'inlet_cap')
        if os.path.exists(inlet_cap_dir):
            inlet_files = [f for f in os.listdir(inlet_cap_dir) if f.endswith('.vtp')]
            if inlet_files:
                inlet_cap_from_dir = "cap_" + inlet_files[0]
            else:
                inlet_cap_from_dir = None
        else:
            inlet_cap_from_dir = None
    else:
        inlet_cap_from_dir = None
    
    if not os.path.exists(mesh_surfaces_dir):
        raise FileNotFoundError(
            f"mesh-surfaces directory not found. Tried:\n"
            f"  - {os.path.join(geo_dir, 'mesh', 'fluid_msh_0', 'mesh-surfaces')}\n"
            f"  - {os.path.join(geo_dir, 'mesh-complete', 'mesh-surfaces')}"
        )
    
    cap_files = [f for f in os.listdir(mesh_surfaces_dir) if f.startswith('cap_') and f.endswith('.vtp')]
    
    if not cap_files:
        raise ValueError(f"No cap files found in {mesh_surfaces_dir}")
    
    # Calculate areas to determine inlet (largest cap)
    cap_areas = []
    for cap_file in cap_files:
        cap_path = os.path.join(mesh_surfaces_dir, cap_file)
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(cap_path)
        reader.Update()
        polydata = reader.GetOutput()
        
        mass_props = vtk.vtkMassProperties()
        mass_props.SetInputData(polydata)
        mass_props.Update()
        area = mass_props.GetSurfaceArea()
        
        cap_areas.append((cap_file, area))
    
    # Sort by area (largest first)
    cap_areas.sort(key=lambda x: x[1], reverse=True)
    
    # Use provided inlet_cap if found, otherwise use largest
    if inlet_cap_from_dir and inlet_cap_from_dir in [c[0] for c in cap_areas]:
        inlet_cap_name = inlet_cap_from_dir
        outlet_caps = [c[0] for c in cap_areas if c[0] != inlet_cap_from_dir]
    else:
        inlet_cap_name = cap_areas[0][0]
        outlet_caps = [c[0] for c in cap_areas[1:]]
    
    print(f"  Inlet cap: {inlet_cap_name} (area: {cap_areas[0][1]:.6f})")
    print(f"  Found {len(outlet_caps)} outlet caps")
    print(f"  Mesh surfaces directory: {mesh_surfaces_dir}")
    
    return inlet_cap_name, outlet_caps, mesh_surfaces_dir

def find_inlet_outlet_caps_from_centerline(centerline_path, geometric_input_path=None):
    """
    Find inlet and outlet caps from centerline data.
    Uses branchId 0 as the inlet, and identifies terminal branches as outlets.
    Names outlet caps based on outlet vessel names.
    
    Args:
        centerline_path: Path to centerline VTP file (or 1D solution VTP)
        geometric_input_path: Optional path to geometric input JSON (for vessel names)
        
    Returns:
        inlet_cap: Inlet cap filename (e.g., "cap_branch0_seg0.vtp")
        outlet_caps: List of outlet cap filenames (e.g., ["cap_branch1_seg0.vtp", ...])
        mesh_surfaces_dir: None (not applicable for centerline-based workflow)
    """
    # Read centerline data
    centerline_data, _ = read_centerline_vtp(centerline_path)
    
    branch_id = centerline_data.get('BranchId', None)
    if branch_id is None:
        raise ValueError("BranchId array not found in centerline")
    
    # Get unique branches
    unique_branches = np.unique(branch_id)
    
    # Branch 0 is the inlet
    if 0 not in unique_branches:
        raise ValueError("BranchId 0 not found in centerline (expected inlet branch)")
    
    inlet_cap = "cap_branch0_seg0.vtp"
    
    # Identify terminal branches (outlets)
    # A terminal branch is one that doesn't have any downstream branches
    # We can use BifurcationId if available, or identify by checking branch connectivity
    
    outlet_caps = []
    
    if geometric_input_path and os.path.exists(geometric_input_path):
        # Use geometric input to identify terminal vessels
        with open(geometric_input_path, 'r') as f:
            geo_input = json.load(f)
        
        vessels = geo_input.get('vessels', [])
        junctions = geo_input.get('junctions', [])
        
        # Find all vessels that are outlets of junctions
        outlet_vessel_indices = set()
        for junc in junctions:
            outlet_vessel_indices.update(junc.get('outlet_vessels', []))
        
        # Terminal vessels are those that are not outlets of any junction
        terminal_vessel_indices = set(range(len(vessels))) - outlet_vessel_indices
        
        # Create cap names for terminal vessels
        for vessel_idx in terminal_vessel_indices:
            if vessel_idx < len(vessels):
                vessel_name = vessels[vessel_idx].get('vessel_name', f'branch{vessel_idx}_seg0')
                # Extract branch index from vessel name
                if vessel_name.startswith('branch'):
                    branch_str = vessel_name.split('_')[0]  # "branch0"
                    branch_idx = int(branch_str.replace('branch', ''))
                    if branch_idx != 0:  # Skip inlet
                        cap_name = f"cap_{vessel_name}.vtp"
                        outlet_caps.append(cap_name)
    else:
        # Use BifurcationId to identify terminal branches if available
        # Terminal branches are those that don't appear as inlet to any bifurcation
        bifurcation_id = centerline_data.get('BifurcationId', None)
        
        if bifurcation_id is not None:
            # Find branches that are inlets to bifurcations
            inlet_branches = set()
            for i, bif_id in enumerate(bifurcation_id):
                if bif_id >= 0:  # Valid bifurcation ID
                    # The branch at this point is an inlet to a bifurcation
                    inlet_branches.add(branch_id[i])
            
            # Terminal branches are those not in inlet_branches (except branch 0)
            for branch_idx in unique_branches:
                if branch_idx != 0 and branch_idx not in inlet_branches:
                    cap_name = f"cap_branch{branch_idx}_seg0.vtp"
                    outlet_caps.append(cap_name)
        else:
            # Fallback: use all branches except 0 as outlets
            # This is a simplified approach - in practice, you'd want to identify terminal branches
            for branch_idx in unique_branches:
                if branch_idx != 0:
                    cap_name = f"cap_branch{branch_idx}_seg0.vtp"
                    outlet_caps.append(cap_name)
    
    print(f"  Inlet cap: {inlet_cap} (branchId 0)")
    print(f"  Found {len(outlet_caps)} outlet caps: {outlet_caps}")
    
    return inlet_cap, outlet_caps, None

def identify_junctions_and_bcs(centerline_data, vessels):
    """
    Identify junctions and boundary conditions from centerline topology.
    
    Args:
        centerline_data: Dictionary of centerline arrays
        vessels: List of vessel dictionaries
        
    Returns:
        junctions: List of junction dictionaries
        boundary_conditions: List of BC dictionaries
        vessel_bc_map: Mapping from vessels to BCs
    """
    branch_id = centerline_data.get('BranchId', None)
    points = centerline_data['Points']
    cells = centerline_data.get('Cells', [])
    
    # Build connectivity graph
    branch_connections = defaultdict(lambda: {'inlets': [], 'outlets': []})
    
    # Find branch endpoints
    branch_endpoints = defaultdict(lambda: {'start': None, 'end': None})
    for i, bid in enumerate(branch_id):
        if branch_endpoints[bid]['start'] is None:
            branch_endpoints[bid]['start'] = i
        branch_endpoints[bid]['end'] = i
    
    # Find connections between branches (junctions)
    # This is simplified - in practice, you'd need to check spatial proximity
    # or use BifurcationId if available
    junctions = []
    junction_id = 0
    
    # Group vessels by branch
    branch_to_vessel = {}
    for i, vessel in enumerate(vessels):
        branch_name = vessel['vessel_name'].split('_')[0]  # e.g., "branch0"
        branch_idx = int(branch_name.replace('branch', ''))
        branch_to_vessel[branch_idx] = i
    
    # Simple junction detection: assume tree structure
    # Branch 0 is typically the root (inlet)
    # Other branches connect to it or to each other
    junction_vessels = defaultdict(lambda: {'inlets': [], 'outlets': []})
    
    # Find inlet (branch 0)
    if 0 in branch_to_vessel:
        inlet_vessel_id = branch_to_vessel[0]
        # Assume branch 0 connects to branches 1 and 2 (or more)
        for branch_idx in sorted(branch_to_vessel.keys()):
            if branch_idx == 0:
                continue
            if branch_idx <= 2:  # First two branches after root
                junction_vessels[junction_id]['inlets'].append(inlet_vessel_id)
                junction_vessels[junction_id]['outlets'].append(branch_to_vessel[branch_idx])
        junction_id += 1
    
    # Create junction dictionaries
    junctions = []
    for jid, conn in junction_vessels.items():
        if len(conn['inlets']) > 0 and len(conn['outlets']) > 0:
            junctions.append({
                "inlet_vessels": conn['inlets'],
                "junction_name": f"J{jid}",
                "junction_type": "NORMAL_JUNCTION",
                "outlet_vessels": conn['outlets']
            })
    
    # Create boundary conditions
    # Inlet BC
    boundary_conditions = [{
        "bc_name": "INFLOW",
        "bc_type": "FLOW",
        "bc_values": {
            "Q": [0.0, 0.0],  # Will be replaced with actual data
            "t": [0.0, 1.0]
        }
    }]
    
    # Outlet BCs (one per terminal branch)
    outlet_counter = 1
    vessel_bc_map = {}
    
    # Find terminal vessels (not connected to any junction as outlet)
    all_outlet_vessels = set()
    for junc in junctions:
        all_outlet_vessels.update(junc['outlet_vessels'])
    
    for i, vessel in enumerate(vessels):
        if i not in all_outlet_vessels:
            # This is a terminal vessel
            bc_name = f"OUT{outlet_counter}"
            boundary_conditions.append({
                "bc_name": bc_name,
                "bc_type": "RESISTANCE",
                "bc_values": {
                    "Pd": 0.0,
                    "R": 1.0
                }
            })
            vessel_bc_map[bc_name] = {
                "name": vessel['vessel_name'],
                "pressure": "pressure",
                "flow": "flow"
            }
            # Add BC to vessel
            vessel['boundary_conditions'] = {"outlet": bc_name}
            outlet_counter += 1
    
    # Add inlet BC to first vessel
    if vessels:
        vessels[0]['boundary_conditions'] = {"inlet": "INFLOW"}
        vessel_bc_map["INFLOW"] = {
            "name": vessels[0]['vessel_name'],
            "pressure": "pressure",
            "flow": "flow"
        }
    
    return junctions, boundary_conditions, vessel_bc_map

def extract_vessel_segments(centerline_data, centerline_polydata):
    """
    Extract vessel segments from centerline data.
    Groups points by BranchId and creates segments.
    
    Args:
        centerline_data: Dictionary of centerline arrays
        centerline_polydata: VTK polydata object
        
    Returns:
        List of vessel segment dictionaries
    """
    branch_id = centerline_data.get('BranchId', None)
    if branch_id is None:
        raise ValueError("BranchId array not found in centerline")
    
    area = centerline_data.get('CenterlineSectionArea', None)
    if area is None:
        raise ValueError("CenterlineSectionArea array not found in centerline")
    
    path = centerline_data.get('Path', None)
    if path is None:
        # Calculate path from points if not available
        points = centerline_data['Points']
        path = np.zeros(len(points))
        for i in range(1, len(points)):
            path[i] = path[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    # Group points by branch
    branches = defaultdict(list)
    for i in range(len(branch_id)):
        branches[branch_id[i]].append(i)
    
    # Sort points within each branch by path
    for branch_idx in branches:
        branches[branch_idx].sort(key=lambda i: path[i])
    
    # Create vessel segments
    vessels = []
    vessel_id = 0
    
    for branch_idx in sorted(branches.keys()):
        branch_points = branches[branch_idx]
        if len(branch_points) < 2:
            continue
        
        # Calculate segment properties
        segment_length = 0.0
        segment_areas = []
        segment_radii = []
        
        for i in range(len(branch_points) - 1):
            p1_idx = branch_points[i]
            p2_idx = branch_points[i + 1]
            
            # Length
            p1 = centerline_data['Points'][p1_idx]
            p2 = centerline_data['Points'][p2_idx]
            segment_length += np.linalg.norm(p2 - p1)
            
            # Area and radius
            if area[p1_idx] > 0:
                segment_areas.append(area[p1_idx])
                segment_radii.append(np.sqrt(area[p1_idx] / np.pi))
        
        if len(segment_areas) == 0:
            continue
        
        # Average properties
        mean_area = np.mean(segment_areas)
        mean_radius = np.mean(segment_radii)
        
        # Calculate geometric 0D parameters
        # Poiseuille resistance: R = 8*mu*L / (pi*r^4)
        R_poiseuille = 8 * MU * segment_length / (np.pi * mean_radius**4) if mean_radius > 0 else 0.0
        
        # Capacitance: C = 3*pi*r^3*L / (2*E*h) where E*h is wall stiffness
        # Using typical value: E*h = 1e6 dyn/cm^2 (approximate)
        E_h = 1e6  # Wall stiffness (dyn/cm^2)
        C = 3 * np.pi * mean_radius**3 * segment_length / (2 * E_h) if mean_radius > 0 else 0.0
        
        # Inductance: L = rho*L / A
        L = RHO * segment_length / mean_area if mean_area > 0 else 0.0
        
        vessel = {
            "vessel_id": vessel_id,
            "vessel_length": float(segment_length),
            "vessel_name": f"branch{branch_idx}_seg0",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "R_poiseuille": float(R_poiseuille),
                "C": float(C),
                "L": float(L),
                "stenosis_coefficient": 0.0
            }
        }
        
        vessels.append(vessel)
        vessel_id += 1
    
    return vessels

def read_centerline_vtp(centerline_path):
    """
    Read centerline VTP file and extract geometric data.
    
    Args:
        centerline_path: Path to centerline VTP file
        
    Returns:
        Dictionary with centerline data arrays
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(centerline_path)
    reader.Update()
    centerline = reader.GetOutput()
    
    # Extract point data arrays
    point_data = centerline.GetPointData()
    arrays = {}
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        arrays[array.GetName()] = v2n(array)
    
    # Extract points
    points = v2n(centerline.GetPoints().GetData())
    arrays['Points'] = points
    
    # Extract connectivity
    cells = []
    for i in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(i)
        if cell.GetNumberOfPoints() == 2:  # Line segment
            cells.append([cell.GetPointId(0), cell.GetPointId(1)])
    arrays['Cells'] = cells
    
    return arrays, centerline

def timestep_from_1D(centerline_soln_path, geo_dir):
    """
    Extract timestep information from 1D centerline solution and XML file.
    
    Args:
        centerline_soln_path: Path to 1D centerline solution VTP file
        geo_dir: Geometry directory containing XML file
        
    Returns:
        tuple: (num_timesteps, time_step_size, bc_time) where:
            - num_timesteps: Number of timesteps in the solution
            - time_step_size: Time step size from XML (or None if not found)
            - bc_time: List of time values (or None if time_step_size not found)
    """
    # Read centerline solution
    centerline_data, _ = read_centerline_vtp(centerline_soln_path)
    flow_timesteps = [key for key in centerline_data.keys() if key.startswith('velocity_') or key.startswith('flow_')]
    
    def extract_timestep(name):
        try:
            return int(name.split('_')[-1])
        except:
            return 0
    
    flow_timesteps.sort(key=extract_timestep)
    num_timesteps = len(flow_timesteps)
    time_increment = extract_timestep(flow_timesteps[1]) - extract_timestep(flow_timesteps[0])

    # Get timestep size from XML (if geo_dir is available)
    if geo_dir is not None and os.path.exists(geo_dir):
        xml_path = os.path.join(geo_dir, 'fluid_simulation_0-0.xml')

        if not os.path.exists(xml_path):
            # If no XML found, use default
            raise ValueError("No XML file found in {geo_dir}")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find GeneralSimulationParameters
        gen_params = root.find('GeneralSimulationParameters')
        if gen_params is None:
            gen_params = root.find('General_Parameters')
        
        threeD_time_step_size = None
        if gen_params is not None:
            time_step_size_elem = gen_params.find('Time_step_size')
            if time_step_size_elem is not None:
                threeD_time_step_size = float(time_step_size_elem.text)

    # if we are using a VMR type geometry, get the dt from dictionary
    if 'VMR' in centerline_soln_path:
        geometry_name = centerline_soln_path.split('/')[-2]
        VMR_time_step_dict = {'0063_1001': 0.00041666875}
        threeD_time_step_size = VMR_time_step_dict[geometry_name]
        print(f"  Found time_step_size in VMR dictionary: {threeD_time_step_size:.6f} s")

    if threeD_time_step_size is None:
        raise ValueError("Could not extract time step size from XML")
    else:
        time_step_size = time_increment * threeD_time_step_size
    
    return time_step_size

def parse_simulation_xml(xml_path):
    """
    Parse fluid_simulation XML file to extract simulation parameters and inlet BC name.
    
    Returns:
        dict with 'num_time_steps', 'time_step_size', and 'inlet_bc_name'
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find GeneralSimulationParameters
        gen_params = root.find('GeneralSimulationParameters')
        if gen_params is None:
            return None
        
        num_time_steps = gen_params.find('Number_of_time_steps')
        time_step_size = gen_params.find('Time_step_size')
        
        if num_time_steps is None or time_step_size is None:
            return None
        
        # Find inlet boundary condition (Dirichlet with Unsteady time dependence)
        inlet_bc_name = None
        add_equation = root.find('Add_equation')
        if add_equation is not None:
            for bc in add_equation.findall('Add_BC'):
                bc_type = bc.find('Type')
                time_dep = bc.find('Time_dependence')
                
                if (bc_type is not None and bc_type.text == 'Dirichlet' and
                    time_dep is not None and time_dep.text == 'Unsteady'):
                    inlet_bc_name = bc.get('name')
                    break
        
        result = {
            'num_time_steps': int(num_time_steps.text),
            'time_step_size': float(time_step_size.text)
        }
        
        if inlet_bc_name:
            result['inlet_bc_name'] = inlet_bc_name
        
        return result
    except Exception as e:
        print(f"Warning: Could not parse simulation XML: {e}")
        return None

def read_flow_file(flow_path):
    """
    Read flow file (.flow format).
    
    Format:
    First line: number_of_points (optional)
    Subsequent lines: time    flow_value
    
    Returns:
        tuple (times, flows) as lists
    """
    times = []
    flows = []
    
    try:
        with open(flow_path, 'r') as f:
            lines = f.readlines()
            
            # Skip first line if it's just a number
            start_idx = 0
            if len(lines) > 0:
                first_line = lines[0].strip().split()
                if len(first_line) == 1 or (len(first_line) == 2 and first_line[0].isdigit()):
                    start_idx = 1
            
            for line in lines[start_idx:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        time = float(parts[0])
                        flow = float(parts[1])
                        times.append(time)
                        flows.append(flow)
                    except ValueError:
                        continue
        
        return times, flows
    except Exception as e:
        print(f"Warning: Could not read flow file {flow_path}: {e}")
        return None, None

def convert_numpy_to_list(obj):
    """
    Recursively convert numpy arrays to lists for JSON serialization.
    
    Args:
        obj: Object that may contain numpy arrays (dict, list, numpy array, or other)
        
    Returns:
        Object with all numpy arrays converted to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


def verify_inlet_flow_matches_bc(input_data, results, inlet_vessel_name='branch0_seg0'):
    """
    Verify that the inlet flow in simulation results matches the boundary condition.
    
    Args:
        input_data: Input JSON data (contains BC)
        results: Simulation results dictionary or CSV path
        inlet_vessel_name: Name of inlet vessel
        
    Returns:
        True if matches (within tolerance), False otherwise
    """
    # Extract BC flow values
    bc_times = None
    bc_flows = None
    for bc in input_data.get('boundary_conditions', []):
        if bc.get('bc_name') == 'INFLOW':
            bc_values = bc.get('bc_values', {})
            bc_times = bc_values.get('t', [])
            bc_flows = bc_values.get('Q', [])
            break
    
    if bc_times is None or bc_flows is None:
        print("  Warning: Could not find INFLOW BC in input data")
        return False
    
    # Extract inlet flow from results
    if isinstance(results, str):
        # Results is a CSV path, read it
        import csv
        inlet_flows = []
        result_times = []
        with open(results, 'r') as f:
            reader = csv.DictReader(f)
            # Check which column name is used for vessel identifier by reading fieldnames
            fieldnames = reader.fieldnames
            if fieldnames is None:
                print("  Warning: Could not read CSV fieldnames")
                return False
            
            # Determine vessel identifier column name
            vessel_id_col = None
            if 'location' in fieldnames:
                vessel_id_col = 'location'
            elif 'name' in fieldnames:
                vessel_id_col = 'name'
            else:
                print("  Warning: Could not find 'location' or 'name' column in CSV")
                print(f"    Available columns: {fieldnames}")
                return False
            
            # Process all rows
            for row in reader:
                if row.get(vessel_id_col) == inlet_vessel_name:
                    result_times.append(float(row['time']))
                    inlet_flows.append(float(row['flow_in']))
    else:
        # Results is a dictionary from pysvzerod
        names = results.get('name', [])
        times = results.get('time', [])
        flow_in = results.get('flow_in', [])
        
        inlet_flows = []
        result_times = []
        for i, name in enumerate(names):
            if name == inlet_vessel_name:
                result_times.append(times[i] if i < len(times) else 0.0)
                inlet_flows.append(flow_in[i] if i < len(flow_in) else 0.0)
    
    if not inlet_flows:
        print(f"  Warning: Could not find inlet flow for vessel '{inlet_vessel_name}' in results")
        return False
    
    # Interpolate BC flows to match result times
    if HAS_SCIPY_INTERP:
        from scipy.interpolate import interp1d
        try:
            interp_func = interp1d(bc_times, bc_flows, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            bc_flows_interp = [float(interp_func(t)) for t in result_times]
        except:
            bc_flows_interp = np.interp(result_times, bc_times, bc_flows).tolist()
    else:
        bc_flows_interp = np.interp(result_times, bc_times, bc_flows).tolist()
    
    # Compare
    max_diff = 0.0
    max_diff_time = None
    tolerance = 1e-3  # Allow small numerical differences
    
    for i, (result_flow, bc_flow) in enumerate(zip(inlet_flows, bc_flows_interp)):
        diff = abs(result_flow - bc_flow)
        if diff > max_diff:
            max_diff = diff
            max_diff_time = result_times[i] if i < len(result_times) else None
    
    if max_diff > tolerance:
        print(f"  ⚠ WARNING: Inlet flow does not match BC!")
        print(f"    Maximum difference: {max_diff:.6f} cm³/s at t={max_diff_time:.3f}")
        print(f"    BC flow range: [{min(bc_flows):.2f}, {max(bc_flows):.2f}] cm³/s")
        print(f"    Result flow range: [{min(inlet_flows):.2f}, {max(inlet_flows):.2f}] cm³/s")
        print(f"    This may indicate convergence issues or BC application problems")
        return False
    else:
        print(f"  ✓ Verified: Inlet flow matches BC (max diff: {max_diff:.6e} cm³/s)")
        return True


def convert_simulation_results_to_csv(sim_results, output_csv_path):
    """
    Convert pysvzerod simulation results to CSV format.
    
    Args:
        sim_results: Dictionary returned by pysvzerod.simulate()
        output_csv_path: Path to save CSV file
        
    Returns:
        Path to saved CSV file
    """
    import csv
    
    # CSV format: location, time, flow_in, flow_out, pressure_in, pressure_out
    rows = []
    rows.append(["location", "time", "flow_in", "flow_out", "pressure_in", "pressure_out"])
    
    # Extract data from simulation results
    # Results are typically arrays where each index corresponds to a (vessel, time) pair
    if 'name' in sim_results and 'time' in sim_results:
        names = np.array(sim_results['name']) if isinstance(sim_results['name'], list) else sim_results['name']
        times = np.array(sim_results['time']) if isinstance(sim_results['time'], list) else sim_results['time']
        flow_in = np.array(sim_results.get('flow_in', [])) if isinstance(sim_results.get('flow_in', []), list) else sim_results.get('flow_in', np.array([]))
        flow_out = np.array(sim_results.get('flow_out', [])) if isinstance(sim_results.get('flow_out', []), list) else sim_results.get('flow_out', np.array([]))
        pressure_in = np.array(sim_results.get('pressure_in', [])) if isinstance(sim_results.get('pressure_in', []), list) else sim_results.get('pressure_in', np.array([]))
        pressure_out = np.array(sim_results.get('pressure_out', [])) if isinstance(sim_results.get('pressure_out', []), list) else sim_results.get('pressure_out', np.array([]))
        
        # Convert to numpy arrays for easier handling
        if not isinstance(names, np.ndarray):
            names = np.array(names)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        
        # Write all data points
        num_points = len(names) if len(names) > 0 else len(times)
        for i in range(num_points):
            if i < len(names) and i < len(times):
                row = [
                    str(names[i]) if i < len(names) else "unknown",
                    str(float(times[i])) if i < len(times) else "0.0",
                    str(float(flow_in[i])) if i < len(flow_in) else "0.0",
                    str(float(flow_out[i])) if i < len(flow_out) else "0.0",
                    str(float(pressure_in[i])) if i < len(pressure_in) else "0.0",
                    str(float(pressure_out[i])) if i < len(pressure_out) else "0.0"
                ]
                rows.append(row)
    
    # Write CSV file
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Simulation results saved to: {output_csv_path}")
    return output_csv_path

