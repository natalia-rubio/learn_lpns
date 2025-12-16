# Function Call Order in generate_zerod_inputs.py

## Main Execution Flow

### 1. **main()** (line 1630)
   - Entry point: Parses command-line arguments, sets up file paths
   - Orchestrates the entire workflow

### 2. **Step 1: Create Geometric 0D Input**

   #### 2.1 **create_geometric_zerod_input_rom()** (line 459)
   - Creates initial 0D input using SimVascular ROM workflow
   - Calls:
     - **find_inlet_outlet_caps()** (line 374): Identifies inlet/outlet caps from mesh
     - **update_geometric_input_from_simulation()** (line 745): Updates BCs with actual simulation data
       - **parse_simulation_xml()** (line 654): Extracts time steps and inlet BC name from XML
       - **read_flow_file()** (line 703): Reads flow values from .flow file
       - If `calibration_input.json` exists, extracts inflow BC from there instead

### 3. **Step 2: Create Calibration Input**

   #### 3.1 **extract_observations_from_1d()** (line 1054)
   - Extracts pressure/flow observations from 1D centerline solution VTP
   - Calls:
     - **refine_curve()** (line 1022): Uses cubic spline to refine observation curves
   - Returns: Dictionary with observations (y, dy)

   #### 3.2 **create_calibration_input()** (line 1246)
   - Creates calibration input JSON from geometric input + observations
   - **Sets inlet BC time values**: `bc_time = np.linspace(0.0, 1.0, NUM_OBS).tolist()` (line 1263)
   - **Writes BC time values**: `bc["bc_values"]["t"] = bc_time` (line 1269)
   - Sets all element values to zero for calibration

   #### 3.3 **Main function updates geometric input** (lines 1740-1768)
   - Ensures geometric input uses same inflow BC as calibration input
   - Directly copies BC values from `calibration_input.json` to `geometric_input.json`

### 4. **Step 3: Run Calibration**

   #### 4.1 **run_calibration()** (line 1312)
   - Runs pysvzerod.calibrate() to optimize element values
   - **Preserves inflow BC**: Extracts BC from calibration input and ensures it's in calibrated output (line 1347)
   - **BC time values preserved**: `bc['bc_values'] = inflow_bc['bc_values'].copy()` (includes 't' array)

### 5. **Step 4: Run Forward Simulations**

   #### 5.1 **run_forward_simulation()** (geometric) (line 1515)
   - Runs forward simulation with geometric input
   - Calls:
     - **pysvzerod.simulate()** or **solve_casadi_unsteady()** (fallback)
     - **convert_simulation_results_to_csv()** (line 1368): Converts results to CSV format
     - **verify_inlet_flow_matches_bc()** (line 1425): Verifies inlet flow matches BC

   #### 5.2 **run_forward_simulation()** (calibrated) (line 1515)
   - Same as above, but with calibrated input

## Key Points About Time Values

1. **Time values are normalized to [0, 1]** in `create_calibration_input()`:
   ```python
   bc_time = np.linspace(0.0, 1.0, NUM_OBS).tolist()
   ```

2. **Time values are written in multiple places**:
   - `create_calibration_input()`: Line 1269 - Sets BC time for calibration input
   - `update_geometric_input_from_simulation()`: Line 816 - Copies time from calibration_input.json (if exists)
   - `update_geometric_input_from_simulation()`: Line 961 - Creates time from simulation time steps (if reading from flow file)
   - `run_calibration()`: Line 1347 - Preserves time values in calibrated output
   - Main function: Line 1761 - Copies time values from calibration_input.json to geometric_input.json

3. **The actual physical time period** is determined from the 3D simulation XML, but the 0D solver uses normalized time [0, 1].

