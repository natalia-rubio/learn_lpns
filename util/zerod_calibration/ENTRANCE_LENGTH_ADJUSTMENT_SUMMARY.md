# Entrance Length Adjustment Logic Summary

## Overview
The entrance length adjustment modifies junction boundaries to extend a distance **EL (Entrance Length)** down each outlet vessel. EL is defined as **10 × MaximumInscribedSphereRadius** at the outlet vessel inlet (where it connects to the junction). This accounts for the entrance length effect in fluid dynamics, where flow needs a certain distance to fully develop after a junction.

## Scope
- **Only processes junctions with ≥ 2 outlet vessels** (single-outlet junctions are skipped as they represent straight connections)
- Processes each outlet vessel independently

## Calculation
For each outlet vessel:
1. Extract `MaximumInscribedSphereRadius` at the vessel inlet (junction boundary)
2. Calculate: **EL = 10.0 × radius_at_inlet**
3. Compare vessel length to EL

## Case 1: Vessel Length < EL (Short Vessel)

### Step 1: Attempt Chain Extension
If the vessel is shorter than EL, attempt to extend through a chain of vessels connected by **single-outlet junctions**:

1. **Find downstream junction** connected to the current vessel
2. **Check if downstream junction has exactly 1 outlet** (required for chain extension)
3. **Get next vessel** in the chain
4. **Check for stopping conditions:**
   - No downstream junction found
   - Downstream junction has multiple outlets (multi-outlet junction)
   - Next vessel not found
   - Next vessel has an outlet boundary condition (BC)
   - Cannot find centerline points for next vessel

### Step 2: Merge Vessels
If chain extension can continue:
- **Merge vessels**: Combine lengths, update vessel name (e.g., `branch4_seg0_1`), merge parameters:
  - **R, L, stenosis_coefficient**: Sum (resistances in series)
  - **C**: Sum (parallel capacitance)
- **Transfer boundary conditions**: If next vessel has outlet BC, transfer to merged vessel
- **Update connections**: Remove intermediate single-outlet junction, update downstream junction to use merged vessel ID
- **Update node IDs**: Inlet stays at original inlet, outlet moves to last merged vessel's outlet
- **Remove merged vessels and junctions** from active lists

### Step 3: Check if EL Reached
After each merge:
- If **merged_length ≥ EL**:
  - If merged_length > EL: Shorten merged vessel to exactly EL by moving outlet boundary
  - Mark as `extension_successful = True`
  - **Done** - junction boundary now extends EL distance
- If **merged_length < EL**: Continue chain extension (loop back to Step 1)

### Step 4: Handle Stopping Conditions
If chain extension stops **before reaching EL** (due to any stopping condition):

#### 4a. If vessels were merged (`vessels_merged = True`):
- **Convert merged vessel to connector**:
  - Rename: append `_connector` (unless name already contains "connector")
  - Set length to 0.0
  - Set parameters to minimal values: R=0, C=1e-10, L=0, stenosis_coefficient=0
  - Update node IDs: both inlet and outlet at endpoint of extended junction (merged vessel outlet)
  - **Preserve boundary conditions** (if any were transferred)

#### 4b. If no merging occurred:
- **Convert original vessel to connector**:
  - Rename: append `_connector` (unless name already contains "connector")
  - Set length to 0.0
  - Set parameters to minimal values: R=0, C=1e-10, L=0, stenosis_coefficient=0
  - Update node IDs: both inlet and outlet at vessel endpoint (endpoint of extended junction)

## Case 2: Vessel Length ≥ EL (Long Vessel)

If the vessel is longer than or equal to EL:
1. **Calculate target path**: `target_path = vessel_start + EL`
2. **Find centerline point** at or beyond target_path
3. **Update vessel**:
   - New inlet moves to target point (junction boundary extends EL distance)
   - Vessel length reduced: `new_length = original_length - EL`
   - Outlet remains at original outlet
   - Update node IDs accordingly

## Key Features

### Boundary Condition Handling
- If a vessel in the chain has an outlet boundary condition, chain extension stops
- The boundary condition is **transferred to the merged vessel**
- When converting to connector, boundary conditions are **preserved**

### Vessel Merging
- Vessels are merged **in-place** (no new vessels created)
- Merged vessel keeps the original vessel ID
- Intermediate single-outlet junctions are **removed**
- Downstream junctions are **updated** to reference the merged vessel ID

### Connector Vessels
- Connectors are created by **renaming and modifying existing vessels** (no rewiring needed)
- Connectors have length = 0.0 and minimal parameters
- Connectors preserve any boundary conditions that were present

### Node ID Tracking
- Each vessel stores `centerline_node_ids` with inlet and outlet `GlobalNodeId`s
- Node IDs are updated when:
  - Vessels are merged (outlet moves to last vessel's outlet)
  - Junction boundaries are extended (inlet moves to new boundary)
  - Vessels are converted to connectors (both inlet and outlet at vessel endpoint - the endpoint of the extended junction)

## Summary Output
The function provides a summary including:
- Number of junctions processed
- Number of vessels converted to connectors
- Final vessel and junction counts

