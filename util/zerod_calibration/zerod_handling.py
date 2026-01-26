import copy

def modify_junction_types(config, junction_type):
    """
    Modify junction types in a config based on the number of outlets.
    Junctions with more than one outlet use the specified junction type.
    Junctions with only one outlet remain as NORMAL_JUNCTION.
    
    Args:
        config: Dictionary with 0D input configuration
        junction_type: String specifying junction type ('BloodVesselJunction', 'NORMAL_JUNCTION', 'DirIndepJunction', 'HybridJunction')
        
    Returns:
        Modified config dictionary
    """
    import copy
    config_modified = copy.deepcopy(config)
    
    # Update junctions based on number of outlets
    if 'junctions' in config_modified:
        for junc in config_modified['junctions']:
            # Get number of outlet vessels
            num_outlets = len(junc.get('outlet_vessels', []))
            
            # Only modify junctions with more than one outlet
            if num_outlets <= 1:
                # Keep as NORMAL_JUNCTION for single outlet
                junc['junction_type'] = 'NORMAL_JUNCTION'
                # Remove junction_values if present (not needed for NORMAL_JUNCTION)
                if 'junction_values' in junc:
                    del junc['junction_values']
                continue
            
            # Update to new junction type for multi-outlet junctions
            junc['junction_type'] = junction_type
            
            # NORMAL_JUNCTION should not have junction_values
            if junction_type == 'NORMAL_JUNCTION':
                # Remove junction_values if present (NORMAL_JUNCTION has no parameters)
                if 'junction_values' in junc:
                    del junc['junction_values']
                continue
            
            # For special junction types, we need to provide junction_values
            if junction_type in ['BloodVesselJunction', 'DirIndepJunction', 'HybridJunction']:
                num_outlets = len(junc.get('outlet_vessels', []))
                
                if num_outlets > 0:
                    # Initialize or update junction values with only the required parameters
                    if 'junction_values' not in junc:
                        junc['junction_values'] = {}
                    
                    # Determine which parameters are needed based on junction type
                    # BloodVesselJunction and DirIndepJunction: R_poiseuille, L, stenosis_coefficient
                    # HybridJunction: R_poiseuille, L, stenosis_coefficient, pressure_recovery_coefficient
                    required_params = ['R_poiseuille', 'L', 'stenosis_coefficient']
                    if junction_type == 'HybridJunction':
                        required_params.append('pressure_recovery_coefficient')
                    
                    # Set the required parameters
                    for param in required_params:
                        if param not in junc['junction_values']:
                            junc['junction_values'][param] = [0.0] * num_outlets
                        elif len(junc['junction_values'][param]) != num_outlets:
                            # Resize to match number of outlets
                            current_len = len(junc['junction_values'][param])
                            if current_len < num_outlets:
                                # Extend with zeros
                                junc['junction_values'][param].extend([0.0] * (num_outlets - current_len))
                            else:
                                # Truncate
                                junc['junction_values'][param] = junc['junction_values'][param][:num_outlets]
                    
                    # Remove C parameter (not supported by any junction type in svzerodsolver)
                    if 'C' in junc['junction_values']:
                        del junc['junction_values']['C']
                    
                    # Remove pressure_recovery_coefficient for non-HybridJunction types
                    if junction_type != 'HybridJunction' and 'pressure_recovery_coefficient' in junc['junction_values']:
                        del junc['junction_values']['pressure_recovery_coefficient']
    
    return config_modified