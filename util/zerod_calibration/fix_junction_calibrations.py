#!/usr/bin/env python3
"""
Fix junction type calibration files by regenerating them with correct parameters.
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
from generate_zerod_inputs import modify_junction_types

def fix_calibration_files(base_dir, junction_types):
    """
    Fix calibration input and output files for each junction type.
    """
    for jtype in junction_types:
        calib_input_path = os.path.join(base_dir, f'calibration_input_{jtype}.json')
        calib_output_path = os.path.join(base_dir, f'calibrated_output_{jtype}.json')
        
        # Fix calibration input
        if os.path.exists(calib_input_path):
            print(f"Fixing {calib_input_path}...")
            with open(calib_input_path, 'r') as f:
                config = json.load(f)
            
            config = modify_junction_types(config, jtype)
            
            with open(calib_input_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"  ✓ Fixed")
        
        # Fix calibrated output
        if os.path.exists(calib_output_path):
            print(f"Fixing {calib_output_path}...")
            with open(calib_output_path, 'r') as f:
                config = json.load(f)
            
            config = modify_junction_types(config, jtype)
            
            with open(calib_output_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"  ✓ Fixed")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix junction type calibration files')
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_4)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_002)')
    parser.add_argument('--data-dir', default='data/zeroD', help='Data directory')
    args = parser.parse_args()
    
    base_dir = os.path.join(args.data_dir, args.set_name, args.geo_name)
    junction_types = ['DirDepJunction', 'DirIndepJunction', 'HybridJunction']
    
    print(f"Fixing calibration files in {base_dir}...")
    fix_calibration_files(base_dir, junction_types)
    print("\nDone!")

