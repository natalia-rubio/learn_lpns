# For a set name and list of geometries, run the data processing pipeline

import os
import sys
import argparse
import csv
import sys
sys.path.append("/Users/natalia/cursor_access/learn_lpns")
from util.data_processing.inputs_from_0d_config import *

def main():
    parser = argparse.ArgumentParser(description='Run the data processing pipeline')
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_1)')
    parser.add_argument('--geometries', nargs='+', help='List of geometries (e.g., tree_000 tree_001)')
    args = parser.parse_args()

    print(f"Running data processing pipeline for {args.set_name} with geometries {args.geometries}")
    for geo in args.geometries:
        print(f"Processing geometry {geo}")

        geometric_input_path = os.path.join('data', 'zeroD', args.set_name, geo, 'bifurcations_geometric_input.json')
        X, feature_names, junction_names = load_junction_geometric_features(geometric_input_path, verbose=True)
        print(f"Loaded {len(X)} junctions with {len(feature_names)} features")
        # Save the features to a csv file
        csv_path = os.path.join('data', 'ml_inputs', args.set_name, geo, 'geometric_features.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(feature_names)
            for row in X:
                writer.writerow(row)
        print(f"Saved features to {csv_path}")

if __name__ == "__main__":
    main()
        