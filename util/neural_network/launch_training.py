
import os
import sys

# Allow running as a script: python util/neural_network/launch_training.py ...
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.neural_network.nn_model import NeuralNet
from util.neural_network.train_nn import train_nn
from util.tools.basic import load_dict


def launch_training(network_params, optimizer_params, training_params):
    network_params["output_type"] = "rri"
    
    print("Training RRI model...")
    lr_init1 = 0.1
    lr_init2 = 0.1
    lr_init3 = 0.1

    print(f"training model 1:  Linear Resistor")
    network_params["target_coef_ind"] = 0
    network_params["layer_width"] = 40
    network_params["num_layers"] = 1
    training_params["num_epochs"] = 1000
    optimizer_params["decay_rate"] = 0.8
    optimizer_params["init"] = lr_init1
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)

    optimizer_params["init"] = lr_init2
    print(f"training model 2:  Stenosis Resistor")
    network_params["target_coef_ind"] = 1
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)

    optimizer_params["init"] = lr_init3
    print(f"training model 3:  Inductor")
    network_params["target_coef_ind"] = 2
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch NN training")
    parser.add_argument("set_name", help="Set name (e.g., VMR)")
    parser.add_argument("num_geos", type=int, help="Number of geometries")
    parser.add_argument("geometry_variant", nargs="?", default="all",
                        help="Geometry variant: bifurcations, bifurcations_EL, or all (default: all)")
    parser.add_argument("--normalize", action="store_true",
                        help="Use normalized jax_arrays (loads *_normalized.pkl)")
    parser.add_argument("--split-path", default=None,
                        help="Path to train/val split pickle (default: data/split_indices/.../train_val_ind_{set_name}_num_geos_{num_geos})")
    parser.add_argument("--model-dir", default=None,
                        help="Directory to save models (default: results/models/{set_name}/{geometry_variant})")
    cli_args = parser.parse_args()

    set_name = cli_args.set_name
    num_geos = cli_args.num_geos
    geometry_variant_arg = cli_args.geometry_variant
    normalize = cli_args.normalize
    norm_suffix = "_normalized" if normalize else ""
    output_type = "rri"
    set_type = "test"

    # Determine which geometry variants to process
    if geometry_variant_arg == "all":
        geometry_variants_to_process = ["bifurcations", "bifurcations_EL"]
    else:
        geometry_variants_to_process = [geometry_variant_arg]
    
    # Process each geometry variant
    for geometry_variant in geometry_variants_to_process:
        print(f"\n{'='*80}")
        print(f"Training models for geometry variant: {geometry_variant}"
              f"{' (normalized)' if normalize else ''}")
        print(f"{'='*80}")
        
        if cli_args.split_path:
            split_path = cli_args.split_path
        else:
            split_path = f"data/split_indices/{set_name}/{geometry_variant}/{set_type}/train_val_ind_{set_name}_num_geos_{num_geos}"
        split_ind_dict = load_dict(split_path)

        network_params = {"num_input_features": 25,
                          "num_layers": 5,
                          "layer_width":100,
                          "output_type": output_type,
                          "set_name": set_name,
                          "set_type": set_type,
                          "num_geos": num_geos,
                          "data_root": "data",
                          "geometry_variant": geometry_variant,
                          "normalize": normalize,
                          "pred_mode": "m1"}
        
        training_params = {"num_epochs": 500, 
                           "batch_size": 1,
                           "train_inds": split_ind_dict["train_ind"],
                           "val_inds": split_ind_dict["val_ind"],
                           "num_offsets": split_ind_dict["num_offsets"],}
        if cli_args.model_dir:
            training_params["output_dir"] = cli_args.model_dir
        
        optimizer_params = {#"step_size": 0.0002,
                            "init" : 0.02,
                            "transition_steps": 1000,
                            "decay_rate" : 0.95}
        
        launch_training(network_params, optimizer_params, training_params)