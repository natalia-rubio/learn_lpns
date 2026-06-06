
import os
import sys

import numpy as np

# Allow running as a script: python util/neural_network/launch_training.py ...
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from util.neural_network.nn_model import NeuralNet
from util.neural_network.train_nn import train_nn
from util.tools.basic import load_dict
from util.data_processing.generate_split_indices import load_split_for_training, resolve_flat_indices
from util.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    run_config_suffix_to_flags,
)


def _jax_arrays_path(
    data_root: str,
    set_name: str,
    geometry_variant: str,
    set_type: str,
    num_geos: int,
    run_config_suffix: str | None,
    *,
    vessel: bool,
) -> str:
    jax_filename = (
        f"jax_arrays_vessel_num_geos_{num_geos}.pkl"
        if vessel
        else f"jax_arrays_num_geos_{num_geos}.pkl"
    )
    parts = [data_root, "jax_arrays", set_name]
    if run_config_suffix:
        parts.append(run_config_suffix)
    parts.extend([geometry_variant, set_type, jax_filename])
    return os.path.join(*parts)


def _default_split_path(
    set_name: str,
    geometry_variant: str,
    set_type: str,
    num_geos: int,
    run_config_suffix: str | None,
) -> str:
    if run_config_suffix:
        return (
            f"data/split_indices/{set_name}/{run_config_suffix}/"
            f"{geometry_variant}/{set_type}/train_val_ind_{set_name}_num_geos_{num_geos}"
        )
    return (
        f"data/split_indices/{set_name}/{geometry_variant}/{set_type}/"
        f"train_val_ind_{set_name}_num_geos_{num_geos}"
    )


def _build_training_params_for_modality(
    *,
    vessel: bool,
    split_dict: dict,
    split_path: str,
    data_root: str,
    set_name: str,
    geometry_variant: str,
    set_type: str,
    num_geos: int,
    run_config_suffix: str | None,
    jax_path: str,
    output_type: str,
    symmetric_loss_eff: bool,
    gen_loss_eff: bool,
    gen_loss_scale: float,
    leaky_relu: bool,
    model_dir: str | None,
) -> tuple[dict, dict]:
    modality = "vessel" if vessel else "junction"
    jax_data = load_dict(jax_path)
    num_input_features = int(jax_data["input"].shape[1])
    train_inds = resolve_flat_indices(split_dict, modality, "train", split_path=split_path)
    val_inds = resolve_flat_indices(split_dict, modality, "val", split_path=split_path)
    num_offsets = int(split_dict.get("num_offsets", 1))

    if vessel and len(train_inds) == 0 and len(val_inds) == 0:
        train_geos = split_dict.get("train_geometries", [])
        val_geos = split_dict.get("val_geometries", [])
        raise ValueError(
            "Vessel split is empty (0 train, 0 val). "
            f"Split file {split_path} has train geos: {sorted(train_geos)}, val geos: {sorted(val_geos)}. "
            "Ensure run_data_processing was run for this set/variant and that vessel CSVs exist."
        )

    if vessel:
        train_geos = split_dict.get("train_geometries", [])
        val_geos = split_dict.get("val_geometries", [])
        print(
            f"  Vessel split: {len(train_inds)} train, {len(val_inds)} val "
            f"(train geos: {sorted(train_geos)}, val geos: {sorted(val_geos)})"
        )

    model_name_suffix = "_vessel" if vessel else ""
    network_params = {
        "num_input_features": num_input_features,
        "num_layers": 5,
        "layer_width": 100,
        "output_type": output_type,
        "set_name": set_name,
        "set_type": set_type,
        "num_geos": num_geos,
        "data_root": data_root,
        "geometry_variant": geometry_variant,
        "run_config_suffix": run_config_suffix,
        "use_leaky_relu": leaky_relu,
        "pred_mode": "m1",
        "model_name_suffix": model_name_suffix,
        "asymmetric_loss_overestimate_weight": 1.0,
        "symmetric_loss": symmetric_loss_eff,
        "gen_loss": gen_loss_eff,
        "gen_loss_scale": gen_loss_scale,
    }
    if vessel:
        network_params["jax_arrays_filename"] = os.path.basename(jax_path)

    n_train = max(len(train_inds), 1)
    training_params = {
        "num_epochs": 500,
        "batch_size": int(np.ceil(n_train / 10)),
        "train_inds": train_inds,
        "val_inds": val_inds,
        "num_offsets": 1 if vessel else num_offsets,
    }
    if vessel:
        out_dir = model_dir or os.path.join(
            "results", "models", set_name, geometry_variant + "_vessel"
        )
        training_params["output_dir"] = out_dir
    elif model_dir:
        training_params["output_dir"] = model_dir

    return network_params, training_params


def launch_training(network_params, optimizer_params, training_params):
    """Train a neural network for linear resistor, quadratic resistor, and inductor for both junctions and vessels.
    Args:
        network_params: Dictionary of network parameters.
        optimizer_params: Dictionary of optimizer parameters.
        training_params: Dictionary of training parameters.
    
    Trained models are saved in the results/models directory.

    Several hyperparameters are overridden with hardcoded values for now.
    Formal hyperparameter optimization still needed.

    coef_ind: 0 for linear resistor, 1 for quadratic resistor, 2 for inductor.
    """

    network_params["output_type"] = "rri"
    symmetric_loss = network_params.pop("symmetric_loss", False)
    
    print("Training RRI model...")

    lr_init1 = 0.01
    lr_init2 = 0.001
    lr_init3 = 0.01

    print(f"training model 1:  Linear Resistor")
    print(f"{network_params['num_input_features']} input features")
    network_params["target_coef_ind"] = 0
    optimizer_params["decay_rate"] = 0.8
    optimizer_params["init"] = lr_init1
    if network_params["model_name_suffix"] == "_vessel":
        network_params["layer_width"] = 10
        network_params["num_layers"] = 2
        training_params["num_epochs"] = 500#1000
        network_params["asymmetric_loss_overestimate_weight"] = 1.0 if symmetric_loss else 10
    else:
        network_params["layer_width"] = 10
        network_params["num_layers"] = 2
        training_params["num_epochs"] = 500#4000#5000
        network_params["asymmetric_loss_overestimate_weight"] = 1.0 if symmetric_loss else 2000
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)

    print(f"training model 2:  Stenosis Resistor")
    optimizer_params["init"] = lr_init2
    network_params["target_coef_ind"] = 1
    if network_params["model_name_suffix"] == "_vessel":
        network_params["layer_width"] = 10
        network_params["num_layers"] = 2
        training_params["num_epochs"] = 500#2000
        network_params["asymmetric_loss_overestimate_weight"] = 1.0 if symmetric_loss else 10
    else:
        network_params["layer_width"] = 10
        network_params["num_layers"] = 2
        training_params["num_epochs"] = 500#2000
        network_params["asymmetric_loss_overestimate_weight"] = 1.0 if symmetric_loss else 100
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)

    print(f"training model 3:  Inductor")
    optimizer_params["init"] = lr_init3
    network_params["target_coef_ind"] = 2
    if network_params["model_name_suffix"] == "_vessel":
        network_params["layer_width"] = 10
        network_params["num_layers"] = 2
        training_params["num_epochs"] = 500#2000
        network_params["asymmetric_loss_overestimate_weight"] = 1.0 if symmetric_loss else 1000
    else:
        network_params["layer_width"] = 20
        network_params["num_layers"] = 4
        training_params["num_epochs"] = 500#2000#5000
        network_params["asymmetric_loss_overestimate_weight"] = 1.0 if symmetric_loss else 10000
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch NN training")
    parser.add_argument("set_name", help="Set name (e.g., VMR)")
    parser.add_argument("num_geos", type=int, help="Number of geometries")
    parser.add_argument("geometry_variant", nargs="?", default=None,
                        help="Geometry variant: bifurcations, bifurcations_EL, or all (default: all). Can also be set via --geometry-variant.")
    parser.add_argument("--geometry-variant", dest="geometry_variant_flag", default=None,
                        help="Geometry variant (overrides positional if set). Use this when passing --vessel so order does not matter.")
    parser.add_argument("--split-path", default=None,
                        help="Path to train/val split pickle (default: data/split_indices/.../train_val_ind_{set_name}_num_geos_{num_geos})")
    parser.add_argument("--model-dir", default=None,
                        help="Directory to save models (default: results/models/{set_name}/{geometry_variant})")
    parser.add_argument("--vessel", action="store_true",
                        help="Train vessel NN (R/S/L per vessel); uses vessel jax arrays and same geometry-based split")
    parser.add_argument("--leaky-relu", action="store_true",
                        help="Use Leaky ReLU instead of ReLU (helps gradient flow when inputs span large ranges)")
    parser.add_argument("--print-gradients", action="store_true",
                        help="Print gradient stats for the first batch before training (for debugging)")
    parser.add_argument(
        "--verbose-epochs",
        action="store_true",
        help="Print per-epoch train/validation loss during train_nn (off by default; very chatty).",
    )
    parser.add_argument("--symmetric-loss", action="store_true", dest="symmetric_loss",
                        help="Use symmetric loss (overestimate weight 1.0 for all models). When off, per-model asymmetric weights are used.")
    parser.add_argument(
        "--run-config",
        default=DEFAULT_CLI_RUN_CONFIG,
        help="Run config suffix for path separation (e.g. stenosis_off_symmetric). "
        "jax_arrays and split_indices use .../set_name/<config>/... "
        "Use the exact suffix for jax/split paths (e.g. stenosis_off_symmetric_gen_loss). "
        "Training-only suffix _gen_loss also enables generation-weighted loss unless overridden. "
        "Default: %(default)s. Pass an empty string only for legacy layouts without a run-config subfolder.",
    )
    parser.add_argument(
        "--gen-loss",
        action="store_true",
        dest="gen_loss",
        help="Weight training loss by bifurcation generation: weight = scale / 2^generation (larger weight for smaller generation; requires generation in jax pkl).",
    )
    parser.add_argument(
        "--gen-loss-scale",
        type=float,
        default=1.0,
        dest="gen_loss_scale",
        metavar="S",
        help="Overall multiplier for gen loss weights (default: 1.0 gives weight 1/2^gen).",
    )
    cli_args = parser.parse_args()

    set_name = cli_args.set_name
    num_geos = cli_args.num_geos
    print(f"num_geos: {num_geos}")

    geometry_variant_arg = getattr(cli_args, "geometry_variant_flag", None) or cli_args.geometry_variant or "all"
    run_config_raw = (cli_args.run_config or "").strip() or None
    # Paths use the full --run-config string (e.g. ..._gen_loss is its own jax/split tree).
    data_paths_suffix = run_config_raw
    if run_config_raw:
        rc_flags = run_config_suffix_to_flags(run_config_raw)
        symmetric_loss_eff = bool(cli_args.symmetric_loss or rc_flags["symmetric_loss"])
        gen_loss_eff = bool(cli_args.gen_loss or rc_flags["gen_loss"])
    else:
        symmetric_loss_eff = bool(cli_args.symmetric_loss)
        gen_loss_eff = bool(cli_args.gen_loss)
    output_type = "rri"
    set_type = "all"
    data_root = "data"

    # Determine which geometry variants to process
    if geometry_variant_arg == "all":
        geometry_variants_to_process = ["bifurcations", "bifurcations_EL"]
    else:
        geometry_variants_to_process = [geometry_variant_arg]
    
    # Process each geometry variant
    for geometry_variant in geometry_variants_to_process:
        print(f"\n{'='*80}")
        print(f"Training {'vessel' if cli_args.vessel else 'junction'} models for geometry variant: {geometry_variant}")
        if symmetric_loss_eff:
            print("Symmetric loss: overestimate weight = 1.0 for all models")
        if gen_loss_eff:
            print(
                f"Generation-weighted loss: ON (scale={float(cli_args.gen_loss_scale):g}; "
                f"from --gen-loss and/or --run-config ..._gen_loss)"
            )
        print(f"{'='*80}")
        
        split_path = cli_args.split_path or _default_split_path(
            set_name, geometry_variant, set_type, num_geos, data_paths_suffix
        )
        split_dict = load_split_for_training(split_path)

        vessel = bool(cli_args.vessel)
        jax_path = _jax_arrays_path(
            data_root,
            set_name,
            geometry_variant,
            set_type,
            num_geos,
            data_paths_suffix,
            vessel=vessel,
        )
        network_params, training_params = _build_training_params_for_modality(
            vessel=vessel,
            split_dict=split_dict,
            split_path=split_path,
            data_root=data_root,
            set_name=set_name,
            geometry_variant=geometry_variant,
            set_type=set_type,
            num_geos=num_geos,
            run_config_suffix=data_paths_suffix,
            jax_path=jax_path,
            output_type=output_type,
            symmetric_loss_eff=symmetric_loss_eff,
            gen_loss_eff=gen_loss_eff,
            gen_loss_scale=float(getattr(cli_args, "gen_loss_scale", 1.0)),
            leaky_relu=getattr(cli_args, "leaky_relu", False),
            model_dir=cli_args.model_dir,
        )
        training_params["print_gradients"] = getattr(cli_args, "print_gradients", False)
        training_params["verbose_epochs"] = getattr(cli_args, "verbose_epochs", False)

        optimizer_params = {"init": 0.02,
                           "transition_steps": 1000,
                           "decay_rate": 0.95}

        launch_training(network_params, optimizer_params, training_params)