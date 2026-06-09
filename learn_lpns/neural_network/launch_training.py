import os
import sys
from dataclasses import dataclass

import numpy as np

# Allow running as a script: python learn_lpns/neural_network/launch_training.py ...
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from learn_lpns.data_processing.generate_split_indices import (
    load_split_for_training,
    resolve_flat_indices,
)
from learn_lpns.neural_network.nn_model import (
    L_OUTPUT_COLUMN,
    R_OUTPUT_COLUMN,
    S_OUTPUT_COLUMN,
    NeuralNet,
)
from learn_lpns.neural_network.train_nn import train_nn
from learn_lpns.tools.basic import load_dict
from learn_lpns.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    run_config_suffix_to_flags,
)


@dataclass(frozen=True)
class RriCoefTrainSpec:
    """One single-output network trained against a column of output_rri."""

    label: str
    target_output_column: int
    lr_init: float
    junction_num_layers: int
    junction_layer_width: int
    junction_asymmetric_overestimate_weight: float
    vessel_asymmetric_overestimate_weight: float


# Each RRI coefficient (R, S, L) gets its own network with one scalar output.
RRI_COEF_TRAIN_SPECS = (
    RriCoefTrainSpec(
        "Linear Resistor (R)",
        R_OUTPUT_COLUMN,
        lr_init=0.01,
        junction_num_layers=2,
        junction_layer_width=10,
        junction_asymmetric_overestimate_weight=2000,
        vessel_asymmetric_overestimate_weight=10,
    ),
    RriCoefTrainSpec(
        "Stenosis Resistor (S)",
        S_OUTPUT_COLUMN,
        lr_init=0.001,
        junction_num_layers=2,
        junction_layer_width=10,
        junction_asymmetric_overestimate_weight=100,
        vessel_asymmetric_overestimate_weight=10,
    ),
    RriCoefTrainSpec(
        "Inductor (L)",
        L_OUTPUT_COLUMN,
        lr_init=0.01,
        junction_num_layers=4,
        junction_layer_width=20,
        junction_asymmetric_overestimate_weight=10000,
        vessel_asymmetric_overestimate_weight=1000,
    ),
)

VESSEL_NUM_LAYERS = 2
VESSEL_LAYER_WIDTH = 10
TRAIN_EPOCHS = 500


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
    jax_filename = f"jax_arrays_vessel_num_geos_{num_geos}.pkl" if vessel else f"jax_arrays_num_geos_{num_geos}.pkl"
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
    return f"data/split_indices/{set_name}/{geometry_variant}/{set_type}/train_val_ind_{set_name}_num_geos_{num_geos}"


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
    asymmetric_loss_eff: bool,
    generation_weighted_loss_eff: bool,
    generation_weighted_loss_scale: float,
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
        "output_type": output_type,
        "set_name": set_name,
        "set_type": set_type,
        "num_geos": num_geos,
        "data_root": data_root,
        "geometry_variant": geometry_variant,
        "run_config_suffix": run_config_suffix,
        "jax_arrays_path": jax_path,
        "data_dict": jax_data,
        "use_leaky_relu": leaky_relu,
        "model_name_suffix": model_name_suffix,
        "asymmetric_loss_overestimate_weight": 1.0,
        "asymmetric_loss": asymmetric_loss_eff,
        "generation_weighted_loss": generation_weighted_loss_eff,
        "generation_weighted_loss_scale": generation_weighted_loss_scale,
    }
    if vessel:
        network_params["jax_arrays_filename"] = os.path.basename(jax_path)

    n_train = max(len(train_inds), 1)
    training_params = {
        "num_epochs": TRAIN_EPOCHS,
        "batch_size": int(np.ceil(n_train / 10)),
        "train_inds": train_inds,
        "val_inds": val_inds,
        "num_offsets": 1 if vessel else num_offsets,
    }
    if vessel:
        out_dir = model_dir or os.path.join("results", "models", set_name, geometry_variant + "_vessel")
        training_params["output_dir"] = out_dir
    elif model_dir:
        training_params["output_dir"] = model_dir

    return network_params, training_params


def launch_training(network_params, optimizer_params, training_params):
    """Train three single-output networks for R, S, and L (junction or vessel).

    Each network predicts one scalar; ``target_output_column`` selects which
    column of ``output_rri`` is the training target (0=R, 1=S, 2=L).

    Trained models are saved under ``results/models`` (or ``training_params['output_dir']``).
    """
    network_params["output_type"] = "rri"
    asymmetric_loss = bool(network_params["asymmetric_loss"])
    is_vessel = network_params.get("model_name_suffix") == "_vessel"

    shared_data_dict = network_params.get("data_dict")
    print("Training RRI models (one network per coefficient)...")

    for spec in RRI_COEF_TRAIN_SPECS:
        print(f"training model {spec.target_output_column + 1}:  {spec.label}")
        print(f"{network_params['num_input_features']} input features")

        network_params["target_output_column"] = spec.target_output_column
        optimizer_params["init"] = spec.lr_init
        training_params["num_epochs"] = TRAIN_EPOCHS

        if is_vessel:
            network_params["num_layers"] = VESSEL_NUM_LAYERS
            network_params["layer_width"] = VESSEL_LAYER_WIDTH
            overestimate_weight = spec.vessel_asymmetric_overestimate_weight if asymmetric_loss else 1.0
        else:
            network_params["num_layers"] = spec.junction_num_layers
            network_params["layer_width"] = spec.junction_layer_width
            overestimate_weight = spec.junction_asymmetric_overestimate_weight if asymmetric_loss else 1.0
        network_params["asymmetric_loss_overestimate_weight"] = overestimate_weight

        if shared_data_dict is not None:
            network_params["data_dict"] = shared_data_dict

        model = NeuralNet(network_params, optimizer_params)
        if shared_data_dict is None:
            shared_data_dict = model.data_dict
        train_nn(model, training_params)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch NN training")
    parser.add_argument("set_name", help="Set name (e.g., VMR)")
    parser.add_argument("num_geos", type=int, help="Number of geometries")
    parser.add_argument(
        "geometry_variant",
        nargs="?",
        default=None,
        help=(
            "Geometry variant: bifurcations, bifurcations_EL, or all (default: all). "
            "Can also be set via --geometry_variant."
        ),
    )
    parser.add_argument(
        "--geometry_variant",
        dest="geometry_variant_flag",
        default=None,
        help="Geometry variant (overrides positional if set). Use this when passing --vessel so order does not matter.",
    )
    parser.add_argument(
        "--split_path",
        default=None,
        help=(
            "Path to train/val split pickle "
            "(default: data/split_indices/.../train_val_ind_{set_name}_num_geos_{num_geos})"
        ),
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Directory to save models (default: results/models/{set_name}/{geometry_variant})",
    )
    parser.add_argument(
        "--vessel",
        action="store_true",
        help="Train vessel NN (R/S/L per vessel); uses vessel jax arrays and same geometry-based split",
    )
    parser.add_argument(
        "--leaky_relu",
        action="store_true",
        help="Use Leaky ReLU instead of ReLU (helps gradient flow when inputs span large ranges)",
    )
    parser.add_argument(
        "--print_gradients",
        action="store_true",
        help="Print gradient stats for the first batch before training (for debugging)",
    )
    parser.add_argument(
        "--quiet_epochs",
        action="store_true",
        help="Suppress per-epoch train/validation loss output during train_nn.",
    )
    parser.add_argument(
        "--asymmetric_loss",
        action="store_true",
        dest="asymmetric_loss",
        help=(
            "Use asymmetric loss (per-model overestimate weights). "
            "When off, symmetric loss (overestimate weight 1.0 for all models)."
        ),
    )
    parser.add_argument(
        "--run_config",
        default=DEFAULT_CLI_RUN_CONFIG,
        help="Run config suffix for path separation (e.g. gen_loss). "
        "jax_arrays and split_indices use .../set_name/<config>/... "
        "Use the exact suffix for jax/split paths (e.g. gen_loss or quadratic_resistor_gen_loss). "
        "Training-only suffix _gen_loss also enables generation-weighted loss unless overridden. "
        "Default: %(default)s. Pass an empty string for layouts without a run-config subfolder.",
    )
    parser.add_argument(
        "--generation_weighted_loss",
        action="store_true",
        dest="generation_weighted_loss",
        help=(
            "Weight training loss by bifurcation generation: weight = scale / 2^generation "
            "(larger weight for smaller generation; requires generation in jax pkl)."
        ),
    )
    parser.add_argument(
        "--generation_weighted_loss_scale",
        type=float,
        default=1.0,
        dest="generation_weighted_loss_scale",
        metavar="S",
        help="Overall multiplier for generation-weighted loss (default: 1.0 gives weight 1/2^gen).",
    )
    cli_args = parser.parse_args()

    set_name = cli_args.set_name
    num_geos = cli_args.num_geos
    print(f"num_geos: {num_geos}")

    geometry_variant_arg = getattr(cli_args, "geometry_variant_flag", None) or cli_args.geometry_variant or "all"
    run_config_raw = (cli_args.run_config or "").strip() or None
    # Paths use the full --run_config string (e.g. ..._gen_loss is its own jax/split tree).
    data_paths_suffix = run_config_raw
    if run_config_raw:
        rc_flags = run_config_suffix_to_flags(run_config_raw)
        asymmetric_loss_eff = bool(cli_args.asymmetric_loss or rc_flags["asymmetric_loss"])
        generation_weighted_loss_eff = bool(cli_args.generation_weighted_loss or rc_flags["generation_weighted_loss"])
    else:
        asymmetric_loss_eff = bool(cli_args.asymmetric_loss)
        generation_weighted_loss_eff = bool(cli_args.generation_weighted_loss)
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
        print(f"\n{'=' * 80}")
        print(f"Training {'vessel' if cli_args.vessel else 'junction'} models for geometry variant: {geometry_variant}")
        if asymmetric_loss_eff:
            print("Asymmetric loss: per-model overestimate weights")
        elif data_paths_suffix:
            print("Symmetric loss: overestimate weight = 1.0 for all models")
        if generation_weighted_loss_eff:
            print(
                f"Generation-weighted loss: ON (scale={float(cli_args.generation_weighted_loss_scale):g}; "
                f"from --generation_weighted_loss and/or --run_config ..._gen_loss)"
            )
        print(f"{'=' * 80}")

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
            asymmetric_loss_eff=asymmetric_loss_eff,
            generation_weighted_loss_eff=generation_weighted_loss_eff,
            generation_weighted_loss_scale=float(getattr(cli_args, "generation_weighted_loss_scale", 1.0)),
            leaky_relu=getattr(cli_args, "leaky_relu", False),
            model_dir=cli_args.model_dir,
        )
        training_params["print_gradients"] = getattr(cli_args, "print_gradients", False)
        training_params["verbose_epochs"] = not cli_args.quiet_epochs

        optimizer_params = {"init": 0.02, "transition_steps": 1000, "decay_rate": 0.95}

        launch_training(network_params, optimizer_params, training_params)


if __name__ == "__main__":
    main()
