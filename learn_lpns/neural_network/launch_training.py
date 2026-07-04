import glob
import os
import re

from learn_lpns.config import TrainingConfig, get_pipeline_config
from learn_lpns.data_processing.generate_split_indices import (
    load_split_for_training,
    resolve_flat_indices,
)
from learn_lpns.neural_network.nn_model import RRI_NUM_OUTPUTS, NeuralNet
from learn_lpns.neural_network.nn_util import append_output_rri_to_input
from learn_lpns.neural_network.train_nn import train_nn
from learn_lpns.tools.basic import load_dict
from learn_lpns.zerod_calibration.run_config_canonical import (
    DEFAULT_CLI_RUN_CONFIG,
    run_config_suffix_to_flags,
)

GEOMETRY_VARIANT_NAMES = frozenset({"bifurcations", "bifurcations_EL", "all"})
DEFAULT_GEOMETRY_VARIANT = "bifurcations_EL"


def _num_geos_from_path(path: str) -> int | None:
    match = re.search(r"num_geos_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def _jax_arrays_dir(
    data_root: str,
    set_name: str,
    geometry_variant: str,
    set_type: str,
    run_config_suffix: str | None,
) -> str:
    parts = [data_root, "jax_arrays", set_name]
    if run_config_suffix:
        parts.append(run_config_suffix)
    parts.extend([geometry_variant, set_type])
    return os.path.join(*parts)


def _discover_num_geos_from_jax_arrays(
    data_root: str,
    set_name: str,
    geometry_variant: str,
    set_type: str,
    run_config_suffix: str | None,
    *,
    vessel: bool,
) -> list[int]:
    directory = _jax_arrays_dir(data_root, set_name, geometry_variant, set_type, run_config_suffix)
    pattern = "jax_arrays_vessel_num_geos_*.pkl" if vessel else "jax_arrays_num_geos_*.pkl"
    nums: list[int] = []
    for path in glob.glob(os.path.join(directory, pattern)):
        num_geos = _num_geos_from_path(path)
        if num_geos is not None:
            nums.append(num_geos)
    return sorted(set(nums))


def _infer_num_geos(
    *,
    data_root: str,
    set_name: str,
    geometry_variant: str,
    set_type: str,
    run_config_suffix: str | None,
    vessel: bool,
    explicit_num_geos: int | None,
    split_path: str | None,
) -> int:
    if explicit_num_geos is not None:
        return explicit_num_geos

    if split_path is not None:
        from_split = _num_geos_from_path(split_path)
        if from_split is not None:
            return from_split

    junction_candidates = _discover_num_geos_from_jax_arrays(
        data_root,
        set_name,
        geometry_variant,
        set_type,
        run_config_suffix,
        vessel=False,
    )
    if vessel:
        vessel_candidates = set(
            _discover_num_geos_from_jax_arrays(
                data_root,
                set_name,
                geometry_variant,
                set_type,
                run_config_suffix,
                vessel=True,
            )
        )
        junction_candidates = [n for n in junction_candidates if n in vessel_candidates]

    if not junction_candidates:
        directory = _jax_arrays_dir(data_root, set_name, geometry_variant, set_type, run_config_suffix)
        raise SystemExit(
            f"Could not infer num_geos: no jax arrays under {directory!r}. "
            "Run data processing or pass num_geos explicitly."
        )

    if split_path is not None:
        return max(junction_candidates)

    with_split = [
        n
        for n in junction_candidates
        if os.path.isfile(_default_split_path(data_root, set_name, geometry_variant, set_type, n, run_config_suffix))
    ]
    if not with_split:
        raise SystemExit(
            f"Could not infer num_geos: found jax arrays for num_geos={junction_candidates} "
            "but no matching split_indices files. Run data processing or pass num_geos / --split_path."
        )
    return max(with_split)


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
    data_root: str,
    set_name: str,
    geometry_variant: str,
    set_type: str,
    num_geos: int,
    run_config_suffix: str | None,
) -> str:
    if run_config_suffix:
        return (
            f"{data_root}/split_indices/{set_name}/{run_config_suffix}/"
            f"{geometry_variant}/{set_type}/train_val_ind_{set_name}_num_geos_{num_geos}"
        )
    return (
        f"{data_root}/split_indices/{set_name}/{geometry_variant}/{set_type}/"
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
    asymmetric_loss_eff: bool,
    generation_weighted_loss_eff: bool,
    generation_weighted_loss_decay_base: float,
    leaky_relu: bool,
    model_dir: str | None,
    training_cfg: TrainingConfig,
    oracle_inputs: bool = False,
    stenosis_generation_limit_enabled: bool = False,
    stenosis_generation_max: float = 1.0,
) -> tuple[dict, dict]:
    modality = "vessel" if vessel else "junction"
    jax_data = load_dict(jax_path)
    if oracle_inputs:
        jax_data = append_output_rri_to_input(jax_data)
        asymmetric_loss_eff = False
        generation_weighted_loss_eff = False
        print(
            "  Oracle inputs: appended R/S/L to feature matrix; "
            "generation-weighted and asymmetric loss disabled"
        )
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
        "generation_weighted_loss_decay_base": generation_weighted_loss_decay_base,
        "stenosis_generation_limit_enabled": stenosis_generation_limit_enabled,
        "stenosis_generation_max": stenosis_generation_max,
    }
    if vessel:
        network_params["jax_arrays_filename"] = os.path.basename(jax_path)

    n_train = max(len(train_inds), 1)
    training_params = {
        "num_epochs": training_cfg.num_epochs,
        "batch_size": training_cfg.batch_size_for_n_train(n_train),
        "train_inds": train_inds,
        "val_inds": val_inds,
        "num_offsets": 1 if vessel else num_offsets,
        "early_stop_loss_threshold": training_cfg.early_stop_loss_threshold,
    }
    if vessel:
        out_dir = model_dir or os.path.join("results", "models", set_name, geometry_variant + "_vessel")
        training_params["output_dir"] = out_dir
    elif model_dir:
        training_params["output_dir"] = model_dir

    return network_params, training_params


def launch_training(
    network_params,
    optimizer_params,
    training_params,
    training_cfg: TrainingConfig | None = None,
    *,
    multi_output_rri: bool | None = None,
):
    """Train RRI models: one network per coefficient (default) or one 3-output network."""
    training_cfg = training_cfg or get_pipeline_config().training
    use_multi_output = training_cfg.multi_output_rri if multi_output_rri is None else multi_output_rri
    if use_multi_output:
        _launch_training_multi_output(
            network_params,
            optimizer_params,
            training_params,
            training_cfg,
        )
        return

    network_params["output_type"] = "rri"
    asymmetric_loss = bool(network_params["asymmetric_loss"])
    is_vessel = network_params.get("model_name_suffix") == "_vessel"

    shared_data_dict = network_params.get("data_dict")
    print("Training RRI models (one network per coefficient)...")

    for spec in training_cfg.rri_coefficients:
        print(f"training model {spec.target_output_column + 1}:  {spec.label}")
        print(f"{network_params['num_input_features']} input features")

        network_params["target_output_column"] = spec.target_output_column
        network_params["num_output_features"] = 1
        optimizer_params["init"] = spec.lr_init
        training_params["num_epochs"] = spec.training_epochs(
            vessel=is_vessel,
            default=training_cfg.num_epochs,
        )

        if is_vessel:
            network_params["num_layers"] = training_cfg.vessel.num_layers
            network_params["layer_width"] = training_cfg.vessel.layer_width
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


def _launch_training_multi_output(
    network_params,
    optimizer_params,
    training_params,
    training_cfg: TrainingConfig,
):
    network_params["output_type"] = "rri"
    asymmetric_loss = bool(network_params["asymmetric_loss"])
    is_vessel = network_params.get("model_name_suffix") == "_vessel"

    print("Training RRI model (single network with R, S, L outputs)...")
    print(f"{network_params['num_input_features']} input features")

    network_params["num_output_features"] = RRI_NUM_OUTPUTS
    network_params.pop("target_output_column", None)
    optimizer_params["init"] = training_cfg.optimizer.init
    training_params["num_epochs"] = training_cfg.num_epochs

    if is_vessel:
        network_params["num_layers"] = training_cfg.vessel.num_layers
        network_params["layer_width"] = training_cfg.vessel.layer_width
        overestimate_weights = [
            spec.vessel_asymmetric_overestimate_weight if asymmetric_loss else 1.0
            for spec in training_cfg.rri_coefficients
        ]
    else:
        network_params["num_layers"] = max(spec.junction_num_layers for spec in training_cfg.rri_coefficients)
        network_params["layer_width"] = max(spec.junction_layer_width for spec in training_cfg.rri_coefficients)
        overestimate_weights = [
            spec.junction_asymmetric_overestimate_weight if asymmetric_loss else 1.0
            for spec in training_cfg.rri_coefficients
        ]
    network_params["asymmetric_loss_overestimate_weights"] = overestimate_weights

    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)


def main():
    import argparse

    training_defaults = get_pipeline_config().training
    parser = argparse.ArgumentParser(description="Launch NN training")
    parser.add_argument("--set_name", required=True, help="Set name (e.g., VMR_aorta_starter)")
    parser.add_argument(
        "--num_geos",
        type=int,
        default=None,
        help="Number of geometries (default: infer from jax_arrays / split_indices).",
    )
    parser.add_argument(
        "--geometry_variant",
        default=None,
        help=(f"Geometry variant: {', '.join(sorted(GEOMETRY_VARIANT_NAMES))} (default: {DEFAULT_GEOMETRY_VARIANT})."),
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
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use Leaky ReLU instead of ReLU (helps gradient flow when inputs span large ranges). "
            "Default follows training.leaky_relu in config (false)."
        ),
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
            "Weight training loss by bifurcation generation: weight = 1 / base^generation "
            "(larger weight for smaller generation; requires generation in jax pkl; base from config)."
        ),
    )
    parser.add_argument(
        "--generation_weighted_loss_decay_base",
        type=float,
        default=training_defaults.generation_weighted_loss_decay_base,
        dest="generation_weighted_loss_decay_base",
        metavar="B",
        help=(
            f"Decay base for generation-weighted loss (weight = 1 / B^generation; must be > 1). "
            f"Default: {training_defaults.generation_weighted_loss_decay_base} from config."
        ),
    )
    parser.add_argument(
        "--multi_output_rri",
        action="store_true",
        help=(
            "Train one network with R/S/L outputs instead of three separate networks. "
            "Default follows training.multi_output_rri in config (false)."
        ),
    )
    parser.add_argument(
        "--oracle_inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append R/S/L targets to the input feature matrix (training sanity check only; "
            "not for deploy/inference)."
        ),
    )
    cli_args = parser.parse_args()

    training_cfg = get_pipeline_config(set_name=cli_args.set_name).training
    set_name = cli_args.set_name
    multi_output_rri = bool(cli_args.multi_output_rri or training_cfg.multi_output_rri)
    leaky_relu = training_cfg.leaky_relu if cli_args.leaky_relu is None else bool(cli_args.leaky_relu)

    geometry_variant_arg = cli_args.geometry_variant or DEFAULT_GEOMETRY_VARIANT
    if geometry_variant_arg not in GEOMETRY_VARIANT_NAMES:
        parser.error(
            f"Invalid --geometry_variant {geometry_variant_arg!r}. "
            f"Expected one of: {', '.join(sorted(GEOMETRY_VARIANT_NAMES))}."
        )
    run_config_raw = (cli_args.run_config or "").strip() or None
    data_paths_suffix = run_config_raw
    if run_config_raw:
        rc_flags = run_config_suffix_to_flags(run_config_raw)
        asymmetric_loss_eff = bool(cli_args.asymmetric_loss or rc_flags["asymmetric_loss"])
        generation_weighted_loss_eff = bool(cli_args.generation_weighted_loss or rc_flags["generation_weighted_loss"])
        quadratic_resistor_eff = bool(rc_flags["quadratic_resistor"])
    else:
        asymmetric_loss_eff = bool(cli_args.asymmetric_loss)
        generation_weighted_loss_eff = bool(cli_args.generation_weighted_loss)
        quadratic_resistor_eff = False
    vessel = bool(cli_args.vessel)
    dp_limit = get_pipeline_config(set_name=set_name).data_processing.stenosis_generation_limit
    stenosis_generation_limit_enabled = bool(dp_limit.enabled and quadratic_resistor_eff)
    stenosis_generation_max = float(
        dp_limit.vessel_limit() if vessel else dp_limit.junction_limit()
    )
    output_type = "rri"
    set_type = "all"
    data_root = "data"
    explicit_num_geos = cli_args.num_geos

    if geometry_variant_arg == "all":
        geometry_variants_to_process = ["bifurcations", "bifurcations_EL"]
    else:
        geometry_variants_to_process = [geometry_variant_arg]

    opt = training_cfg.optimizer
    optimizer_params = {
        "init": opt.init,
        "transition_steps": opt.transition_steps,
        "decay_rate": opt.decay_rate,
    }

    for geometry_variant in geometry_variants_to_process:
        num_geos = _infer_num_geos(
            data_root=data_root,
            set_name=set_name,
            geometry_variant=geometry_variant,
            set_type=set_type,
            run_config_suffix=data_paths_suffix,
            vessel=vessel,
            explicit_num_geos=explicit_num_geos,
            split_path=cli_args.split_path,
        )
        print(f"num_geos: {num_geos}")
        print(f"\n{'=' * 80}")
        print(f"Training {'vessel' if cli_args.vessel else 'junction'} models for geometry variant: {geometry_variant}")
        if asymmetric_loss_eff:
            print("Asymmetric loss: per-model overestimate weights")
        elif data_paths_suffix:
            print("Symmetric loss: overestimate weight = 1.0 for all models")
        if generation_weighted_loss_eff:
            print(
                f"Generation-weighted loss: ON (decay_base={float(cli_args.generation_weighted_loss_decay_base):g}; "
                f"from --generation_weighted_loss and/or --run_config ..._gen_loss)"
            )
        if stenosis_generation_limit_enabled:
            modality_label = "vessel" if vessel else "junction"
            print(
                f"Stenosis generation limit: ON ({modality_label} S training for generation <= "
                f"{stenosis_generation_max:g})"
            )
        if multi_output_rri:
            print("Multi-output RRI: one network with R, S, L outputs")
        if cli_args.oracle_inputs:
            print("Oracle inputs: ON (R/S/L appended to features; not for deploy)")
        if leaky_relu:
            print("Leaky ReLU: ON")
        print(f"{'=' * 80}")

        split_path = cli_args.split_path or _default_split_path(
            data_root, set_name, geometry_variant, set_type, num_geos, data_paths_suffix
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
            generation_weighted_loss_decay_base=float(cli_args.generation_weighted_loss_decay_base),
            leaky_relu=leaky_relu,
            model_dir=cli_args.model_dir,
            training_cfg=training_cfg,
            oracle_inputs=bool(cli_args.oracle_inputs),
            stenosis_generation_limit_enabled=stenosis_generation_limit_enabled,
            stenosis_generation_max=stenosis_generation_max,
        )
        training_params["print_gradients"] = getattr(cli_args, "print_gradients", False)
        training_params["verbose_epochs"] = not cli_args.quiet_epochs

        launch_training(
            network_params,
            optimizer_params,
            training_params,
            training_cfg=training_cfg,
            multi_output_rri=multi_output_rri,
        )


if __name__ == "__main__":
    main()
