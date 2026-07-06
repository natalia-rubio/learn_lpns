import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from learn_lpns.config import get_pipeline_config
from learn_lpns.neural_network.nn_util import attach_train_output_bounds, dill_save, get_batch_indices


def _model_checkpoint_basename(model) -> str:
    if model.num_output_features > 1:
        return f"{model.output_type}_{model.set_name}{model.model_name_suffix}_pred_rsl"
    return f"{model.output_type}_{model.set_name}{model.model_name_suffix}_pred_{model.target_output_column}"


def _save_training_plot(
    *,
    out_dir: str,
    model_name: str,
    epoch: int,
    train_pure_hist: list,
    val_pure_hist: list,
    train_training_hist: list,
    val_training_hist: list,
) -> None:
    epochs = np.linspace(0, epoch, epoch + 1, True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(epochs, np.asarray(train_pure_hist), label="Train", color="cornflowerblue")
    axes[0].plot(epochs, np.asarray(val_pure_hist), label="Val", color="salmon")
    axes[0].set_ylabel("Pure RMSE")
    axes[0].set_title("Unweighted RMSE (no generation / asymmetric weighting)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, np.asarray(train_training_hist), label="Train", color="cornflowerblue")
    axes[1].plot(epochs, np.asarray(val_training_hist), label="Val", color="salmon")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training loss (weighted MSE)")
    axes[1].set_title("Training objective (generation- and/or asymmetric-weighted)")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{model_name}_training_plot.png"), bbox_inches="tight")
    plt.close(fig)


def _copy_weights(weights):
    return jax.tree_util.tree_map(jnp.copy, weights)


def train_nn(model, training_params):
    model_name = _model_checkpoint_basename(model)
    verbose_epochs = training_params.get("verbose_epochs", True)
    train_pure_hist: list[float] = []
    val_pure_hist: list[float] = []
    train_training_hist: list[float] = []
    val_training_hist: list[float] = []

    training_defaults = get_pipeline_config().training
    restore_best_weights = training_params.get(
        "restore_best_weights",
        training_defaults.restore_best_weights,
    )
    early_stop_threshold = training_params.get(
        "early_stop_loss_threshold",
        training_defaults.early_stop_loss_threshold,
    )

    best_train_loss = float("inf")
    best_epoch: int | None = None
    best_weights = None

    out_dir = training_params.get("output_dir")
    if out_dir is None:
        out_dir = os.path.join(
            "results",
            "models",
            str(model.set_name),
            model.geometry_variant + model.model_name_suffix,
        )

    num_offsets = training_params["num_offsets"]
    print("Number of offsets: ", num_offsets)
    train_inds = np.concatenate([training_params["train_inds"] * num_offsets + i for i in range(num_offsets)])
    val_inds = np.concatenate([training_params["val_inds"] * num_offsets + i for i in range(num_offsets)])
    print("Number of training points: ", len(train_inds))
    print("Number of validation points: ", len(val_inds))

    batch_size = training_params["batch_size"]
    if len(train_inds) == 0:
        raise ValueError(
            "No training points. Check that the train/val split and data (e.g. vessel pkl geometry names) "
            "match; see launch_training vessel split error message if applicable."
        )
    if batch_size > len(train_inds):
        print(
            f"  Warning: batch_size ({batch_size}) > number of training points ({len(train_inds)}). "
            f"Setting batch_size to {len(train_inds)}."
        )
        batch_size = len(train_inds)

    if training_params.get("print_gradients", False):
        inds = train_inds[:batch_size] if len(train_inds) >= batch_size else train_inds
        if len(inds) > 0:
            grads = model.get_gradients(inds)
            print("\n  Gradients (first batch, before training):")
            for layer_i, (gw, gb) in enumerate(grads):
                gw_np = np.array(gw)
                gb_np = np.array(gb)
                print(
                    f"    layer {layer_i}: grad_w norm={np.linalg.norm(gw_np):.2e} "
                    f"min={gw_np.min():.2e} max={gw_np.max():.2e}  "
                    f"grad_b norm={np.linalg.norm(gb_np):.2e} min={gb_np.min():.2e} max={gb_np.max():.2e}"
                )
            print("")
        else:
            print("\n  print_gradients: no training indices, skipping.\n")

    final_val_pure = float("nan")
    for epoch in range(training_params["num_epochs"]):
        start_time = time.time()
        batch_ind_list = get_batch_indices(train_inds, batch_size)
        for batch_inds in batch_ind_list:
            model.update(indices=batch_inds)
        epoch_time = time.time() - start_time

        train_pure = model.eval_pure_loss(train_inds)
        train_training = model.eval_training_loss(train_inds)
        train_pure_hist.append(train_pure)
        train_training_hist.append(train_training)

        if train_training < best_train_loss:
            best_train_loss = train_training
            best_epoch = epoch
            best_weights = _copy_weights(model.weights)

        if len(val_inds) > 0:
            val_pure = model.eval_pure_loss(val_inds)
            val_training = model.eval_training_loss(val_inds)
            val_pure_hist.append(val_pure)
            val_training_hist.append(val_training)
            final_val_pure = val_pure
            if verbose_epochs:
                print(
                    f"Epoch {epoch} in {epoch_time:0.2f} sec  |  "
                    f"Train pure RMSE {train_pure:e}  |  Train loss {train_training:e}  |  "
                    f"Val pure RMSE {val_pure:e}  |  Val loss {val_training:e}"
                )
        else:
            val_pure = float("nan")
            val_training = float("nan")
            val_pure_hist.append(val_pure)
            val_training_hist.append(val_training)
            if verbose_epochs:
                print(
                    f"Epoch {epoch} in {epoch_time:0.2f} sec  |  "
                    f"Train pure RMSE {train_pure:e}  |  Train loss {train_training:e}  |  "
                    "Val: N/A (100% train)"
                )

        if train_training < early_stop_threshold:
            print(
                f"\n  Early stopping: train training loss ({train_training:.2e}) "
                f"is below threshold ({early_stop_threshold:g})"
            )
            print(f"  Stopping training at epoch {epoch + 1}/{training_params['num_epochs']}")
            break

    final_epoch = epoch
    restored_from_best = False
    if restore_best_weights and best_weights is not None:
        restored_from_best = best_epoch != final_epoch
        model.weights = best_weights
        print(
            f"\n  Restored best weights from epoch {best_epoch} "
            f"(train training loss {best_train_loss:.2e})"
        )
        if restored_from_best:
            print(f"  (last epoch was {final_epoch})")
    elif best_epoch is not None:
        print(
            f"\n  Best train training loss {best_train_loss:.2e} at epoch {best_epoch} "
            f"(keeping last-epoch weights from epoch {final_epoch})"
        )

    model.best_epoch = best_epoch
    model.best_train_loss = best_train_loss if best_epoch is not None else None
    model.restored_from_best = restored_from_best

    if len(val_inds) > 0 and not np.isnan(final_val_pure):
        print(f"  Final val pure RMSE (monitoring only): {final_val_pure:.2e}")

    _save_training_plot(
        out_dir=out_dir,
        model_name=model_name,
        epoch=epoch,
        train_pure_hist=train_pure_hist,
        val_pure_hist=val_pure_hist,
        train_training_hist=train_training_hist,
        val_training_hist=val_training_hist,
    )
    train_inds = training_params["train_inds"]
    attach_train_output_bounds(model, train_inds)
    dill_save(model, os.path.join(out_dir, f"{model_name}_model"))

    final_val_pure_rmse = None if len(val_inds) == 0 or np.isnan(final_val_pure) else float(final_val_pure)
    return {
        "best_train_loss": float(best_train_loss) if best_epoch is not None else None,
        "best_epoch": best_epoch,
        "final_val_pure_rmse": final_val_pure_rmse,
    }
