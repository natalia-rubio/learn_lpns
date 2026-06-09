import os
import time

import matplotlib.pyplot as plt
import numpy as np

from learn_lpns.neural_network.nn_model import loss_pure
from learn_lpns.neural_network.nn_util import dill_save, get_batch_indices

EARLY_STOP_LOSS_THRESHOLD = 1e-7


def train_nn(model, training_params):
    output_column = model.target_output_column
    model_name = f"{model.output_type}_{model.set_name}{model.model_name_suffix}_pred_{output_column}"
    verbose_epochs = training_params.get("verbose_epochs", True)
    train_hist = []
    val_hist = []

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

    for epoch in range(training_params["num_epochs"]):
        start_time = time.time()
        batch_ind_list = get_batch_indices(train_inds, batch_size)
        for batch_inds in batch_ind_list:
            model.update(indices=batch_inds)
        epoch_time = time.time() - start_time

        train_loss = loss_pure(
            input=model.input[train_inds, :],
            outputs=model.output[train_inds, :],
            target_output_column=output_column,
            use_leaky_relu=model.use_leaky_relu,
            weights=model.weights,
        )
        train_hist.append(train_loss)

        if len(val_inds) > 0:
            val_loss = loss_pure(
                input=model.input[val_inds, :],
                outputs=model.output[val_inds, :],
                target_output_column=output_column,
                use_leaky_relu=model.use_leaky_relu,
                weights=model.weights,
            )
            val_hist.append(val_loss)
            if verbose_epochs:
                print(
                    f"Epoch {epoch} in {epoch_time:0.2f} sec  |  "
                    f"Training set accuracy {train_loss:e}  |  "
                    f"Validation set accuracy {val_loss:e}"
                )
        else:
            val_loss = float("nan")
            val_hist.append(val_loss)
            if verbose_epochs:
                print(
                    f"Epoch {epoch} in {epoch_time:0.2f} sec  |  "
                    f"Training set accuracy {train_loss:e}  |  "
                    "Validation set: N/A (100% train)"
                )

        loss_to_check = val_loss if len(val_inds) > 0 and not np.isnan(val_loss) else train_loss
        if loss_to_check < EARLY_STOP_LOSS_THRESHOLD:
            print(f"\n  Early stopping: Loss ({loss_to_check:.2e}) is below threshold ({EARLY_STOP_LOSS_THRESHOLD:g})")
            print(f"  Stopping training at epoch {epoch + 1}/{training_params['num_epochs']}")
            break

    plt.clf()
    plt.plot(
        np.linspace(0, epoch, epoch + 1, True),
        np.asarray(train_hist),
        label="Training Loss",
        color="cornflowerblue",
    )
    plt.plot(
        np.linspace(0, epoch, epoch + 1, True),
        np.asarray(val_hist),
        label="Validation Loss",
        color="salmon",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss (RMSE) (mmHg)")
    plt.title("Training and Validation Loss")
    plt.yscale("log")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{model_name}_training_plot.png"), bbox_inches="tight")
    dill_save(model, os.path.join(out_dir, f"{model_name}_model"))

    if len(val_inds) > 0:
        return val_loss.item()
    return float("nan")
