from tarfile import tar_filter
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import optax
import os
from util.neural_network.nn_util import get_batch_indices, dill_save
from util.neural_network.nn_model import loss, loss_pure
import time
import matplotlib.pyplot as plt
import pdb
import multiprocessing
import dill



def train_nn(model, training_params):
    model_name2 = f"{model.output_type}_{model.set_name}{getattr(model, 'model_name_suffix', '')}_pred_{model.target_coef_ind}"
    model_name = f"{model.output_type}_{model.set_name}_ng_{model.num_geos}_nl_{model.num_layers}_lw_{model.layer_width}_ne_{training_params['num_epochs']}_bs_{training_params['batch_size']}_dr_{model.decay_rate}_{model.set_type}_pred_{model.target_coef_ind}"
    plotting = True
    train_hist = []
    val_hist = []

    # Output directory: use override from training_params if provided (e.g. for CV trials)
    out_dir = training_params.get("output_dir")
    if out_dir is None:
        geometry_variant = getattr(model, 'geometry_variant', 'bifurcations')
        out_dir = os.path.join("results", "models", str(model.set_name), geometry_variant + getattr(model, 'model_name_suffix', ''))
    
    num_offsets = training_params["num_offsets"]
    print("Number of offsets: ", num_offsets)
    train_inds = np.concatenate([training_params["train_inds"] * num_offsets  + i for i in range(num_offsets)])
    val_inds   = np.concatenate([training_params["val_inds"] * num_offsets + i for i in range(num_offsets)])
    # Print the number of training and validation points
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

    # Optionally print gradients for the first batch (before training)
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
    
    for epoch in range(training_params['num_epochs']): # Loop through the epochs
        start_time = time.time() # Time each epoch
        #import pdb; pdb.set_trace()
        batch_ind_list = get_batch_indices(train_inds, batch_size)  # Split the training set into random batches
        for i, batch_inds in enumerate(batch_ind_list): # Loop through the batches
            model.update(indices = batch_inds) # Update the model based on the batch
        # with multiprocessing.Pool() as pool:
        #         pool.map(model.update, batch_ind_list) # Update the model based on the batch

        epoch_time = time.time() - start_time
 
        train_loss = loss_pure(
            input=model.input[train_inds, :],
            outputs=model.output[train_inds, :],
            target_coef_ind=model.target_coef_ind,
            use_leaky_relu=getattr(model, "use_leaky_relu", False),
            weights=model.weights,
        )
        train_hist.append(train_loss)

        # Handle empty validation set (100% train)
        if len(val_inds) > 0:
            val_loss = loss_pure(
                input=model.input[val_inds, :],
                outputs=model.output[val_inds, :],
                target_coef_ind=model.target_coef_ind,
                use_leaky_relu=getattr(model, "use_leaky_relu", False),
                weights=model.weights,
            )
            val_hist.append(val_loss)
            print("Epoch {} in {:0.2f} sec  |  ".format(epoch, epoch_time) + \
                "Training set accuracy {:e}  |  ".format(train_loss) + \
                "Validation set accuracy {:e}".format(val_loss))
        else:
            val_loss = float('nan')
            val_hist.append(val_loss)
            print("Epoch {} in {:0.2f} sec  |  ".format(epoch, epoch_time) + \
                  "Training set accuracy {:e}  |  ".format(train_loss) + \
                  "Validation set: N/A (100% train)")
        
        # Early stopping: stop training if loss goes below 10^-3
        # Use validation loss if available, otherwise use training loss
        loss_to_check = val_loss if len(val_inds) > 0 and not np.isnan(val_loss) else train_loss
        if loss_to_check < 1e-7:
            print(f"\n  Early stopping: Loss ({loss_to_check:.2e}) is below threshold (1e-3)")
            print(f"  Stopping training at epoch {epoch+1}/{training_params['num_epochs']}")
            break

        # Check improvement in validation loss, if less than 1% for 10 consecutive epochs, stop training
        # if len(val_hist) > 10:
        #     if val_hist[-10] - val_hist[-1] < 0.000001 * val_hist[-1]:
        #         print(f"\n  Early stopping: Validation loss improvement is less than 1% for 10 consecutive epochs")
        #         print(f"  Stopping training at epoch {epoch+1}/{training_params['num_epochs']}")
        #         break
                
        
    if (epoch+1)%100 == 0 and plotting:

            plt.clf()
            plt.plot(np.linspace(0, epoch, epoch+1, True), np.asarray(train_hist), label = "Training Loss", color = 'cornflowerblue')
            plt.plot(np.linspace(0, epoch, epoch+1, True), np.asarray(val_hist), label = "Validation Loss", color = 'salmon')
            plt.xlabel("Epoch"); plt.ylabel("Loss (RMSE) (mmHg)"); plt.title("Training and Validation Loss")
            plt.yscale("log")
            plt.legend()
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{model_name}_training_plot.png"), bbox_inches='tight')

    # if model.target_coef_ind == 2:
    # import pdb; pdb.set_trace()
    plt.clf()
    plt.plot(np.linspace(0, epoch, epoch+1, True), np.asarray(train_hist), label = "Training Loss", color = 'cornflowerblue')
    plt.plot(np.linspace(0, epoch, epoch+1, True), np.asarray(val_hist), label = "Validation Loss", color = 'salmon')
    plt.xlabel("Epoch"); plt.ylabel("Loss (RMSE) (mmHg)"); plt.title("Training and Validation Loss")
    plt.yscale("log")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{model_name}_training_plot.png"), bbox_inches='tight')

    os.makedirs(out_dir, exist_ok=True)
    dill_save(model, os.path.join(out_dir, f"{model_name}_model"))
    dill_save(model, os.path.join(out_dir, f"{model_name2}_model"))
    # Return final validation loss (or NaN if 100% train)
    if len(val_inds) > 0:
        return val_loss.item()
    else:
        return float('nan')