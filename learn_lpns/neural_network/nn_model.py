import os
from functools import partial

import jax.numpy as jnp
import optax
from jax import grad, jit

from learn_lpns.neural_network.nn_util import batched_forward_pass, get_L2, init_weights
from learn_lpns.tools.basic import load_dict

# Column indices in output_rri (each trained by a separate single-output network, or jointly).
R_OUTPUT_COLUMN = 0
S_OUTPUT_COLUMN = 1
L_OUTPUT_COLUMN = 2
RRI_NUM_OUTPUTS = 3


class NeuralNet:
    def __init__(
        self,
        network_params,
        optimizer_params,
    ):
        self.set_name = network_params["set_name"]

        self.set_type = network_params.get("set_type", "test")
        self.geometry_variant = network_params.get("geometry_variant", "bifurcations")

        if "data_dict" in network_params:
            self.data_dict = network_params["data_dict"]
        else:
            jax_arrays_path = network_params.get("jax_arrays_path")
            if jax_arrays_path is None:
                data_root = network_params.get("data_root", "data")
                run_config_suffix = network_params.get("run_config_suffix")
                jax_filename = network_params.get(
                    "jax_arrays_filename",
                    f"jax_arrays_num_geos_{network_params['num_geos']}.pkl",
                )
                path_parts = [data_root, "jax_arrays", self.set_name]
                if run_config_suffix:
                    path_parts.append(run_config_suffix)
                path_parts.extend([self.geometry_variant, self.set_type, jax_filename])
                jax_arrays_path = os.path.join(*path_parts)
            print(f"  Loading jax_arrays from: {jax_arrays_path}")
            self.data_dict = load_dict(jax_arrays_path)

        self.model_name_suffix = network_params.get("model_name_suffix", "")
        self.use_leaky_relu = network_params.get("use_leaky_relu", False)
        self.asymmetric_loss = bool(network_params["asymmetric_loss"])
        self.num_output_features = int(network_params.get("num_output_features", 1))
        if self.num_output_features == 1:
            self.target_output_column = int(network_params["target_output_column"])
            self.asymmetric_loss_overestimate_weight = float(network_params["asymmetric_loss_overestimate_weight"])
            self.asymmetric_loss_overestimate_weights = None
        else:
            self.target_output_column = None
            self.asymmetric_loss_overestimate_weight = None
            self.asymmetric_loss_overestimate_weights = jnp.asarray(
                network_params["asymmetric_loss_overestimate_weights"],
                dtype=jnp.float32,
            )
        if self.asymmetric_loss:
            if self.num_output_features == 1:
                print(f"  asymmetric_loss: ON  (overestimate weight={self.asymmetric_loss_overestimate_weight:g})")
            else:
                print(
                    "  asymmetric_loss: ON  "
                    f"(per-output overestimate weights={list(map(float, self.asymmetric_loss_overestimate_weights))})"
                )
        else:
            print("  asymmetric_loss: OFF  (symmetric loss; overestimate weight = 1.0)")

        self.output_type = network_params["output_type"]

        self.weights = init_weights(network_params)
        self.num_input_features = network_params["num_input_features"]
        self.num_layers = network_params["num_layers"]
        self.layer_width = network_params["layer_width"]

        self.input = self.data_dict["input"]
        self.output = self.data_dict[f"output_{self.output_type}"]
        n_rows = int(self.input.shape[0])
        graw = self.data_dict.get("generation")
        self.generation_weighted_loss = bool(network_params["generation_weighted_loss"])
        if graw is not None:
            garr = jnp.asarray(graw, dtype=jnp.float32).reshape(n_rows)
        elif self.generation_weighted_loss:
            raise ValueError(
                "generation_weighted_loss is enabled but jax pickle has no 'generation' array. "
                "Re-run run_data_processing so geometric CSVs include generation."
            )
        else:
            garr = jnp.zeros((n_rows,), dtype=jnp.float32)
        self._generation_full = garr
        self.generation_weighted_loss_scale = float(network_params["generation_weighted_loss_scale"])
        if self.generation_weighted_loss:
            print(
                f"  generation_weighted_loss: ON  "
                f"(sample weight = {self.generation_weighted_loss_scale} / 2^generation); "
                f"generation rows={n_rows}"
            )

        self.num_geos = network_params["num_geos"]
        self.decay_rate = optimizer_params["decay_rate"]

        self.scheduler = optax.exponential_decay(
            init_value=optimizer_params["init"],
            transition_steps=optimizer_params["transition_steps"],
            decay_rate=optimizer_params["decay_rate"],
        )
        self.optimizer = optax.adam(learning_rate=self.scheduler)
        self.opt_state = self.optimizer.init(self.weights)

    def get_gradients(self, indices):
        """Compute gradients of loss w.r.t. weights for the given batch (no update)."""
        idx = jnp.asarray(indices)
        if self.generation_weighted_loss:
            gen_b = self._generation_full[idx]
            sample_w = self.generation_weighted_loss_scale / jnp.power(2.0, gen_b)
        else:
            sample_w = jnp.ones((idx.shape[0],), dtype=jnp.float32)
        if self.num_output_features == 1:
            return grad(loss, argnums=-3)(
                self.input[indices, :],
                self.output[indices, :],
                self.target_output_column,
                self.num_output_features,
                self.use_leaky_relu,
                self.weights,
                self.asymmetric_loss_overestimate_weight,
                sample_w,
            )
        return grad(loss, argnums=-3)(
            self.input[indices, :],
            self.output[indices, :],
            0,
            self.num_output_features,
            self.use_leaky_relu,
            self.weights,
            self.asymmetric_loss_overestimate_weights,
            sample_w,
        )

    def update(self, indices):
        grads = self.get_gradients(indices)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.weights = optax.apply_updates(self.weights, updates)


@partial(jit, static_argnums=(2,))
def predict(input, weights, use_leaky_relu=False):
    return batched_forward_pass(input, weights, use_leaky_relu)


@partial(jit, static_argnums=(2, 3, 4))
def loss(
    input,
    outputs,
    target_output_column,
    num_output_features,
    use_leaky_relu,
    weights,
    overestimate_weight,
    sample_weights,
):
    coefs_pred = predict(input, weights, use_leaky_relu)
    if num_output_features == 1:
        residual = coefs_pred[:, 0] - outputs[:, target_output_column]
        absolute_residual = jnp.abs(coefs_pred[:, 0]) - jnp.abs(outputs[:, target_output_column])
        asymmetric_residual_weight = jnp.where(absolute_residual > 0, overestimate_weight, 1.0)
        residual_weight = asymmetric_residual_weight * sample_weights
        squared_residual = jnp.square(residual)
        L2_penalty = get_L2(weights) / (len(weights) * jnp.size(weights[0][0]))
        return (
            jnp.sum(residual_weight * squared_residual) / jnp.maximum(jnp.sum(residual_weight), 1e-8) + L2_penalty * 0
        )

    residuals = coefs_pred - outputs[:, :num_output_features]
    absolute_residual = jnp.abs(coefs_pred) - jnp.abs(outputs[:, :num_output_features])
    asymmetric_residual_weight = jnp.where(absolute_residual > 0, overestimate_weight, 1.0)
    residual_weight = asymmetric_residual_weight * sample_weights[:, None]
    squared_residual = jnp.square(residuals)
    per_sample = jnp.sum(residual_weight * squared_residual, axis=1)
    L2_penalty = get_L2(weights) / (len(weights) * jnp.size(weights[0][0]))
    return jnp.sum(per_sample) / jnp.maximum(jnp.sum(residual_weight), 1e-8) + L2_penalty * 0


@partial(jit, static_argnums=(2, 3, 4))
def loss_pure(
    input,
    outputs,
    target_output_column,
    num_output_features,
    use_leaky_relu,
    weights,
):
    coefs_pred = predict(input, weights, use_leaky_relu)
    if num_output_features == 1:
        return jnp.sqrt(jnp.mean(jnp.square(coefs_pred[:, 0] - outputs[:, target_output_column])))
    residuals = coefs_pred - outputs[:, :num_output_features]
    return jnp.sqrt(jnp.mean(jnp.square(residuals)))
