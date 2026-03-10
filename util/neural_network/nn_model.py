import os

import jax.numpy as jnp
from jax import grad, jit
import optax

from util.tools.basic import load_dict
from util.neural_network.nn_util import get_L2, init_weights, batched_forward_pass

class NeuralNet():
   
    def __init__(self, network_params, optimizer_params,):
        # Naming: prefer `set_name`, keep `anatomy` as backward-compatible alias.
        if "set_name" in network_params:
            self.set_name = network_params["set_name"]
        elif "anatomy" in network_params:
            self.set_name = network_params["anatomy"]
        else:
            raise ValueError("network_params must include 'set_name' (preferred) or legacy 'anatomy'")

        # Default set_type to "test" for convenience.
        self.set_type = network_params.get("set_type", "test")
        
        # Geometry variant (default: "bifurcations" for backward compatibility)
        self.geometry_variant = network_params.get("geometry_variant", "bifurcations")
        
        # Normalization flag (default: False for backward compatibility)
        self.normalize = network_params.get("normalize", False)
        norm_suffix = "_normalized" if self.normalize else ""

        data_root = network_params.get("data_root", "data")
        run_config_suffix = network_params.get("run_config_suffix")
        jax_filename = network_params.get(
            "jax_arrays_filename",
            f"jax_arrays_num_geos_{network_params['num_geos']}{norm_suffix}.pkl",
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
        # Asymmetric loss: overestimates (pred > target) weighted more than underestimates. None or 1.0 = symmetric.
        self.asymmetric_loss_overestimate_weight = network_params.get("asymmetric_loss_overestimate_weight", 1.0)

        # scaling_dict is not used in the current loss, but keep attribute for API compatibility.
        self.scaling_dict = network_params.get("scaling_dict", {})
        self.output_type    = network_params["output_type"]
        self.target_coef_ind = network_params["target_coef_ind"]
        self.weights        =  init_weights(network_params)
        self.num_input_features = network_params["num_input_features"]
        self.num_layers     = network_params["num_layers"]
        self.layer_width    = network_params["layer_width"]
        
        self.input = self.data_dict["input"]
        self.output = self.data_dict[f"output_{self.output_type}"]
        
        self.num_output_coefs = 3

            
        self.num_geos       = network_params["num_geos"]
        self.decay_rate     = optimizer_params["decay_rate"]

        self.scheduler = optax.exponential_decay(init_value = optimizer_params["init"], 
                                                 transition_steps = optimizer_params["transition_steps"], 
                                                 decay_rate = optimizer_params["decay_rate"])
        self.optimizer = optax.adam(learning_rate = self.scheduler)
        #self.optimizer = optax.sgd(learning_rate = self.scheduler)
        self.opt_state = self.optimizer.init(self.weights)
        return
    
    def get_gradients(self, indices):
        """Compute gradients of loss w.r.t. weights for the given batch (no update)."""
        print(f" Overestimate weight: {self.asymmetric_loss_overestimate_weight}")
        return grad(loss, argnums=-2)(
            self.input[indices, :],
            self.output[indices, :],
            self.data_dict["scaling_factors"][indices, :],
            self.scaling_dict,
            self.target_coef_ind,
            self.use_leaky_relu,
            self.weights,
            self.asymmetric_loss_overestimate_weight,
        )

    def update(self, indices):
        grads = self.get_gradients(indices)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.weights = optax.apply_updates(self.weights, updates)
        return


@jit(static_argnums=(2,))
def predict(input, weights, use_leaky_relu=False):
    output = batched_forward_pass(input, weights, use_leaky_relu)
    return output


@jit(static_argnums=(4, 5))  # target_coef_ind, use_leaky_relu
def loss(input, outputs, scaling_factors, scaling_dict, target_coef_ind, use_leaky_relu, weights, overestimate_weight=1.0):
    coefs_pred = predict(input, weights, use_leaky_relu)
    residual = coefs_pred[:, 0] - outputs[:, target_coef_ind]
    av_residual = jnp.abs(coefs_pred[:, 0]) - jnp.abs(outputs[:, target_coef_ind]) 
    # Overestimate (residual > 0) weighted more than underestimate (residual <= 0)
    # w = jnp.where((outputs[:, target_coef_ind] > 0) & (residual > 0), overestimate_weight, 1.0)
    # w = jnp.where((outputs[:, target_coef_ind] < 0) & (residual <= 0), overestimate_weight, w)
    w = jnp.where(av_residual > 0, overestimate_weight, 1.0)
    L2_penalty = get_L2(weights) / (len(weights) * jnp.size(weights[0][0]))
    return jnp.mean(w * jnp.square(residual)) + L2_penalty * 0


@jit(static_argnums=(4, 5))  # target_coef_ind, use_leaky_relu
def loss_pure(input, outputs, scaling_factors, scaling_dict, target_coef_ind, use_leaky_relu, weights):
    coefs_pred = predict(input, weights, use_leaky_relu)
    return jnp.sqrt(jnp.mean(jnp.square(coefs_pred[:, 0] - outputs[:, target_coef_ind])))



