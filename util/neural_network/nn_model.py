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

        data_root = network_params.get("data_root", "data")
        jax_arrays_path = os.path.join(
            data_root,
            "jax_arrays",
            self.set_name,
            self.geometry_variant,
            self.set_type,
            f"jax_arrays_num_geos_{network_params['num_geos']}.pkl",
        )
        self.data_dict = load_dict(jax_arrays_path)

        import pdb; pdb.set_trace()
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
    
    def update(self, indices):
        grads = grad(loss, argnums = -1)(self.input[indices,:],
            self.output[indices,:],
            self.data_dict["scaling_factors"][indices,:],
            self.scaling_dict,
            self.target_coef_ind,
            self.weights,
            )
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.weights = optax.apply_updates(self.weights, updates)
        #pdb.set_trace()
        return


@jit
def predict(input, weights):
    output = batched_forward_pass(input, weights)
    return output 

@jit
def loss(input, outputs, scaling_factors, scaling_dict, target_coef_ind, weights):
    coefs_pred = predict(input, weights)
    #pdb.set_trace()
    L2_penalty = get_L2(weights)/(len(weights) * jnp.size(weights[0][0]))
    #return jnp.mean(jnp.square(coefs_pred[:,target_coef_ind] - outputs[:,target_coef_ind])) + L2_penalty*0 #*1#L2 regularization term
    return jnp.mean(jnp.square(coefs_pred[:,0] - outputs[:,target_coef_ind])) + L2_penalty*0 #*1#L2 regularization term

@jit
def loss_pure(input, outputs, scaling_factors, scaling_dict, target_coef_ind, weights):
    coefs_pred = predict(input, weights)
    #pdb.set_trace()
    L2_penalty = get_L2(weights)/(len(weights) * jnp.size(weights[0][0]))
    #return jnp.sqrt(jnp.mean(jnp.square(coefs_pred[:,target_coef_ind] - outputs[:,target_coef_ind]))) #L2 regularization term
    return jnp.sqrt(jnp.mean(jnp.square(coefs_pred[:,0] - outputs[:,target_coef_ind]))) #L2 regularization term



