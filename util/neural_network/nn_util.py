import jax.numpy as jnp
from jax import grad, jit, vmap, random
import numpy as np
np.random.seed(0)
import pdb
import dill

# Neural Network Architecture
def get_sizes(network_params):
    # Get the sizes of the network layers
    if network_params["output_type"] == "rri":
        num_output_features = 3
    elif network_params["output_type"] == "ri":
        num_output_features = 2
    elif network_params["output_type"] == "rr":
        num_output_features = 2
    num_output_features = 1
    num_input_features = network_params["num_input_features"]
    num_layers = network_params["num_layers"]
    layer_width = network_params["layer_width"]
    #num_output_features = network_params["num_output_features"]
    #sizes = [num_input_features] + [layer_width]*num_layers + [num_output_features]
    sizes_in = [num_input_features] + [layer_width]*num_layers
    sizes_out = [layer_width]*num_layers + [num_output_features]
    return sizes_in, sizes_out

def random_layer_params(m, n, key, scale=1e-1):
    # Randomly initialize the weights of a layer
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), 0*scale * random.normal(b_key, (n,))

def init_weights(network_params):
    # Initialize the weights of the network
    key = random.key(0)
    #sizes = get_sizes(network_params)
    sizes_in, sizes_out = get_sizes(network_params)
    keys = random.split(key, len(sizes_in))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes_in, sizes_out, keys)]


# Data Handling
def get_train_val_split(num_pts, percent_train=0.8):
    # Split the data into training and validation sets
    indices = np.random.permutation(num_pts)
    train_indices = indices[:int(percent_train*num_pts)]
    val_indices = indices[int(percent_train*num_pts):]
    return train_indices, val_indices

def get_batch_indices(indices, batch_size):
    # Split data set into random batches
    num_batches = len(indices)//batch_size
    np.random.shuffle(indices)
    indices = indices[:num_batches*batch_size] # drop the last few if they don't make a full batch
    return [indices[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

# Network Utilities
def scale_jax(scaling_dict, field, field_name):
    # Inverse data normalization function
    mean = scaling_dict[field_name][0]
    std = scaling_dict[field_name][1]
    scaled_field = jnp.divide(jnp.subtract(field, mean), std)
    return jnp.reshape(scaled_field, (-1,1))

def inv_scale_jax(scaling_dict, field, field_name):
    # Inverse data normalization function
    mean = scaling_dict[field_name][0]
    std = scaling_dict[field_name][1]
    scaled_field = jnp.add(jnp.multiply(field, std), mean)
    return jnp.reshape(scaled_field, (-1,1))

def relu(x):
  # Rectified Linear Unit activation function
  return jnp.maximum(0, x)

def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + jnp.exp(-x))

def tanh(x):
    # Hyperbolic tangent activation function
    return jnp.tanh(x)

    
def forward_pass(input,weights):
    # Forward pass through the network
    latent_rep = input    
    for w, b in weights[:-1]:
        #jnp.concatenate((latent_rep, input), axis=-1)
        lin_comb = jnp.dot(w, latent_rep) + b
        latent_rep = relu(lin_comb)

    final_w, final_b = weights[-1]
    output = jnp.dot(final_w, latent_rep) + final_b
    #output = input[5:] * weights
    return output

def get_L2(weights):
    # Get the L2 norm of the weights
    return jnp.sum(jnp.array([jnp.linalg.norm(w) for w, _ in weights]))

batched_forward_pass = vmap(forward_pass, in_axes=(0, None))

def dill_save(di_, filename_):
    with open(filename_, 'wb') as f:
        dill.dump(di_, f)
    return

def dill_load(filename_):
    with open(filename_, 'rb') as f:
        dict = dill.load(f)
    return dict

# Optimization
# opt_state = optimizer.init(params)