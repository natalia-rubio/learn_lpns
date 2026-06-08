import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
import dill

np.random.seed(0)


def get_sizes(network_params):
    """Layer shapes for a single-output network."""
    num_output_features = 1
    num_input_features = network_params["num_input_features"]
    layer_width = network_params["layer_width"]
    num_layers = network_params["num_layers"]
    sizes_in = [num_input_features] + [layer_width] * num_layers
    sizes_out = [layer_width] * num_layers + [num_output_features]
    return sizes_in, sizes_out


def random_layer_params(m, n, key, scale=1e-1):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), 0 * scale * random.normal(b_key, (n,))


def init_weights(network_params):
    key = random.key(0)
    sizes_in, sizes_out = get_sizes(network_params)
    keys = random.split(key, len(sizes_in))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes_in, sizes_out, keys)]


def get_batch_indices(indices, batch_size):
    num_batches = len(indices) // batch_size
    np.random.shuffle(indices)
    indices = indices[:num_batches * batch_size]
    return [indices[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def relu(x):
    return jnp.maximum(0, x)


def leaky_relu(x, negative_slope=0.01):
    """Leaky ReLU: max(negative_slope * x, x). Gradient flows when x < 0."""
    return jnp.where(x >= 0, x, negative_slope * x)


def forward_pass(input, weights, use_leaky_relu=False):
    activation = leaky_relu if use_leaky_relu else relu
    latent_rep = input
    for w, b in weights[:-1]:
        lin_comb = jnp.dot(w, latent_rep) + b
        latent_rep = activation(lin_comb)

    final_w, final_b = weights[-1]
    return jnp.dot(final_w, latent_rep) + final_b


def batched_forward_pass(input, weights, use_leaky_relu=False):
    """Vectorized forward pass over batch dimension (axis 0 of input)."""
    return vmap(forward_pass, in_axes=(0, None, None))(input, weights, use_leaky_relu)


def get_L2(weights):
    return jnp.sum(jnp.array([jnp.linalg.norm(w) for w, _ in weights]))


def dill_save(di_, filename_):
    with open(filename_, "wb") as f:
        dill.dump(di_, f)


def dill_load(filename_):
    with open(filename_, "rb") as f:
        return dill.load(f)
