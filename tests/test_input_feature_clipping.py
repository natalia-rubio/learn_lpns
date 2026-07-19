from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from learn_lpns.neural_network.nn_util import (
    clip_data_dict_inputs_to_train_bounds,
    clip_inputs_to_train_bounds,
    resolve_train_input_bounds,
)


def test_clip_data_dict_uses_only_training_rows_for_bounds():
    data_dict = {
        "input": jnp.asarray(
            [
                [0.0, 10.0],
                [2.0, 20.0],
                [-5.0, 15.0],
                [8.0, 30.0],
            ]
        )
    }

    clipped = clip_data_dict_inputs_to_train_bounds(data_dict, [0, 1])

    np.testing.assert_allclose(clipped["train_input_min"], [0.0, 10.0])
    np.testing.assert_allclose(clipped["train_input_max"], [2.0, 20.0])
    np.testing.assert_allclose(
        clipped["input"],
        [
            [0.0, 10.0],
            [2.0, 20.0],
            [0.0, 15.0],
            [2.0, 20.0],
        ],
    )
    assert clipped["clip_input_features"] is True
    np.testing.assert_allclose(data_dict["input"][2], [-5.0, 15.0])


def test_inference_clipping_uses_checkpoint_bounds():
    model = SimpleNamespace(
        train_input_min=np.asarray([0.0, 10.0]),
        train_input_max=np.asarray([2.0, 20.0]),
    )

    clipped, changed_count = clip_inputs_to_train_bounds(
        np.asarray([[-1.0, 15.0], [1.0, 25.0]]),
        model,
    )

    np.testing.assert_allclose(clipped, [[0.0, 15.0], [1.0, 20.0]])
    assert changed_count == 2


def test_input_clipping_requires_bounds_on_checkpoint():
    with pytest.raises(ValueError, match="Retrain the model"):
        resolve_train_input_bounds(SimpleNamespace(data_dict={}))
