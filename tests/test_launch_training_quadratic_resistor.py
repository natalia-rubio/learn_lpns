"""Tests for skipping stenosis (S) training when quadratic_resistor is off."""

from unittest.mock import patch

from learn_lpns.config import load_pipeline_config
from learn_lpns.neural_network import launch_training as lt


class _FakeModel:
    def __init__(self, network_params, optimizer_params):
        self.target_output_column = network_params["target_output_column"]

    data_dict = None


def _minimal_network_params():
    return {
        "output_type": "rri",
        "num_input_features": 4,
        "asymmetric_loss": False,
        "generation_weighted_loss": False,
        "generation_weighted_loss_decay_base": 2.0,
        "activation": "relu",
        "model_name_suffix": "",
    }


def test_launch_training_skips_stenosis_when_quadratic_resistor_off():
    training_cfg = load_pipeline_config().training
    trained_columns = []

    def fake_train_nn(model, training_params):
        trained_columns.append(model.target_output_column)

    with patch.object(lt, "NeuralNet", _FakeModel):
        with patch.object(lt, "train_nn", side_effect=fake_train_nn):
            lt.launch_training(
                _minimal_network_params(),
                {"init": 0.01},
                {"num_epochs": 1},
                training_cfg=training_cfg,
                quadratic_resistor=False,
            )

    assert trained_columns == [0, 2]


def test_launch_training_trains_all_coefficients_when_quadratic_resistor_on():
    training_cfg = load_pipeline_config().training
    trained_columns = []

    def fake_train_nn(model, training_params):
        trained_columns.append(model.target_output_column)

    with patch.object(lt, "NeuralNet", _FakeModel):
        with patch.object(lt, "train_nn", side_effect=fake_train_nn):
            lt.launch_training(
                _minimal_network_params(),
                {"init": 0.01},
                {"num_epochs": 1},
                training_cfg=training_cfg,
                quadratic_resistor=True,
            )

    assert trained_columns == [0, 1, 2]
