"""Tests for skipping stenosis (S) training when quadratic_resistor is off."""

from unittest.mock import patch

from learn_lpns.config import load_pipeline_config
from learn_lpns.neural_network import launch_training as lt


class _FakeModel:
    def __init__(self, network_params, optimizer_params):
        self.target_output_column = network_params["target_output_column"]
        self.generation_weighted_loss_decay_base = network_params[
            "generation_weighted_loss_decay_base"
        ]

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


def test_per_coefficient_decay_override_does_not_leak_to_next_model():
    training_cfg = load_pipeline_config(run_config="quadratic_resistor_gen_loss").training
    coefficients = tuple(
        spec.model_copy(
            update={
                "generation_weighted_loss_decay_base": (
                    4.0 if spec.name == "S" else None
                )
            }
        )
        for spec in training_cfg.rri_coefficients
    )
    training_cfg = training_cfg.model_copy(
        update={
            "generation_weighted_loss_decay_base": 3.0,
            "rri_coefficients": coefficients,
        }
    )
    trained_decay_bases = []

    def fake_train_nn(model, training_params):
        trained_decay_bases.append(model.generation_weighted_loss_decay_base)

    network_params = _minimal_network_params()
    network_params["generation_weighted_loss_decay_base"] = 3.0
    with patch.object(lt, "NeuralNet", _FakeModel):
        with patch.object(lt, "train_nn", side_effect=fake_train_nn):
            lt.launch_training(
                network_params,
                {"init": 0.01},
                {"num_epochs": 1},
                training_cfg=training_cfg,
                quadratic_resistor=True,
            )

    assert trained_decay_bases == [3.0, 4.0, 3.0]
