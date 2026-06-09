"""Pydantic models for centralized pipeline configuration."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field


class PhysicsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rho: float = Field(default=1.06, gt=0, description="Blood density (g/cm^3)")
    mu: float = Field(default=0.04, gt=0, description="Blood viscosity (Poise)")


class SolverConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    absolute_tolerance: float = Field(default=1e-5, gt=0)
    maximum_nonlinear_iterations: int = Field(default=50, gt=0)
    number_of_cardiac_cycles: int = Field(default=1, gt=0)
    steady_initial: bool = False


class SetL2Penalty(BaseModel):
    """Per-cohort L2 regularization for calibration (R_poiseuille, stenosis_coefficient)."""

    model_config = ConfigDict(frozen=True)

    l2_r: float = Field(gt=0)
    l2_stenosis: float = Field(gt=0)


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    tolerance_gradient: float = Field(default=1e-4, gt=0)
    tolerance_increment: float = Field(default=1e-4, gt=0)
    maximum_iterations: int = Field(default=20, gt=0)
    default_l2_r: float = Field(default=1e5, gt=0)
    default_l2_stenosis: float = Field(default=1e10, gt=0)
    set_l2_penalties: dict[str, SetL2Penalty] = Field(default_factory=dict)

    def l2_penalties_for_set(self, set_name: str | None) -> tuple[float, float]:
        """Return (L2_penalty_R_poiseuille, L2_penalty_stenosis_coefficient) for a cohort."""
        if set_name and set_name in self.set_l2_penalties:
            entry = self.set_l2_penalties[set_name]
            return entry.l2_r, entry.l2_stenosis
        return self.default_l2_r, self.default_l2_stenosis


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    percent_train: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Default train fraction for CV geometry splits (learn-lpns-cv).",
    )
    data_processing_percent_train: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Default train fraction for learn-lpns-data-processing.",
    )
    cv_num_trials: int = Field(default=5, gt=0, description="Default k-fold CV trial count.")
    seed: int = Field(default=0, description="Default RNG seed for standalone split generation.")
    cv_trial_seed_stride: int = Field(
        default=1000,
        gt=0,
        description="CV trial RNG seed = trial_index * stride + resample_attempt.",
    )


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    init: float = Field(default=0.02, gt=0, description="Default Adam LR before per-coefficient override.")
    transition_steps: int = Field(default=1000, gt=0)
    decay_rate: float = Field(default=0.95, gt=0.0, lt=1.0)


class VesselArchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_layers: int = Field(default=2, gt=0)
    layer_width: int = Field(default=10, gt=0)


class RriCoefficientConfig(BaseModel):
    """One single-output network per RRI coefficient (R, S, or L)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Short id, e.g. R, S, L.")
    label: str = Field(description="Human-readable name for training logs.")
    target_output_column: int = Field(ge=0, le=2, description="Column in output_rri (0=R, 1=S, 2=L).")
    lr_init: float = Field(gt=0)
    junction_num_layers: int = Field(gt=0)
    junction_layer_width: int = Field(gt=0)
    junction_asymmetric_overestimate_weight: float = Field(gt=0)
    vessel_asymmetric_overestimate_weight: float = Field(gt=0)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_epochs: int = Field(default=500, gt=0)
    early_stop_loss_threshold: float = Field(default=1e-7, gt=0)
    batch_size_divisor: int = Field(
        default=10,
        gt=0,
        description="batch_size = ceil(n_train / batch_size_divisor).",
    )
    generation_weighted_loss_scale: float = Field(default=1.0, gt=0)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    vessel: VesselArchConfig = Field(default_factory=VesselArchConfig)
    rri_coefficients: tuple[RriCoefficientConfig, ...] = Field(
        default=(
            RriCoefficientConfig(
                name="R",
                label="Linear Resistor (R)",
                target_output_column=0,
                lr_init=0.01,
                junction_num_layers=2,
                junction_layer_width=10,
                junction_asymmetric_overestimate_weight=2000,
                vessel_asymmetric_overestimate_weight=10,
            ),
            RriCoefficientConfig(
                name="S",
                label="Stenosis Resistor (S)",
                target_output_column=1,
                lr_init=0.001,
                junction_num_layers=2,
                junction_layer_width=10,
                junction_asymmetric_overestimate_weight=100,
                vessel_asymmetric_overestimate_weight=10,
            ),
            RriCoefficientConfig(
                name="L",
                label="Inductor (L)",
                target_output_column=2,
                lr_init=0.01,
                junction_num_layers=4,
                junction_layer_width=20,
                junction_asymmetric_overestimate_weight=10000,
                vessel_asymmetric_overestimate_weight=1000,
            ),
        )
    )

    def batch_size_for_n_train(self, n_train: int) -> int:
        return max(1, math.ceil(max(n_train, 1) / self.batch_size_divisor))


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def apply_solver_parameters(simulation_parameters: dict, solver: SolverConfig) -> None:
    """Write svZeroDSolver simulation_parameters fields from config."""
    simulation_parameters["number_of_cardiac_cycles"] = solver.number_of_cardiac_cycles
    simulation_parameters["steady_initial"] = solver.steady_initial
    simulation_parameters["absolute_tolerance"] = solver.absolute_tolerance
    simulation_parameters["maximum_nonlinear_iterations"] = solver.maximum_nonlinear_iterations


def build_calibration_parameters(
    calibration: CalibrationConfig,
    *,
    quadratic_resistor: bool,
    penalty_on: bool,
    set_name: str | None,
) -> dict:
    """Build svzerodcalibrator calibration_parameters dict from config and run flags."""
    if quadratic_resistor and penalty_on:
        l2_r, l2_stenosis = calibration.l2_penalties_for_set(set_name)
    else:
        l2_r = 0.0
        l2_stenosis = 0.0

    return {
        "tolerance_gradient": calibration.tolerance_gradient,
        "tolerance_increment": calibration.tolerance_increment,
        "maximum_iterations": calibration.maximum_iterations,
        "calibrate_stenosis_coefficient": quadratic_resistor,
        "calibrate_capacitance": False,
        "set_capacitance_to_zero": False,
        "L2_penalty_R_poiseuille": l2_r,
        "L2_penalty_stenosis_coefficient": l2_stenosis,
        "L2_penalty_L": 0,
    }
