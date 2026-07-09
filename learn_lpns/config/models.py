"""Pydantic models for centralized pipeline configuration."""

from __future__ import annotations

import math

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PhysicsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rho: float = Field(default=1.06, gt=0, description="Blood density (g/cm^3)")
    mu: float = Field(default=0.04, gt=0, description="Blood viscosity (Poise)")
    reference_reynolds: float = Field(
        default=4500.0,
        gt=0,
        description=(
            "Reference Reynolds number Re_c for physics-based R/S/L non-dimensionalization "
            "(arXiv:2508.21165 Eq. 9)."
        ),
    )


class SolverConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    absolute_tolerance: float = Field(default=1e-5, gt=0)
    maximum_nonlinear_iterations: int = Field(default=50, gt=0)
    number_of_cardiac_cycles: int = Field(default=1, gt=0)
    steady_initial: bool = False


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    tolerance_gradient: float = Field(default=1e-4, gt=0)
    tolerance_increment: float = Field(default=1e-4, gt=0)
    maximum_iterations: int = Field(default=20, gt=0)
    default_l2_r: float = Field(default=1e5, gt=0)
    default_l2_stenosis: float = Field(default=1e10, gt=0)
    decoupled_l2_r: float = Field(
        default=0.0,
        ge=0,
        description="Tikhonov L2 on R in decoupled_ls (penalty lambda * R^2 toward 0)",
    )
    decoupled_l2_stenosis: float = Field(
        default=0.0,
        ge=0,
        description="Tikhonov L2 on S in decoupled_ls (penalty lambda * S^2 toward 0)",
    )
    decoupled_l2_l: float = Field(
        default=0.0,
        ge=0,
        description="Tikhonov L2 on L in decoupled_ls (penalty lambda * L^2 toward 0)",
    )
    plot_rsl_fits: bool = Field(
        default=False,
        description="When using decoupled_ls, save per-element dP vs Q fit plots under results/RSL_fits/",
    )
    decoupled_nonneg_r: bool = Field(
        default=True,
        description=(
            "When using decoupled_ls with quadratic_resistor (calibrate_stenosis_coefficient), "
            "enforce R_poiseuille >= 0 in bounded least-squares fits. Ignored when S is not calibrated."
        ),
    )
    decoupled_nonneg_l: bool = Field(
        default=True,
        description="When using decoupled_ls, enforce L >= 0 in bounded least-squares fits.",
    )


class CohortCalibrationOverrides(BaseModel):
    """Optional per-cohort calibration overrides (future use under cohorts.sets.<name>)."""

    model_config = ConfigDict(frozen=True)

    l2_r: float = Field(gt=0)
    l2_stenosis: float = Field(gt=0)


class CohortEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    display_label: str
    calibration: CohortCalibrationOverrides | None = None


class CohortsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_cv_set_names: tuple[str, ...] = Field(default_factory=tuple)
    sets: dict[str, CohortEntry] = Field(default_factory=dict)

    def display_label_for(self, set_name: str) -> str | None:
        entry = self.sets.get(set_name)
        return entry.display_label if entry else None

    def format_display_label(self, set_name: str) -> str:
        """Plot/LaTeX label from config, with VMR_ fallback for unknown sets."""
        label = self.display_label_for(set_name)
        if label:
            return label
        if set_name.startswith("VMR_"):
            return set_name.replace("VMR_", "VMR\n", 1)
        return set_name

    def l2_penalties_for_set(
        self,
        set_name: str | None,
        calibration: CalibrationConfig,
    ) -> tuple[float, float]:
        """Return L2 penalties, using cohort calibration overrides when present."""
        if set_name and set_name in self.sets:
            overrides = self.sets[set_name].calibration
            if overrides is not None:
                return overrides.l2_r, overrides.l2_stenosis
        return calibration.default_l2_r, calibration.default_l2_stenosis


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
    cv_max_parallel_trials: int = Field(
        default=1,
        ge=1,
        description="Max CV trials to run in parallel (1 = serial, current default).",
    )
    cv_worker_cpu_threads: int | None = Field(
        default=None,
        gt=0,
        description=(
            "When cv_max_parallel_trials > 1, cap OMP/XLA CPU threads per worker subprocess. "
            "Omit to leave environment defaults unchanged."
        ),
    )


FlowSplitMethod = Literal["mean_over_time", "peak_inlet_flow"]
ActivationName = Literal["relu", "leaky_relu", "tanh"]


class StenosisGenerationLimitConfig(BaseModel):
    """Gate stenosis S to low-generation elements (RSL vs RL calibration, S training, inference)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=False,
        description=(
            "When true (with quadratic_resistor), fit/train/predict S only for elements at or below "
            "the junction/vessel max_generation thresholds; higher-generation elements use RL (S=0)."
        ),
    )
    junction_max_generation: float = Field(
        default=1.0,
        ge=0.0,
        description="Inclusive junction stenosis generation bound (inlet-vessel generation per junction row).",
    )
    vessel_max_generation: float = Field(
        default=1.0,
        ge=0.0,
        description="Inclusive vessel stenosis generation bound (each vessel's own bifurcation generation).",
    )
    max_generation: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Deprecated legacy alias: when set, overrides both junction_max_generation and "
            "vessel_max_generation to this value."
        ),
    )

    def junction_limit(self) -> float:
        if self.max_generation is not None:
            return float(self.max_generation)
        return float(self.junction_max_generation)

    def vessel_limit(self) -> float:
        if self.max_generation is not None:
            return float(self.max_generation)
        return float(self.vessel_max_generation)


class DataProcessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_split_method: FlowSplitMethod = Field(
        default="mean_over_time",
        description=(
            "How to compute flow_split from geometric simulation CSV: "
            "mean_over_time averages outlet/inlet ratios; "
            "peak_inlet_flow uses the timestep of maximum original-inlet flow."
        ),
    )
    stenosis_generation_limit: StenosisGenerationLimitConfig = Field(
        default_factory=StenosisGenerationLimitConfig,
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
    num_epochs: int | None = Field(
        default=None,
        gt=0,
        description="Max training epochs for this coefficient (junction). Omit to use training.num_epochs.",
    )
    vessel_num_epochs: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Max training epochs for this coefficient (vessel). "
            "Omit to use num_epochs, then training.num_epochs."
        ),
    )
    leaky_relu: bool | None = Field(
        default=None,
        description="Deprecated legacy flag; when set, maps to leaky_relu or relu before activation.",
    )
    activation: ActivationName | None = Field(
        default=None,
        description=(
            "Hidden-layer activation for this coefficient's MLP "
            "(relu, leaky_relu, tanh). Omit to use training.activation."
        ),
    )
    generation_weighted_loss_decay_base: float | None = Field(
        default=None,
        gt=1.0,
        description=(
            "gen_loss sample-weight decay base for this coefficient "
            "(weight = 1 / base^generation). Omit to use training.generation_weighted_loss_decay_base."
        ),
    )
    restore_best_weights: bool | None = Field(
        default=None,
        description=(
            "When true, save weights from the epoch with lowest weighted train training loss. "
            "Omit to use training.restore_best_weights."
        ),
    )
    early_stop_loss_threshold: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Stop when weighted train training loss drops below this. "
            "Omit to use training.early_stop_loss_threshold."
        ),
    )

    def training_epochs(self, *, vessel: bool, default: int) -> int:
        """Resolve epoch count for junction or vessel training."""
        if vessel and self.vessel_num_epochs is not None:
            return self.vessel_num_epochs
        if self.num_epochs is not None:
            return self.num_epochs
        return default

    def effective_activation(self, default: ActivationName) -> ActivationName:
        """Resolve activation for this coefficient."""
        if self.activation is not None:
            return self.activation
        if self.leaky_relu is not None:
            return "leaky_relu" if self.leaky_relu else "relu"
        return default

    def effective_generation_weighted_loss_decay_base(self, default: float) -> float:
        """Resolve gen_loss decay base for this coefficient."""
        if self.generation_weighted_loss_decay_base is not None:
            return float(self.generation_weighted_loss_decay_base)
        return default

    def effective_restore_best_weights(self, default: bool) -> bool:
        """Resolve whether to restore best train-loss weights for this coefficient."""
        if self.restore_best_weights is not None:
            return self.restore_best_weights
        return default

    def effective_early_stop_loss_threshold(self, default: float) -> float:
        """Resolve early-stop threshold for this coefficient."""
        if self.early_stop_loss_threshold is not None:
            return float(self.early_stop_loss_threshold)
        return default


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_epochs: int = Field(
        default=500,
        gt=0,
        description="Default max epochs per R/S/L network when a coefficient omits num_epochs.",
    )
    multi_output_rri: bool = Field(
        default=False,
        description="Train one network with R/S/L outputs instead of three single-output networks.",
    )
    activation: ActivationName = Field(
        default="relu",
        description=(
            "Default hidden-layer activation for each R/S/L MLP when a coefficient omits activation. "
            "Choices: relu, leaky_relu, tanh. Stored on each model checkpoint for inference."
        ),
    )
    leaky_relu: bool | None = Field(
        default=None,
        description="Deprecated global flag; when set, overrides activation for all coefficients.",
    )
    clip_predictions: bool = Field(
        default=False,
        description=(
            "Clip R/S/L NN predictions at inference to train-set output_rri min/max "
            "stored on the model checkpoint."
        ),
    )
    nondimensionalize_rsl: bool = Field(
        default=False,
        description=(
            "Train on physics-non-dimensionalized R*, S*, L* targets and re-dimensionalize "
            "predictions to physical units at inference (arXiv:2508.21165)."
        ),
    )
    restore_best_weights: bool = Field(
        default=True,
        description=(
            "When true, restore and save weights from the epoch with lowest weighted train training loss."
        ),
    )
    early_stop_loss_threshold: float = Field(
        default=1e-7,
        gt=0,
        description="Stop when weighted train training loss drops below this.",
    )
    quiet_epochs: bool = Field(
        default=False,
        description="Suppress per-epoch train/validation loss lines during train_nn.",
    )
    batch_size_divisor: int = Field(
        default=10,
        gt=0,
        description="batch_size = ceil(n_train / batch_size_divisor).",
    )
    generation_weighted_loss_decay_base: float = Field(
        default=2.0,
        gt=1.0,
        description=(
            "Default gen_loss decay base per R/S/L network when a coefficient omits "
            "generation_weighted_loss_decay_base (sample weight = 1 / base^generation)."
        ),
    )
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

    def resolved_activation(self) -> ActivationName:
        """Global default activation, honoring legacy leaky_relu when present."""
        if self.leaky_relu is not None:
            return "leaky_relu" if self.leaky_relu else "relu"
        return self.activation


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    cohorts: CohortsConfig = Field(default_factory=CohortsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
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
    set_name: str | None = None,
    cohorts: CohortsConfig | None = None,
) -> dict:
    """Build svzerodcalibrator calibration_parameters dict from config and run flags."""
    if quadratic_resistor and penalty_on:
        if cohorts is not None:
            l2_r, l2_stenosis = cohorts.l2_penalties_for_set(set_name, calibration)
        else:
            l2_r, l2_stenosis = calibration.default_l2_r, calibration.default_l2_stenosis
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
        "freeze_connector_segments": True,
    }
