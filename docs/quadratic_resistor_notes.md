# Quadratic resistor (RRI) — behavior and troubleshooting

Notes on why **`quadratic_resistor`** run configs can produce **worse** CV / forward-simulation results than **`gen_loss`** (RI-only), and what to check.

## What `quadratic_resistor` changes

The `quadratic_resistor` token in `--run_config` turns on the **RRI** junction model: calibrate and predict **R**, **stenosis S**, and **L** (instead of R + L only with `S = 0`).

| Stage | `gen_loss` (default) | `quadratic_resistor_gen_loss` |
| ----- | -------------------- | ----------------------------- |
| Calibration input | `stenosis_coefficient = 0`, `calibrate_stenosis_coefficient = False` | Stenosis kept from geometry as initial value; stenosis calibrated |
| Calibrated labels | R, L only (S = 0) | R, S, L |
| NN training | Three nets trained (R, S, L) | Same |
| NN inference | `pred_S` **zeroed** before applying to JSON | `pred_S` **used** in forward config |
| Forward physics | Linear in Q (R, L) | Nonlinear `\|Q\|Q` term via S |

Related tokens:

- **`penalty_on`** (requires `quadratic_resistor`): L2 penalties on R and S during **C++** calibration only.
- **`gen_loss`**: generation-weighted NN training — can be combined with either RI or RRI.

`--calibration_backend` (`decoupled_ls` vs `svzerod`) is **independent** of `run_config`; see [usage.md](usage.md#calibration-backend---calibration_backend).

## Why worse results are plausible (not necessarily a bug)

### 1. Ill-conditioned calibration labels (primary suspect)

Default calibration is **decoupled least squares** per vessel/junction outlet:

\[
P_{in}(t) - P_{out}(t) \approx R\,Q_{in}(t) + S\,|Q_{in}(t)|\,Q_{in}(t) + L\,\dot Q_{out}(t)
\]

Implementation: `learn_lpns/zerod_calibration/decoupled_ls_calibration.py`.

- Only **L** is clamped \(\geq 0\); **R and S can be negative** and **trade off** in a local fit.
- With `quadratic_resistor_gen_loss` and **without** `penalty_on`, there is **no L2 regularization** toward geometric priors.
- RI-only (`gen_loss`) is a **two-parameter** fit (R, L) with S fixed at 0 — usually much more stable.

**Example — `VMR_pulmo` / `quadratic_resistor_gen_loss` / 5-geo summary**  
(`data/feature_histograms/VMR_pulmo/quadratic_resistor_gen_loss/bifurcations_EL/all/data_summary_VMR_pulmo_num_geos_5.csv`):

| Output | min | max | std |
| ------ | --- | --- | --- |
| `R_poiseuille_outlet0` | −1509 | 1428 | ~209 |
| `stenosis_coefficient_outlet0` | −814 | 3843 | ~209 |
| `L_outlet0` | 0.19 | 154 | ~21 |

Those ranges are poor supervision for the stenosis NN head.

### 2. Decoupled fit vs global coupling

Decoupled LS fits each element from **local** P/Q and ignores network constraints. For RRI, **R and S are highly correlated** in \(\Delta P = RQ + S|Q|Q + L\dot Q\).

The C++ **`svzerodcalibrator`** minimizes **governing-equation residuals** globally (Levenberg–Marquardt). Labels from the two backends **generally differ**; comparing old `gen_loss` (possibly C++) vs new RRI (decoupled) mixes calibration quality with model choice.

### 3. Nonlinear forward sensitivity

With `S = 0`, pressure drop is linear in Q. With `S \neq 0`, the **\|Q\|Q** term is nonlinear and flow-sensitive. Similar relative NN error on S can produce **larger** pressure/flow MSE than error on R alone.

### 4. Harder NN problem

With `quadratic_resistor` on:

- The **S network output is used** at inference (`nn_inference.py` zeroes `pred_S` only when `quadratic_resistor` is off).
- All three single-output nets are still trained (`launch_training.py` loops `training.rri_coefficients`).
- The S head uses a **smaller** architecture in `config/defaults.yaml` (`junction_num_layers: 1`, `junction_layer_width: 2`, `lr_init: 0.001`) while targets can span orders of magnitude.

With `gen_loss`, the S net trains on ~0 targets and predictions are discarded — effectively only **R and L** must be good at deploy.

### 5. Physics / cohort mismatch

For some cohorts (e.g. **pulmonary**), a quadratic resistance at **bifurcation junctions** may not match 1D centerline behavior as well as RI+L. Extra degrees of freedom can **fit noise** rather than structure. Simpler RI can generalize better even if RRI is “more complete” on paper.

### 6. Comparisons are not apples-to-apples

`gen_loss` vs `quadratic_resistor_gen_loss` differ in:

- On-disk paths and **calibrated ground truth**
- **Forward physics** at deploy (`S = 0` vs `S \neq 0`)
- Possibly **calibration backend** if data was built at different times

Worse CV MSE does not by itself mean stenosis is wrong in principle — it can mean the **RRI + decoupled calibration + NN** stack is less identifiable on that cohort.

## Diagnostic checklist

1. **Calibrated baseline MSE** (forward sim with calibrated JSON, no NN): if RRI calibrated is already worse than RI calibrated, the issue is calibration/physics, not the NN. Done, calibrated RSL outperformed calibrated RL in cross validation.
2. **Label quality**: inspect `data/ml_inputs/.../junction_lumped_parameters.csv` or feature histograms for negative or extreme **S** (and **R**). (This is happening, it is a real phenomenon.)
3. **Decoupled fit warnings**: run calibration and watch for `UserWarning` when relative RMS fit error exceeds 10% (`decoupled_ls_calibration.py`). 
4. **Fit plots**: enable `--plot_rsl_fits` on `generate_zerod_inputs` / batch (or `calibration.plot_rsl_fits: true` in config) to write per-element \(\Delta P\) vs \(Q\) plots under `results/RSL_fits/<set_name>/<geo_name>/`.  done. these look ok
5. **Try alternatives**:
   - `--calibration_backend svzerod` (global C++ calibrator; needs `SVZEROD_INSTALL_DIR`)
   - `--run_config quadratic_resistor_penalty_on_gen_loss` (L2 on R and S in C++ calibrator only)
7. **Clamp resistances**: doesn't take care of the case where stenosis is very negative and resistance is very positive.  Also, R is rarely negative with RSL fit, which is good.
8. **One NN predicts both R and S**: can the nn learn the compensation mechanism?
9. **Add "speedup" feature?**

## Commands (reference)

```bash
# RI-only (default physics path)
learn-lpns-cv --set_name VMR_pulmo --run_config gen_loss

# RRI + generation-weighted training (no L2 calibration penalty)
learn-lpns-cv --set_name VMR_pulmo --run_config quadratic_resistor_gen_loss

# RRI + L2 penalties during C++ calibration (if using svzerod backend)
learn-lpns-cv --set_name VMR_pulmo --run_config quadratic_resistor_penalty_on_gen_loss --calibration_backend svzerod
```

## Related code

| Topic | Location |
| ----- | -------- |
| Run-config tokens | `learn_lpns/zerod_calibration/run_config_canonical.py` |
| Calibration input (zero S when off) | `learn_lpns/zerod_calibration/calibration.py` → `create_calibration_input` |
| Decoupled R/S/L fit | `learn_lpns/zerod_calibration/decoupled_ls_calibration.py` |
| Zero S at inference when off | `learn_lpns/zerod_calibration/nn_inference.py` |
| NN training (always R, S, L) | `learn_lpns/neural_network/launch_training.py` |
| Training hyperparameters | `config/defaults.yaml` → `training.rri_coefficients` |

## Summary

Turning on **quadratic resistor** adds a **poorly identified, often unregularized, nonlinear** parameter from **decoupled** local fits, then asks the NN to learn it. **RI-only** collapses to a stabler R+L problem with S unused at inference — which can yield **better** CV metrics even when RRI looks more physical. Treat unexpected regressions as a signal to inspect **calibration labels** and **baseline calibrated MSE** before tuning the NN.
