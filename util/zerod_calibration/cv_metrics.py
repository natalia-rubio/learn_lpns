"""Shared helpers for cross-validation MSE summary CSV read/write."""

import csv
import os

import numpy as np

MSE_METRIC_KEYS = (
    "overall_mse",
    "mean_pressure_mse",
    "mean_flow_mse",
    "overall_max_error",
    "mean_pressure_max_error",
    "mean_flow_max_error",
    "overall_max_rel_error",
    "mean_pressure_max_rel_error",
    "mean_flow_max_rel_error",
)

METRIC_CSV_SUFFIXES = (
    ("_pressure_mse.csv", "PressureMSE_"),
    ("_flow_mse.csv", "FlowMSE_"),
    ("_max_error.csv", "MaxError_"),
    ("_pressure_max_error.csv", "PressureMaxError_"),
    ("_flow_max_error.csv", "FlowMaxError_"),
    ("_max_rel_error.csv", "MaxRelError_"),
    ("_pressure_max_rel_error.csv", "PressureMaxRelError_"),
    ("_flow_max_rel_error.csv", "FlowMaxRelError_"),
)

_MODALITY_PREFIXES = (
    ("MSE_", 4),
    ("PressureMSE_", len("PressureMSE_")),
    ("FlowMSE_", len("FlowMSE_")),
    ("MaxError_", len("MaxError_")),
    ("PressureMaxError_", len("PressureMaxError_")),
    ("FlowMaxError_", len("FlowMaxError_")),
    ("MaxRelError_", len("MaxRelError_")),
    ("PressureMaxRelError_", len("PressureMaxRelError_")),
    ("FlowMaxRelError_", len("FlowMaxRelError_")),
)


def read_cv_summary_rows(summary_path):
    """Return list of (trial_id, val_geometries_str, val_geometries_list)."""
    if not os.path.exists(summary_path):
        return None
    rows = []
    with open(summary_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or header[0] != "trial_id":
            return None
        for row in reader:
            if not row or row[0] in ("", "mean", "std"):
                break
            try:
                trial_id = int(row[0])
            except ValueError:
                break
            val_geometries_str = row[1] if len(row) > 1 else ""
            val_geometries = [g.strip() for g in val_geometries_str.split(",") if g.strip()]
            rows.append((trial_id, val_geometries_str, val_geometries))
    return rows


def collect_modalities(results):
    """Collect sorted modality names from trial result row dicts."""
    out = set()
    for row in results:
        if "error" in row:
            continue
        for key in row:
            for prefix, prefix_len in _MODALITY_PREFIXES:
                if key.startswith(prefix) and len(key) > prefix_len:
                    out.add(key[prefix_len:])
                    break
    return sorted(out)


def empty_trial_metrics():
    return {k: [] for k in MSE_METRIC_KEYS}


def accumulate_trial_metrics(trial_metrics, mod_data):
    """Merge parsed per-geometry MSE dict into trial_metrics lists."""
    for mod, metrics in mod_data.items():
        if mod not in trial_metrics:
            trial_metrics[mod] = empty_trial_metrics()
        for key, value in metrics.items():
            if key in trial_metrics[mod]:
                trial_metrics[mod][key].append(value)


def trial_metrics_to_row(trial_id, val_geometries_str, trial_metrics):
    """Build one CV summary row dict from accumulated trial metrics."""
    row = {"trial_id": trial_id, "val_geometries": val_geometries_str}
    for mod, data in trial_metrics.items():
        row[f"MSE_{mod}"] = np.nanmean(data["overall_mse"]) if data["overall_mse"] else np.nan
        row[f"PressureMSE_{mod}"] = np.nanmean(data["mean_pressure_mse"]) if data["mean_pressure_mse"] else np.nan
        row[f"FlowMSE_{mod}"] = np.nanmean(data["mean_flow_mse"]) if data["mean_flow_mse"] else np.nan
        row[f"MaxError_{mod}"] = np.nanmean(data["overall_max_error"]) if data["overall_max_error"] else np.nan
        row[f"PressureMaxError_{mod}"] = np.nanmean(data["mean_pressure_max_error"]) if data["mean_pressure_max_error"] else np.nan
        row[f"FlowMaxError_{mod}"] = np.nanmean(data["mean_flow_max_error"]) if data["mean_flow_max_error"] else np.nan
        row[f"MaxRelError_{mod}"] = np.nanmean(data["overall_max_rel_error"]) if data.get("overall_max_rel_error") else np.nan
        row[f"PressureMaxRelError_{mod}"] = np.nanmean(data["mean_pressure_max_rel_error"]) if data.get("mean_pressure_max_rel_error") else np.nan
        row[f"FlowMaxRelError_{mod}"] = np.nanmean(data["mean_flow_max_rel_error"]) if data.get("mean_flow_max_rel_error") else np.nan
    return row


def _metric_values(all_trial_results, col):
    vals = [r.get(col) for r in all_trial_results if "error" not in r and col in r]
    return [
        float(v) for v in vals
        if v is not None and v != "" and (not isinstance(v, float) or not np.isnan(v))
    ]


def write_metric_summary_csv(path, prefix, modalities, all_trial_results):
    """Write one metric-type CV summary CSV with mean/std footer rows."""
    cols = [prefix + mod for mod in modalities]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "val_geometries"] + cols)
        for row in all_trial_results:
            if "error" in row:
                writer.writerow([row.get("trial_id", ""), row.get("val_geometries", "")] + [""] * len(cols))
            else:
                writer.writerow([row.get("trial_id", ""), row.get("val_geometries", "")] + [row.get(c, "") for c in cols])
        writer.writerow([])
        mean_row = ["mean", ""]
        std_row = ["std", ""]
        for col in cols:
            vals = _metric_values(all_trial_results, col)
            mean_row.append(np.nanmean(vals) if vals else np.nan)
            std_row.append(np.nanstd(vals) if len(vals) > 1 else (0.0 if vals else np.nan))
        writer.writerow(mean_row)
        writer.writerow(std_row)


def write_all_cv_summary_csvs(summary_path, out_dir, geometry_variant, all_trial_results):
    """Write main MSE summary plus per-metric companion CSVs."""
    modalities = collect_modalities(all_trial_results)
    os.makedirs(out_dir, exist_ok=True)

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "val_geometries"] + [f"MSE_{m}" for m in modalities])
        for row in all_trial_results:
            if "error" in row:
                writer.writerow([row.get("trial_id", ""), row.get("val_geometries", "")] + [""] * len(modalities))
            else:
                writer.writerow(
                    [row.get("trial_id", ""), row.get("val_geometries", "")]
                    + [row.get(f"MSE_{m}", "") for m in modalities]
                )
        writer.writerow([])
        mean_row = ["mean", ""]
        std_row = ["std", ""]
        for mod in modalities:
            col = f"MSE_{mod}"
            vals = _metric_values(all_trial_results, col)
            mean_row.append(np.nanmean(vals) if vals else np.nan)
            std_row.append(np.nanstd(vals) if len(vals) > 1 else (0.0 if vals else np.nan))
        writer.writerow(mean_row)
        writer.writerow(std_row)

    base = os.path.join(out_dir, f"{geometry_variant}_cv_summary")
    for suffix, prefix in METRIC_CSV_SUFFIXES:
        write_metric_summary_csv(base + suffix, prefix, modalities, all_trial_results)

    return summary_path, modalities
