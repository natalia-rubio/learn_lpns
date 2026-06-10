"""Grouped bar charts comparing zero-D parameters (R, S, L) across modalities."""

import json
import os

import numpy as np

from learn_lpns.visualizations.matplotlib_tex import configure_matplotlib_latex

_STYLE_MAP = {
    "geometric": {"color": "green", "label": "0D Poiseuille"},
    "NORMAL_JUNCTION": {"color": "red", "label": "0D $\\Delta P = 0$ Junction (Calibrated)"},
    "BloodVesselJunction": {"color": "orange", "label": "0D RRI Junction (Calibrated)"},
    "BloodVesselJunction_NN": {"color": "dodgerblue", "label": "0D RRI Junction (NN)"},
    "BloodVesselJunction_NN_plus_Vessel_NN": {
        "color": "orchid",
        "label": "0D RRI Junction (NN + Vessel NN)",
    },
    "NN_vessel": {"color": "chartreuse", "label": "0D NN Vessel Only"},
}


def _is_non_el_connector(vessel_name):
    if not vessel_name:
        return True
    vlower = vessel_name.lower()
    return "connector" in vlower and "connectorel" not in vlower


def plot_zero_d_parameter_bars(
    modality_json_paths, output_dir=None, output_name="zero_d_parameter_bars.png", verbose=False
):
    """
    Create grouped bar charts comparing R_poiseuille, stenosis_coefficient, and L
    across multiple modalities (e.g., geometric, NORMAL_JUNCTION, BloodVesselJunction).

    Args:
        modality_json_paths: dict mapping modality name -> path to calibrated JSON
        output_dir: directory to save the plot; if None uses directory of first JSON
        output_name: filename for saved PNG
        verbose: print progress
    """
    if not isinstance(modality_json_paths, dict) or len(modality_json_paths) == 0:
        if verbose:
            print("  ✗ modality_json_paths must be a non-empty dict")
        return None

    modality_data = {}
    for mod, path in modality_json_paths.items():
        if not path or not os.path.exists(path):
            if verbose:
                print(f"  ⚠ Skipping modality '{mod}': file not found: {path}")
            modality_data[mod] = None
            continue
        try:
            with open(path) as f:
                modality_data[mod] = json.load(f)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Failed to read {path}: {e}")
            modality_data[mod] = None

    base_mod = None
    if modality_data.get("bifurcations"):
        base_mod = "bifurcations"
    elif modality_data.get("geometric"):
        base_mod = "geometric"
    else:
        for k, v in modality_data.items():
            if v:
                base_mod = k
                break

    if base_mod is None:
        if verbose:
            print("  ✗ No valid modality JSONs found to determine vessel ordering")
        return None

    vessels = modality_data[base_mod].get("vessels", [])
    vessel_names = [
        v.get("vessel_name")
        for v in vessels
        if v.get("vessel_name") and "connector" not in v.get("vessel_name", "").lower()
    ]
    if len(vessel_names) > 10:
        if verbose:
            print(f"  ⚠ More than 10 vessels ({len(vessel_names)}). Limiting plot to first 10 vessels.")
        vessel_names = vessel_names[:10]

    junctions_to_include = set()
    for mod_json in modality_data.values():
        if not mod_json:
            continue
        for j in mod_json.get("junctions", []):
            jname = j.get("junction_name")
            if jname and j.get("junction_values"):
                junctions_to_include.add(jname)

    junction_outlet_labels = []
    junction_outlet_map = {}  # label -> (junction_name, outlet_index)
    for j in modality_data[base_mod].get("junctions", []):
        jname = j.get("junction_name")
        if not jname or jname not in junctions_to_include:
            continue
        for oi, vidx in enumerate(j.get("outlet_vessels", [])):
            try:
                vidx_int = int(vidx)
            except Exception:
                continue
            out_vname = vessels[vidx_int].get("vessel_name") if 0 <= vidx_int < len(vessels) else None
            if _is_non_el_connector(out_vname):
                continue
            label = f"{jname}:out{oi}"
            junction_outlet_labels.append(label)
            junction_outlet_map[label] = (jname, oi)

    vessel_names_extended = vessel_names + junction_outlet_labels
    if not vessel_names_extended:
        if verbose:
            print("  ✗ No vessels or junction outlets found in base modality JSON")
        return None

    params = [
        ("R_poiseuille", "Poiseuille resistance"),
        ("stenosis_coefficient", "Stenosis coefficient"),
        ("L", "Inductance"),
    ]
    modalities = list(modality_json_paths.keys())
    data_by_param = {p[0]: {mod: [] for mod in modalities} for p in params}

    for p_key, _ in params:
        for mod in modalities:
            mod_json = modality_data.get(mod)
            vmap = {}
            if mod_json:
                for v in mod_json.get("vessels", []):
                    name = v.get("vessel_name")
                    if name:
                        vmap[name] = v.get("zero_d_element_values", {})
            jmap = {}
            if mod_json:
                for j in mod_json.get("junctions", []):
                    jn = j.get("junction_name")
                    if jn:
                        jmap[jn] = j.get("junction_values", {})

            values = []
            for name in vessel_names_extended:
                if name in vmap:
                    v = vmap[name].get(p_key)
                    try:
                        values.append(float(v) if v is not None else np.nan)
                    except Exception:
                        values.append(np.nan)
                elif name in junction_outlet_map:
                    jname, out_idx = junction_outlet_map[name]
                    arr = jmap.get(jname, {}).get(p_key) if jname in jmap else None
                    try:
                        if isinstance(arr, list | tuple) and len(arr) > out_idx:
                            values.append(float(arr[out_idx]))
                        elif jname in jmap:
                            values.append(np.nan)
                        else:
                            values.append(0.0)
                    except Exception:
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            data_by_param[p_key][mod] = values

    if output_dir is None:
        first_path = next((p for p in modality_json_paths.values() if p and os.path.exists(p)), None)
        output_dir = os.path.dirname(first_path) if first_path else os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import patches as mpatches

        n_v = len(vessel_names_extended)
        x = np.arange(n_v)
        n_mod = len(modalities)
        width = 0.7 / n_mod if n_mod > 0 else 0.2

        configure_matplotlib_latex(plt)
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["legend.fontsize"] = 12

        fig, axes = plt.subplots(3, 1, figsize=(max(8, n_v * 0.3 + 6), 10), sharex=True)
        cycle_colors = [_STYLE_MAP.get(mod, {"color": "gray"})["color"] for mod in modalities]
        legend_patches = [
            mpatches.Patch(
                color=_STYLE_MAP.get(mod, {"color": "gray", "label": mod})["color"],
                label=_STYLE_MAP.get(mod, {"color": "gray", "label": mod})["label"],
            )
            for mod in modalities
        ]

        for i, (p_key, p_label) in enumerate(params):
            ax = axes[i]
            for j, mod in enumerate(modalities):
                vals = np.array(data_by_param[p_key][mod], dtype=float)
                offsets = x - 0.35 + j * width + width / 2.0
                col = _STYLE_MAP.get(mod, {"color": "gray"})["color"]
                if mod not in _STYLE_MAP:
                    col = cycle_colors[j % len(cycle_colors)]
                ax.bar(offsets, vals, width=width, color=col)
            ax.set_ylabel(p_label, fontsize=20)
            ax.grid(True, alpha=0.3)

        if legend_patches:
            fig.legend(
                handles=legend_patches,
                loc="upper center",
                ncol=max(1, len(legend_patches)),
                bbox_to_anchor=(0.5, 0.995),
            )
        fig.suptitle(r"Zero-D parameter comparison: $R$, $S$, and $L$", y=1.005, fontsize=18)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(vessel_names_extended, rotation=90, fontsize=8)
        plt.subplots_adjust(top=0.88)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.savefig(out_path.replace(".png", ".pdf"), dpi=150, bbox_inches="tight")
        plt.close()

        if verbose:
            print(f"  ✓ Saved zero-D parameter bar chart to: {out_path}")
        return out_path
    except Exception as e:
        if verbose:
            print(f"  ✗ Failed to create parameter bar chart: {e}")
        return None
