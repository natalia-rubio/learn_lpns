"""
Optional very chatty diagnostics for the zero-D / NN / geometric-params pipeline.

When ``VERBOSE_ZERO_D_PIPELINE`` is False (default):
  - ``generate_zerod_inputs.py`` omits per-junction NN mapping logs, prediction-range dumps,
    and passes ``verbose=False`` into junction feature extraction.
  - ``geometric_params.extract_vessel_junction_areas`` omits per-vessel area lines, outlet
    GID traces, segment distance debug, ``Added geometric_params …`` spam, etc.
    (CLI ``--verbose`` also enables outlet-metrics / EL-extension dumps in that function.)
  - ``centerline_path_extraction.process_geometric_input`` omits per-outlet BranchIdTmp / GID lines
    unless ``verbose=True`` (from ``--verbose`` on ``generate_zerod_inputs.py``).
  - ``bifurcation_splitting.split_junctions*`` omits per-junction cascade logs unless ``verbose=True``.

Set to True temporarily when debugging junction feature extraction, EL path metrics,
or NN prediction mapping.

EL adjustment in ``bifurcation_splitting.adjust_junction_boundaries_by_entrance_length`` follows
the ``verbose`` argument (no longer forced on). Pass ``--verbose`` to ``generate_zerod_inputs.py``
for EL junction-by-junction logs, absorb/rename traces, and connector conversion lines.
"""

VERBOSE_ZERO_D_PIPELINE = False
