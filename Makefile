.PHONY: notebook notebook-setup lfs

# Fetch Git LFS sample data (VTP + JSON under data/).
lfs:
	@./scripts/setup_git_lfs.sh

# One-shot: venv + deps + open examples/nn_parameter_comparison.ipynb (no C++ solver).
notebook:
	@./scripts/setup_notebook.sh --launch

# Install only (no Jupyter launch).
notebook-setup:
	@./scripts/setup_notebook.sh
