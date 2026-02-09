# Copilot Instructions for LLNL Datathon 2026

## Project Overview

This repository is for the LLNL Datathon 2026, focused on **agentic AI for data compression optimization** and scientific data analysis/visualization. The codebase is organized for rapid prototyping of agentic workflows, emphasizing reproducibility and modularity.

## Agentic Data Compression Optimization: Prototype Workflow

The core workflow is designed as an **iterative, tool-using agent** for discovering near-optimal compression strategies for large, heterogeneous Earth system NetCDF datasets, while preserving scientific fidelity.

### Agentic Workflow Steps

1. **Inspect dataset metadata**
   - Gather variable names, dimensions, dtypes
   - Record value ranges, missing values, coordinate structure
   - Analyze temporal and spatial resolution
2. **Hypothesize compression strategies per variable class**
   - Propose chunking strategies based on access patterns
   - Suggest encoding options (zlib level, shuffle, filters)
   - Identify candidates for precision reduction or quantization
3. **Execute compression experiments**
   - Apply candidate encodings to data subsets or copies
   - Generate compressed NetCDF or Zarr outputs
4. **Evaluate trade-offs**
   - Measure file size reduction
   - Compute error metrics (RMSE, max absolute error, relative error)
   - Optionally apply domain-specific thresholds
5. **Iterate and refine strategies**
   - Discard poor candidates
   - Adjust chunk shapes, precision, or encoding parameters
   - Re-run experiments as needed
6. **Recommend an encoding policy with justification**
   - Provide per-variable or per-class recommendations
   - Clearly explain size vs. fidelity trade-offs

### Outputs to Demo

- Before/after file sizes and compression ratios
- Quantitative error metrics (RMSE, max difference, relative error)
- Auto-generated “recommended encoding” table (per variable)
- Concise natural-language summary explaining the recommendations

## Key Directories & Files

- `agentic/`: Core Python package for agentic workflows
  - `main.py`: Main entry point for running workflows
  - `llm.py`, `tools.py`, `config.py`: LLM integration, tool definitions, configuration
  - `workflows/`: Modular workflow scripts (e.g., `compression.py`, `diagnostics.py`)
- `notebooks/`: Proof-of-concept notebooks for experimentation
- `scripts/`: Ad-hoc scripts for running or testing workflows
- `environment.yml`: Conda environment specification (all dependencies managed here)
- `pyproject.toml`: Python packaging and tool configuration

## Environment & Setup

- Always use the provided Conda environment:
  ```bash
  conda env create -f environment.yml
  conda activate datathon-env
  ```
- Update the environment with:
  ```bash
  conda env update -f environment.yml --prune
  ```

## Development Patterns

- Add new workflows as separate modules in `agentic/workflows/`
- Use the `agentic` package for all core logic; avoid duplicating logic in scripts or notebooks
- Notebooks are for prototyping and should import from `agentic` where possible
- Scripts in `scripts/` are for running workflows/utilities, not for core logic

## Conventions

- Follow existing file and module naming patterns (lowercase, underscores)
- Place all configuration in `config.py` or `environment.yml`
- Document new workflows and scripts with clear module-level docstrings

## Integration & Extensibility

- Integrate new tools or LLMs by extending `agentic/tools.py` or `agentic/llm.py`
- Add new workflows in `agentic/workflows/` and register them in `main.py` if needed
- Use the Conda environment for all dependencies; avoid pip installs outside `environment.yml`

## Examples

- See `agentic/workflows/compression.py` for a workflow module example
- See `scripts/example_workflow.py` for running a workflow from script

## Testing & Debugging

- No explicit test framework is present; add tests as scripts or notebooks if needed
- For debugging, use print statements or logging within modules

---

For questions or unclear conventions, review the main `README.md` or ask for clarification.
