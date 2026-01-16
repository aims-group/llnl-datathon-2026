# Copilot Instructions for LLNL Datathon 2026

## Project Overview
- This repository is for the LLNL Datathon 2026, focused on agentic AI for data compression, scientific data analysis, and visualization.
- The codebase is organized for rapid prototyping of agentic workflows, with a strong emphasis on reproducibility and modularity.

## Key Directories & Files
- `agentic/`: Core Python package for agentic workflows.
  - `main.py`: Likely the main entry point for running workflows.
  - `llm.py`, `tools.py`, `config.py`: Contain logic for LLM integration, tool definitions, and configuration.
  - `workflows/`: Contains modular workflow scripts (e.g., `compression.py`, `diagnostics.py`).
- `notebooks/`: Proof-of-concept notebooks for experimentation.
- `scripts/`: Ad-hoc scripts for running or testing workflows.
- `environment.yml`: Conda environment specification. All dependencies should be managed here.
- `pyproject.toml`: For Python packaging and tool configuration.

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
- Workflows are modularized in `agentic/workflows/`. Add new workflows as separate modules.
- Use the `agentic` package for all core logic; avoid duplicating logic in scripts or notebooks.
- Notebooks are for prototyping and should import from `agentic` where possible.
- Scripts in `scripts/` are for running workflows or utilities, not for core logic.

## Conventions
- Follow existing file and module naming patterns (lowercase, underscores).
- Place all configuration in `config.py` or `environment.yml`.
- Document new workflows and scripts with clear module-level docstrings.

## Integration & Extensibility
- Integrate new tools or LLMs by extending `agentic/tools.py` or `agentic/llm.py`.
- Add new workflows in `agentic/workflows/` and register them in `main.py` if needed.
- Use the Conda environment for all dependencies; avoid pip installs outside `environment.yml`.

## Examples
- See `agentic/workflows/compression.py` for a workflow module example.
- See `scripts/example_workflow.py` for running a workflow from script.

## Testing & Debugging
- No explicit test framework is present; add tests as scripts or notebooks if needed.
- For debugging, use print statements or logging within modules.

---
For questions or unclear conventions, review the main `README.md` or ask for clarification.
