
# LLNL Datathon 2026

This repository contains code, environment setup, and documentation for the LLNL Datathon held January 20–21, 2026. The project focuses on agentic AI for data compression optimization, scientific data analysis, and visualization using modern open-source tools.

## Directory Structure

```text
llnl-datathon-2026/
├── README.md           # Project overview and instructions
├── environment.yml     # Conda environment specification
└── ...                 # (Add your code, notebooks, and data here)
```

## Conda Environment Setup

All required dependencies for this project are specified in `environment.yml`.

To create and activate the environment:

```bash
conda env create -f environment.yml
conda activate datathon-env
```

If you update the environment file, run:

```bash
conda env update -f environment.yml --prune
```

## Project Purpose

This repository is intended to:

- Provide a reproducible environment for the LLNL Datathon 2026
- Enable rapid prototyping of agentic AI workflows for data compression and analysis
- Support scientific data processing, visualization, and large language model experimentation

For more details on the technologies and tools used, see `environment.yml` and project documentation.
