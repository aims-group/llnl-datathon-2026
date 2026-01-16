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

## Setup

Before running any code or notebooks, complete the following setup steps.

---

### 1. Install Conda Environment (Python Dependencies)

All Python dependencies are managed with Conda.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate datathon-env
```

If you update `environment.yml`, update your environment with:

```bash
conda env update -f environment.yml --prune
```

Verify the environment:

```bash
python -c "import xarray, numpy; print('Python environment OK')"
```

---

### 2. LLM Backend Options (Local vs Hosted)

This project supports **two LLM backends**:

- **LivAI (hosted)** — LLNL-provided API access to Claude and GPT models
- **Ollama (local)** — runs LLaMA 3.1 on your machine

You may use **either or both**, configured via environment variables.

---

### 3. Option A: Hosted LLM via LLNL LivAI

If you have access to an **LLNL LivAI API key**, you may use **hosted models** such as **Claude 3.5 Sonnet** or **GPT-4.1** instead of (or in addition to) local models.

LivAI provides:

- LLNL-approved provenance
- No local hardware requirements
- Access to higher-quality reasoning models

To use LivAI:

1. Obtain a LivAI API key from the
   [LLNL LivAI API key request page](https://llnl.servicenowservices.com/ess?id=kb_article_view&sysparm_article=KB0026324#a1)
2. Set `LLM_BACKEND=livai` in your `.env`
3. Select a LivAI-supported model, for example:

   - `gpt-5` (recommended default for agentic workflows)
   - `claude-sonnet-3.7` (strongest scientific reasoning and final summaries)
   - `gpt-4.1` (long-context processing)
   - `o4-mini` (fast, low-cost iteration)

### 3. Option B: Local LLM via Ollama (Optional)

Ollama is used as a **local LLM service** for agentic workflows.

> **Important:** Ollama is **not managed by Conda**.
> It must be installed at the system level.

Install Ollama by following the official instructions:
[https://ollama.com/download](https://ollama.com/download)

After installation, verify that Ollama is available:

```bash
ollama --version
```

If the command is not found, restart your terminal.

#### Start Ollama Service

On most systems, Ollama runs automatically in the background.
If not, start it manually:

```bash
ollama serve
```

Leave this running while using the project.

---

#### Pull the Local LLM Model (Ollama)

By default, this project uses **LLaMA 3.1** models via Ollama.

##### Recommended (most systems, ~16 GB RAM)

```bash
ollama pull llama3.1:8b
```

Test the model:

```bash
ollama run llama3.1:8b "Summarize the purpose of Earth system model diagnostics."
```

This model provides **sufficient reasoning quality for agentic scientific
workflows** while remaining lightweight enough for local prototyping and
proof-of-concept development.

##### Optional (larger-memory systems)

If your system has **≥32 GB RAM** or substantial GPU VRAM, you may use a larger
variant for improved reasoning quality:

```bash
ollama pull llama3.1:70b
```

> **Note:** The 70B model is significantly more resource-intensive and is not
> required for Datathon-scale prototypes.

---

### 4. Configure Environment Variables

1. Copy the template:

   ```bash
   cp .env.template .env
   ```

2. Edit `.env` to configure:

   - LLM backend (`ollama` or `livai`)
   - Model selection
   - Temperature and token limits

Example configurations:

**Local (default):**

```env
LLM_BACKEND=ollama
LLM_MODEL=llama3.1:8b
```

**LivAI (hosted):**

```env
LLM_BACKEND=livai
LLM_MODEL=claude-3.5-sonnet
LIVAI_API_KEY=your_key_here
```

---

### 5. Quick Sanity Check

From within the Conda environment, verify that the unified LLM interface works:

```bash
python - << EOF
from llm import call_llm, system_message

messages = [
    system_message(),
    {"role": "user", "content": "What is an Earth system model diagnostic?"}
]

result = call_llm(messages)
print(result["content"])
EOF
```

If this prints a sensible response, your setup is complete.

---

### Notes

- Local **Ollama + LLaMA 3.1** is ideal for rapid iteration and offline work
- **LivAI (Claude / GPT)** is recommended for final summaries and demos
- The backend can be switched **without changing code**, only via `.env`

This repository is designed for **prototyping and proof-of-concept work**, not
production deployment.
