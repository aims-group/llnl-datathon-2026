import json
import os
from typing import Dict

import numpy as np
import xarray as xr


from agentic.llm import call_llm, system_message
from xcdat.temporal import _infer_freq


def _inspect_dataset(ds: xr.Dataset) -> dict:
    """
    Inspect an xarray Dataset and return a lightweight, metadata-only summary.

    This function is intentionally cheap:
    - No data variable values are loaded
    - No full-array statistics are computed
    - Time frequency is inferred from coordinates only (best-effort)

    Returns
    -------
    dict
        Dataset summary suitable for agent reasoning and planning.
    """
    summary: dict = {}

    # ------------------------------------------------------------------
    # Global structure
    # ------------------------------------------------------------------
    summary["dimensions"] = dict(ds.sizes)

    # ------------------------------------------------------------------
    # Data variables
    # ------------------------------------------------------------------
    variables: dict = {}

    for name, da in ds.data_vars.items():
        variables[name] = {
            "dims": da.dims,
            "shape": da.shape,
            "dtype": str(da.dtype),
            "size_mb": da.nbytes / 1e6,
            "coords": list(da.coords),
        }

    summary["variables"] = variables

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    coordinates: dict = {}

    for name, coord in ds.coords.items():
        entry = {
            "dims": coord.dims,
            "shape": coord.shape,
            "dtype": str(coord.dtype),
        }

        # Best-effort time frequency inference (planning only)
        if name == "time" and coord.sizes.get("time", 0) > 1:
            entry["time_frequency"] = _infer_freq(coord)

        coordinates[name] = entry

    summary["coordinates"] = coordinates

    return summary


def _propose_compression_plan(summary: dict) -> dict:
    """
    Propose a compression plan by classifying variables into categories.

    Classification meanings:
        - state: Core/prognostic model variables that represent the evolving
        physical state (e.g., temperature, pressure). These require high
        precision and should be protected from lossy compression.
        - diagnostic: Derived or secondary variables, often used for analysis or
        visualization (e.g., cloud fraction, energy fluxes). Medium precision is
        acceptable; moderate compression can be considered.
        - index: Coordinate, label, or integer-like variables (e.g., time,
        latitude, longitude, bounds). Low precision is acceptable; these can be
        aggressively compressed or quantized.

    This classification guides the agent in selecting appropriate compression
    strategies for each variable, balancing file size reduction and scientific
    fidelity.

    Parameters
    ----------
    summary : dict
        Summary of the NetCDF dataset, including variable metadata and
        statistics.

    Returns
    -------
    dict
        Compression plan with variable classifications and notes. Example
        schema:
        {
            "variable_classes": {
                "<var_name>": "state" | "diagnostic" | "index"
            },
            "notes": "<brief explanation of assumptions>"
        }


    """
    content = f"""
        Given the following NetCDF dataset summary:

        {summary}

        Classify variables into one of the following categories:
        - "state"        (high precision required)
        - "diagnostic"   (medium precision acceptable)
        - "index"        (low precision / integer-like)

        Return JSON ONLY with this schema:

        {{
        "variable_classes": {{
            "<var_name>": "state" | "diagnostic" | "index"
        }},
        "notes": "<brief explanation of assumptions>"
        }}

        Do NOT include code.
        Do NOT include prose outside JSON.
    """

    messages = [
        system_message(),
        {"role": "user", "content": content},
    ]

    result = call_llm(messages)

    return json.loads(result["content"])


def _explain_compression_plan(summary: dict) -> str:
    """
    Generate a conservative compression strategy for scientific Earth system
    model data.

    Given a summary of a NetCDF dataset, this function proposes a compression
    strategy tailored for scientific data. It provides recommendations for each
    variable type, including chunking approach, compression settings, and any
    suggested precision changes.

    The function also explains the tradeoffs and assumptions behind the
    recommendations.

    Parameters
    ----------
    summary : dict
        A summary of the NetCDF dataset, typically including variable types,
        dimensions, and other relevant metadata.

    Returns
    -------
    str
        A detailed explanation of the recommended compression strategy, including
        chunking, compression, and precision guidance for each variable type,
        along with tradeoffs and assumptions.
    """
    content = f"""
        Given the following NetCDF dataset summary:

        {summary}

        Propose a conservative compression strategy suitable for scientific
        Earth system model data.

        For each variable type, recommend:
        - chunking approach (high-level)
        - compression (on/off, level)
        - precision changes (if any)

        Explain tradeoffs and assumptions.
    """

    messages = [
        system_message(),
        {
            "role": "user",
            "content": content,
        },
    ]

    try:
        result = call_llm(messages)
    except Exception as e:
        print(f"LLM API call failed: {e}")
        result = {
            "content": "LLM API call failed. Please check your connection or API service."
        }

    return result["content"]


def _build_encoding_from_plan(ds: xr.Dataset, plan: dict) -> dict:
    """Build a NetCDF encoding dictionary from a compression plan.

    This function generates an encoding dictionary suitable for use with xarray's
    `to_netcdf` method, based on a provided compression plan. The plan can specify
    variable classes and agent-approved compression choices for each variable.
    The function applies conservative, lossless compression by default, and only
    applies lossy compression if explicitly allowed in the plan. Coordinate and
    bounds variables are always protected from lossy or unsafe compression.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the variables to encode.
    plan : dict
        A dictionary specifying compression options. Keys may include:
            - "variable_classes": dict mapping variable names to classes
                (e.g., "state", "diagnostic", "index").
            - "compression_choices": dict mapping variable names to compression
                choices, each of which is a dict with a "mode" key and optional
                parameters (e.g., "scale_factor", "add_offset").

    Returns
    -------
    encoding : dict
        A dictionary mapping variable names to encoding options suitable for
        xarray's `to_netcdf` method.

    Notes
    -----
    - By default, applies zlib compression with complevel 4.
    - Lossy compression (e.g., quantization) is only applied if explicitly
        specified in the plan.
    - Coordinate and bounds variables are always encoded losslessly.
    - The function is designed to be deterministic and auditable.
    """
    encoding = {}

    variable_classes = plan.get("variable_classes", {})
    compression_choices = plan.get("compression_choices", {})  # optional

    for var in ds.data_vars:
        da = ds[var]

        # ------------------------------------------------------------------
        # Default: safe, lossless compression
        # ------------------------------------------------------------------
        enc = {
            "zlib": True,
            "complevel": 4,
        }

        # ------------------------------------------------------------------
        # Hard guard: never modify bounds / coordinate-like variables
        # ------------------------------------------------------------------
        if var.endswith("_bnds") or "bounds" in var.lower():
            encoding[var] = enc
            continue

        cls = variable_classes.get(var, "state")

        # ------------------------------------------------------------------
        # Optional agent-approved compression choice (future-proof)
        # ------------------------------------------------------------------
        choice = compression_choices.get(var)

        if choice:
            mode = choice.get("mode")

            if mode == "quantized_int16":
                # HARD SAFETY CHECKS
                if da.dtype.kind == "f" and cls == "state":
                    enc.update(
                        {
                            "dtype": "int16",
                            "zlib": True,
                            "complevel": 4,
                            "scale_factor": choice["scale_factor"],
                            "add_offset": choice["add_offset"],
                        }
                    )
            elif mode == "lossless":
                pass

        # ------------------------------------------------------------------
        # Conservative class-based rules (current behavior)
        # ------------------------------------------------------------------
        else:
            if cls == "diagnostic" and da.dtype == "float64":
                enc["dtype"] = "float32"

            if cls == "index":
                enc["dtype"] = "int16"

        encoding[var] = enc

    return encoding


def compare_compression_and_accuracy(
    original_path: str,
    compressed_path: str,
) -> Dict:
    """
    Compare file size reduction and numerical differences between
    an original and compressed NetCDF file.

    Returns per-variable error metrics and overall size reduction.
    """
    # -------------------------------------------------------------------------
    # File size comparison
    # -------------------------------------------------------------------------
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)

    size_report = {
        "original_mb": original_size / 1e6,
        "compressed_mb": compressed_size / 1e6,
        "reduction_percent": 100.0 * (1.0 - compressed_size / original_size),
    }

    # -------------------------------------------------------------------------
    # Numerical accuracy comparison
    # -------------------------------------------------------------------------
    ds_orig = xr.open_dataset(original_path)
    ds_comp = xr.open_dataset(compressed_path)

    accuracy_report = {}

    for var in ds_orig.data_vars:
        if var not in ds_comp:
            continue

        a = ds_orig[var]
        b = ds_comp[var]

        # Skip non-numeric variables
        if not np.issubdtype(a.dtype, np.number):
            continue

        diff = a - b

        accuracy_report[var] = {
            "rmse": float(np.sqrt((diff**2).mean())),
            "max_abs_diff": float(np.abs(diff).max()),
            "mean_abs_diff": float(np.abs(diff).mean()),
        }

    return {
        "size": size_report,
        "accuracy": accuracy_report,
    }


def print_compression_report(report: dict) -> None:
    size = report["size"]
    acc = report["accuracy"]

    print("\n=== SIZE REDUCTION ===")
    print(
        f"Original size:   {size['original_mb']:.2f} MB\n"
        f"Compressed size: {size['compressed_mb']:.2f} MB\n"
        f"Reduction:       {size['reduction_percent']:.1f}%"
    )

    print("\n=== ACCURACY (per variable) ===")

    if not acc:
        print("No numeric variables compared.")
        return

    for var, m in sorted(acc.items()):
        print(
            f"{var:20s} "
            f"RMSE={m['rmse']:.3e}  "
            f"MaxAbs={m['max_abs_diff']:.3e}  "
            f"MeanAbs={m['mean_abs_diff']:.3e}"
        )


def summarize_compression_report(report: Dict) -> Dict:
    """
    Use the LLM to interpret compression results and produce
    a concise, structured summary.
    """
    content = f"""
        You are evaluating the results of a NetCDF compression experiment.

        Compression results:
        {report}

        Interpret these results and return JSON ONLY with the following schema:

        {{
        "verdict": "safe" | "conditionally safe" | "unsafe",
        "key_findings": [string],
        "risks": [string],
        "recommended_next_steps": [string]
        }}

        Guidelines:
        - Numerical accuracy determines safety
        - Your verdict must not contradict numerical evidence
        - Use "conditionally safe" only when metrics are near thresholds

    """
    messages = [
        system_message(),
        {"role": "user", "content": content},
    ]

    result = call_llm(messages)

    return json.loads(result["content"])


def print_compression_report_summary(summary: dict) -> None:
    print("=== AGENTIC SUMMARY ===")
    print(f"Verdict: {summary['verdict']}\n")

    print("Key findings:")
    for f in summary["key_findings"]:
        print(f" - {f}")

    if summary["risks"]:
        print("\nRisks:")
        for r in summary["risks"]:
            print(f" - {r}")

    print("\nRecommended next steps:")
    for s in summary["recommended_next_steps"]:
        print(f" - {s}")


def _evaluate_and_accept_plan(
    original_path,
    compressed_path,
    dataset,
    variable,
    thresholds,
    candidate_desc,
    encoding,
    output_json,
    print_agent_opinion=False,
) -> bool:
    evaluation = _evaluate_plan(
        original_path, compressed_path, dataset, variable, thresholds
    )

    _print_plan_evaluation(evaluation)
    _print_before_after_comparison(original_path, compressed_path, variable)

    agent_opinion = _agent_assess_plan(
        candidate=candidate_desc,
        evaluation=evaluation,
    )

    if print_agent_opinion:
        print("\n=== AGENT ASSESSMENT ===")
        print(agent_opinion["agent_opinion"])
        print()

    if evaluation["verdict"] == "safe":
        print(
            f"Decision trace: metrics → SAFE → accept {candidate_desc['description']}"
        )
        _accept_plan(
            encoding=encoding,
            evaluation=evaluation,
            agent_opinion=agent_opinion,
            output_json=output_json,
        )

        return True

    return False


def _evaluate_plan(
    original_path: str,
    compressed_path: str,
    dataset: str,
    variable: str,
    thresholds: Dict[str, float],
) -> Dict:
    """
    Evaluate scientific impact of compression for a single variable.

    thresholds example:
    {
        "rmse_rel": 0.001,
        "max_abs_rel": 0.005,
        "mean_abs_rel": 0.0005,
    }
    """
    ds_orig = xr.open_dataset(original_path)
    ds_comp = xr.open_dataset(compressed_path)

    da0 = ds_orig[variable]
    da1 = ds_comp[variable]

    if da0.shape != da1.shape:
        raise ValueError(
            f"Shape mismatch for variable '{variable}': {da0.shape} vs {da1.shape}"
        )

    diff = da0 - da1

    # Compute value range for relative error normalization
    val_range = float(da0.max() - da0.min())
    if val_range == 0:
        val_range = 1.0  # Avoid division by zero; treat as constant field

    rmse = float(np.sqrt((diff**2).mean()))
    rmse_rel = rmse / val_range

    max_abs = float(np.abs(diff).max())
    max_abs_rel = max_abs / val_range

    mean_abs = float(np.abs(diff).mean())
    mean_abs_rel = mean_abs / val_range

    metrics = {
        "rmse_rel": rmse_rel,
        "max_abs_rel": max_abs_rel,
        "mean_abs_rel": mean_abs_rel,
    }

    # ------------------------------------------------------------
    # Verdict logic (deterministic, auditable)
    # ------------------------------------------------------------
    if (
        rmse_rel <= thresholds["rmse_rel"]
        and max_abs_rel <= thresholds["max_abs_rel"]
        and mean_abs_rel <= thresholds["mean_abs_rel"]
    ):
        verdict = "safe"
    elif rmse_rel <= 2 * thresholds["rmse_rel"]:
        verdict = "caution"
    else:
        verdict = "unsafe"

    return {
        "dataset": dataset,
        "variable": variable,
        "metrics": metrics,
        "thresholds": thresholds,
        "verdict": verdict,
    }


def _print_plan_evaluation(evaluation: dict) -> None:
    """
    Pretty-print the evaluation results to the console.
    """
    print("\n=== EVALUATION RESULTS ===")
    print(f"Variable: {evaluation['variable']}")
    print(f"Verdict:  {evaluation['verdict'].upper()}\n")

    print("Metrics:")
    for k, v in evaluation["metrics"].items():
        thr = evaluation["thresholds"].get(k)
        if thr is not None:
            status = "OK" if v <= thr else "EXCEEDS"
            print(f"  - {k:10s}: {v:.4e} (threshold={thr:.4e}) [{status}]")
        else:
            print(f"  - {k:10s}: {v:.4e}")

    print()


def _print_before_after_comparison(
    original_path: str,
    compressed_path: str,
    variable: str,
    max_samples: int = 1_000_000,
) -> None:
    """
    Print a lightweight before/after comparison of file size and
    sampled variable statistics.

    NOTE:
    This function intentionally uses sampled statistics for speed.
    Full-fidelity error metrics are computed elsewhere.
    """
    print("\n=== BEFORE / AFTER COMPARISON (QUICK) ===")

    # ------------------------------------------------------------------
    # File size
    # ------------------------------------------------------------------
    orig_size = os.path.getsize(original_path) / 1e6
    comp_size = os.path.getsize(compressed_path) / 1e6

    print("File size:")
    print(f"  Original:   {orig_size:.2f} MB")
    print(f"  Compressed: {comp_size:.2f} MB")
    print(f"  Reduction:  {100.0 * (1.0 - comp_size / orig_size):.1f}%")

    # ------------------------------------------------------------------
    # Variable-level comparison
    # ------------------------------------------------------------------
    ds0 = xr.open_dataset(original_path)
    ds1 = xr.open_dataset(compressed_path)

    if variable not in ds0 or variable not in ds1:
        print(f"\nVariable '{variable}' not found in both datasets.")
        return

    a = ds0[variable]
    b = ds1[variable]

    print(f"\nVariable: {variable}")
    print(f"  dtype: {a.dtype} → {b.dtype}")
    print(f"  shape: {a.shape} → {b.shape}")

    # ------------------------------------------------------------------
    # Sampled numeric summaries
    # ------------------------------------------------------------------
    if np.issubdtype(a.dtype, np.number):
        stats_a = _sampled_stats(a, max_samples=max_samples)
        stats_b = _sampled_stats(b, max_samples=max_samples)

        print("\nStatistics (sampled):")
        print(f"  min:  {stats_a['min']:.6e} → {stats_b['min']:.6e}")
        print(f"  max:  {stats_a['max']:.6e} → {stats_b['max']:.6e}")
        print(f"  mean: {stats_a['mean']:.6e} → {stats_b['mean']:.6e}")

    print()


def _sampled_stats(
    da: xr.DataArray,
    max_samples: int = 1_000_000,
) -> dict[str, float]:
    """
    Compute approximate min/max/mean using a bounded number of samples.

    This function is intended for fast, human-facing summaries only.
    It should NOT be used for scientific acceptance criteria.
    """
    # Flatten lazily
    da = da.stack(_flat=da.dims)

    n = da.sizes["_flat"]
    if n > max_samples:
        step = max(1, n // max_samples)
        da = da.isel(_flat=slice(0, n, step))

    return {
        "min": float(da.min().values),
        "max": float(da.max().values),
        "mean": float(da.mean().values),
    }


def _accept_plan(
    encoding: Dict,
    evaluation: Dict,
    output_json: str,
    agent_opinion: Dict | None = None,
):
    """
    Accept a compression strategy and persist the decision.

    This is the point where the agent's recommendation
    becomes an artifact.
    """
    artifact = {
        "accepted": True,
        "variable": evaluation["variable"],
        "numeric_verdict": evaluation["verdict"],
        "metrics": evaluation["metrics"],
        "thresholds": evaluation["thresholds"],
        "encoding": encoding,
    }

    if agent_opinion is not None:
        artifact["agent_opinion"] = agent_opinion

    if not os.path.exists(output_json):
        with open(output_json, "w") as f:
            json.dump([artifact], f, indent=2)
    else:
        with open(output_json, "r") as f:
            try:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []

        existing.append(artifact)

        with open(output_json, "w") as f:
            json.dump(existing, f, indent=2)

    return artifact


def _agent_assess_plan(candidate: Dict, evaluation: Dict) -> Dict:
    """
    Ask the agent to assess a compression candidate given evaluation metrics.

    Returns an advisory opinion (not authoritative).
    """
    messages = [
        system_message(),
        {
            "role": "user",
            "content": f"""
                Compression candidate:
                {candidate}

                Evaluation results:
                {evaluation}

                Assess whether this compression strategy is scientifically acceptable.
                Explain reasoning and potential risks.

                Do NOT override numerical thresholds.
                """,
        },
    ]

    response = call_llm(messages)

    return {
        "candidate_description": candidate.get("description", "<unknown>"),
        "agent_opinion": response["content"],
    }


def _fallback_result(dataset: str, variable: str) -> Dict:
    """
    Result used when no candidate passes evaluation.
    """
    return {
        "dataset": dataset,
        "variable": variable,
        "verdict": "safe",
        "metrics": {
            "rmse": 0.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
        },
        "note": "Fallback to lossless compression",
    }


def _lossless_encoding(ds: xr.Dataset) -> Dict:
    """
    Conservative lossless encoding for all variables.
    """
    encoding = {}

    for var in ds.data_vars:
        encoding[var] = {
            "zlib": True,
            "complevel": 4,
        }

    return encoding
