import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr
from xbitinfo import get_bitinformation
from xcdat.tutorial import open_dataset

from agentic.llm import call_llm, system_message


def inspect_dataset(ds: xr.Dataset) -> Dict:
    summary = {}

    for var in ds.data_vars:
        da = ds[var]

        entry = {
            "dims": da.dims,
            "shape": da.shape,
            "dtype": str(da.dtype),
            "size_mb": da.nbytes / 1e6,
            "min": float(da.min().values)
            if np.issubdtype(da.dtype, np.number)
            else None,
            "max": float(da.max().values)
            if np.issubdtype(da.dtype, np.number)
            else None,
            "missing": int(np.isnan(da.values).sum())
            if np.issubdtype(da.dtype, np.floating)
            else None,
            "coords": list(da.coords),
        }

        # Temporal and spatial resolution
        if "time" in da.dims:
            if da.sizes["time"] > 1:
                time_diffs = np.diff(da["time"].values)
                # Convert timedelta64 to days
                if np.issubdtype(time_diffs.dtype, np.timedelta64):
                    entry["temporal_resolution_days"] = float(
                        np.mean(time_diffs) / np.timedelta64(1, "D")
                    )
                else:
                    entry["temporal_resolution"] = str(np.mean(time_diffs))
            else:
                entry["temporal_resolution"] = None
        for dim in da.dims:
            if (
                dim in ds.coords
                and hasattr(ds[dim], "values")
                and ds[dim].sizes[dim] > 1
            ):
                diffs = np.diff(ds[dim].values)
                if np.issubdtype(diffs.dtype, np.timedelta64):
                    entry[f"{dim}_resolution_days"] = float(
                        np.mean(diffs) / np.timedelta64(1, "D")
                    )
                elif np.issubdtype(diffs.dtype, np.number):
                    entry[f"{dim}_resolution"] = float(np.mean(diffs))
                else:
                    entry[f"{dim}_resolution"] = str(np.mean(diffs))

        if (
            da.dtype.kind == "f"
            and "time" in da.dims
            and da.size > 1e5  # skip tiny arrays
        ):
            try:
                bitinfo_ds = analyze_bitinfo(ds, var)
                entry["bitinfo"] = summarize_bitinfo(bitinfo_ds)
            except Exception as e:
                entry["bitinfo"] = {"error": str(e)}
        else:
            entry["bitinfo"] = "skipped"

        summary[var] = entry

    # Add global coordinate structure
    summary["coordinates"] = {
        name: {
            "dims": ds[name].dims,
            "shape": ds[name].shape,
            "dtype": str(ds[name].dtype),
            "min": float(ds[name].min().values)
            if np.issubdtype(ds[name].dtype, np.number)
            else None,
            "max": float(ds[name].max().values)
            if np.issubdtype(ds[name].dtype, np.number)
            else None,
        }
        for name in ds.coords
    }

    return summary


def analyze_bitinfo(ds: xr.Dataset, var: str):
    """
    Compute bit-level information content for a variable.
    """
    info = get_bitinformation(
        ds[[var]],
        dim="time",  # or appropriate dimension
        implementation="python",
    )
    return info


def summarize_bitinfo(bitinfo_ds) -> Dict:
    """
    Summarize xbitinfo output from the python/plain implementation.

    This function:
    - extracts bit-level metrics (e.g., significant_bits, entropy, information_content) if present
    - works across xbitinfo versions
    - reports only evidence that actually exists
    """
    summary = {
        "implementation": "python",
        "metrics": {},
    }

    for name, da in bitinfo_ds.data_vars.items():
        try:
            metric_summary = {}
            # Extract all attributes from xbitinfo output
            if hasattr(da, "attrs") and da.attrs:
                for key, value in da.attrs.items():
                    metric_summary[key] = value
            # Also summarize values
            values = da.values
            if values.size == 1:
                metric_summary["value"] = float(values)
            else:
                metric_summary["mean"] = float(values.mean())
                metric_summary["max"] = float(values.max())
                metric_summary["min"] = float(values.min())
            summary["metrics"][name] = metric_summary
        except Exception as e:
            summary["metrics"][name] = f"unreadable ({e})"

    if not summary["metrics"]:
        summary["note"] = "No usable information-theoretic metrics returned"

    return summary


def propose_compression_plan(summary: Dict) -> Dict:
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


def explain_compression_strategy(summary: Dict) -> str:
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


def build_encoding_from_plan(ds, plan):
    """
    Build a NetCDF encoding dictionary from a compression plan.

    Design principles:
    - Conservative by default (lossless)
    - Deterministic and auditable
    - Lossy compression only applied if explicitly allowed by plan
    - Coordinate / bounds variables are always protected
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
                # else: silently fall back to lossless

            elif mode == "lossless":
                pass  # already default

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


def evaluate(
    original_path: str,
    compressed_path: str,
    variable: str,
    thresholds: Dict[str, float],
) -> Dict:
    """
    Evaluate scientific impact of compression for a single variable.

    thresholds example:
    {
        "rmse": 0.05,
        "max_abs": 0.2,
        "mean_abs": 0.02,
    }
    """
    ds_orig = xr.open_dataset(original_path)
    ds_comp = xr.open_dataset(compressed_path)

    da0 = ds_orig[variable]
    da1 = ds_comp[variable]

    if da0.shape != da1.shape:
        raise ValueError(
            f"Shape mismatch for variable '{variable}': {a.shape} vs {b.shape}"
        )

    diff = da0 - da1

    rmse = float(np.sqrt((diff**2).mean()))
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())

    metrics = {
        "rmse": rmse,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }

    # ------------------------------------------------------------
    # Verdict logic (deterministic, auditable)
    # ------------------------------------------------------------
    if (
        rmse <= thresholds["rmse"]
        and max_abs <= thresholds["max_abs"]
        and mean_abs <= thresholds["mean_abs"]
    ):
        verdict = "safe"
    elif rmse <= 2 * thresholds["rmse"]:
        verdict = "caution"
    else:
        verdict = "unsafe"

    return {
        "variable": variable,
        "metrics": metrics,
        "thresholds": thresholds,
        "verdict": verdict,
    }


def accept(
    encoding: Dict,
    evaluation: Dict,
    agent_opinion: Dict | None = None,
    output_dir: str = "outputs/compression",
):
    """
    Accept a compression strategy and persist the decision.

    This is the point where the agent's recommendation
    becomes an artifact.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

    with open(output_path / "accepted_compression.json", "w") as f:
        json.dump(artifact, f, indent=2)

    return artifact


def agent_assess(candidate: Dict, evaluation: Dict) -> Dict:
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


def fallback_result(variable: str) -> Dict:
    """
    Result used when no candidate passes evaluation.
    """
    return {
        "variable": variable,
        "verdict": "safe",
        "metrics": {
            "rmse": 0.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
        },
        "note": "Fallback to lossless compression",
    }


def lossless_encoding(ds: xr.Dataset) -> Dict:
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


def write_compressed(
    ds: xr.Dataset,
    encoding: Dict,
    output_path: str,
):
    """
    Write a compressed NetCDF file using a provided encoding dict.
    """
    ds.to_netcdf(
        output_path,
        encoding=encoding,
    )


def print_evaluation(evaluation: dict) -> None:
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


def print_before_after_comparison(
    original_path: str,
    compressed_path: str,
    variable: str,
) -> None:
    import os
    import xarray as xr
    import numpy as np

    print("\n=== BEFORE / AFTER COMPARISON ===")

    # ------------------------------------------------------------------
    # File size
    # ------------------------------------------------------------------
    orig_size = os.path.getsize(original_path) / 1e6
    comp_size = os.path.getsize(compressed_path) / 1e6

    print(f"File size:")
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

    # Numeric summaries
    if np.issubdtype(a.dtype, np.number):
        print("\nStatistics:")
        print(f"  min:  {float(a.min()):.6e} → {float(b.min()):.6e}")
        print(f"  max:  {float(a.max()):.6e} → {float(b.max()):.6e}")
        print(f"  mean: {float(a.mean()):.6e} → {float(b.mean()):.6e}")

    print()
