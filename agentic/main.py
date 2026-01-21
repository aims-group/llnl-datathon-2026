from pathlib import Path

import xbitinfo as xb
from xcdat.tutorial import open_dataset

from agentic.workflows.compression import (
    accept,
    agent_assess,
    build_encoding_from_plan,
    evaluate,
    fallback_result,
    inspect_dataset,
    lossless_encoding,
    print_before_after_comparison,
    print_evaluation,
    propose_compression_plan,
    write_compressed,
)

# Use xcdat.tutorial.open_dataset for example datasets
# See: https://github.com/xCDAT/xcdat/blob/main/xcdat/tutorial.py
# Map each dataset key to its target variable for xcdat-data
DATASET_VARIABLES = [
    # ("pr_amon_access", "pr"),
    # ("tas_amon_access", "tas"),
    # ("tas_3hr_access", "tas"),
    # ("tas_amon_canesm5", "tas"),
    # ("so_omon_cesm2", "so"),
    # ("thetao_omon_cesm2", "thetao"),
    ("cl_amon_e3sm2", "cl"),
    ("ta_amon_e3sm2", "ta"),
]

INPUT_DIR = Path("data/input/")
WORK_DIR = Path("data/output/")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Relative error thresholds for compression safety.
# These values are conservative defaults for climate modeling data:
# - rmse_rel: Root Mean Squared Error divided by variable range (≤ 0.1%)
# - max_abs_rel: Max absolute error divided by variable range (≤ 0.5%)
# - mean_abs_rel: Mean absolute error divided by variable range (≤ 0.05%)
# Adjust as needed for stricter or more permissive scientific requirements.
SAFETY_THRESHOLDS = {
    "rmse_rel": 0.001,  # Relative RMSE (e.g., 0.1% of value range)
    "max_abs_rel": 0.005,  # Relative max absolute error (e.g., 0.5% of value range)
    "mean_abs_rel": 0.0005,  # Relative mean absolute error (e.g., 0.05% of value range)
}

# -----------------------------------------------------------------------------
# Main compression agent loop (with get_keepbits)
# -----------------------------------------------------------------------------


def main() -> None:
    print("=== Compression agent starting ===")

    summary_results = []

    for dataset_name, var in DATASET_VARIABLES:
        print(f"\n=== Dataset: {dataset_name} | Variable: {var} ===")

        ds = open_dataset(dataset_name)
        input_path = INPUT_DIR / f"{dataset_name}_{var}.nc"
        ds.to_netcdf(input_path)

        print("Inspecting dataset...")
        summary = inspect_dataset(ds)

        print("Requesting compression plan from agent...")
        plan = propose_compression_plan(summary)
        print(plan)

        accepted = False
        candidate_used = None
        compressed_path = None
        keepbits = None
        encoding_used = None

        # --- Candidate 1: plan-based compression ---
        print("\n--- Candidate 1: plan-based compression ---")
        encoding = build_encoding_from_plan(ds, plan)
        compressed_path = WORK_DIR / f"compressed_plan_{dataset_name}_{var}.nc"
        write_compressed(ds, encoding, str(compressed_path))

        accepted = _evaluate_and_accept_candidate(
            original_path=input_path,
            compressed_path=str(compressed_path),
            variable=var,
            thresholds=SAFETY_THRESHOLDS,
            candidate_desc={"description": "plan-based compression"},
            encoding=encoding,
        )
        if accepted:
            candidate_used = "plan-based compression"
            encoding_used = encoding

        # --- Candidate 2: xbitinfo bitround (only if needed) ---
        if not accepted:
            print("\n--- Candidate 2: xbitinfo bitround ---")
            bitinfo = xb.get_bitinformation(
                ds[[var]],
                dim="time",
                implementation="python",
            )
            keepbits = xb.get_keepbits(bitinfo, inflevel=0.99)
            print(f"xbitinfo keepbits for {var}: {keepbits}")
            ds_bitrounded = ds.copy()
            ds_bitrounded[var] = xb.xr_bitround(ds_bitrounded[[var]], keepbits)[var]
            compressed_path = WORK_DIR / f"compressed_xbitinfo_{dataset_name}_{var}.nc"
            ds_bitrounded.to_netcdf(compressed_path)

            encoding2 = {
                var: {
                    "dtype": ds[var].dtype,
                    "bit_rounding": int(keepbits),
                }
            }
            accepted = _evaluate_and_accept_candidate(
                original_path=input_path,
                compressed_path=compressed_path,
                variable=var,
                thresholds=SAFETY_THRESHOLDS,
                candidate_desc={
                    "description": f"xbitinfo bitround with keepbits={keepbits}"
                },
                encoding=encoding2,
            )

            if accepted:
                candidate_used = f"xbitinfo bitround (keepbits={keepbits})"
                encoding_used = encoding2

        # --- Fallback: lossless compression (safety net) ---
        if not accepted:
            print(
                "\nNo candidate met safety thresholds. "
                "Falling back to lossless compression."
            )
            encoding3 = lossless_encoding(ds)
            accept(
                encoding=encoding3,
                evaluation=fallback_result(var),
                agent_opinion=None,
            )
            candidate_used = "lossless compression"
            encoding_used = encoding3

        # Collect summary info for this variable
        orig_size_mb = input_path.stat().st_size / (1024 * 1024)
        comp_size_mb = (
            Path(compressed_path).stat().st_size / (1024 * 1024)
            if compressed_path and Path(compressed_path).exists()
            else None
        )
        summary_results.append(
            {
                "dataset": dataset_name,
                "variable": var,
                "plan": plan,
                "candidate": candidate_used,
                "encoding": encoding_used,
                "original_size_mb": orig_size_mb,
                "compressed_size_mb": comp_size_mb,
            }
        )

    print("=== Batch Compression Agent Finished ===")
    _print_batch_summary(summary_results)


def _print_batch_summary(summary_results):
    print("\n=== Batch Compression Summary ===\n")

    for res in summary_results:
        print(f"Dataset: {res['dataset']} | Variable: {res['variable']}")
        print(f"  Plan: {res['plan']}")
        print(f"  Compression method used: {res['candidate']}")
        print(f"  Original size:   {res['original_size_mb']:.2f} MB")

        if res["compressed_size_mb"] is not None:
            print(f"  Compressed size: {res['compressed_size_mb']:.2f} MB")
            reduction = (
                100.0
                * (res["original_size_mb"] - res["compressed_size_mb"])
                / res["original_size_mb"]
            )
            print(f"  Reduction:       {reduction:.1f}%")
        else:
            print("  Compressed size: N/A")
        print(f"  Encoding: {res['encoding']}")
        print()


def _evaluate_and_accept_candidate(
    original_path,
    compressed_path,
    variable,
    thresholds,
    candidate_desc,
    encoding,
    print_agent_opinion=False,
) -> bool:
    evaluation = evaluate(
        original_path=original_path,
        compressed_path=compressed_path,
        variable=variable,
        thresholds=thresholds,
    )

    print_evaluation(evaluation)
    print_before_after_comparison(original_path, compressed_path, variable)

    agent_opinion = agent_assess(
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
        accept(encoding=encoding, evaluation=evaluation, agent_opinion=agent_opinion)

        return True

    return False


if __name__ == "__main__":
    main()
