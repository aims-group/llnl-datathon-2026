from datetime import datetime
from pathlib import Path

from xcdat.tutorial import open_dataset
import xarray as xr

from agentic.workflows.compression import (
    _accept_plan,
    _evaluate_and_accept_plan,
    _build_encoding_from_plan,
    _fallback_result,
    _inspect_dataset,
    _lossless_encoding,
    _propose_compression_plan,
)

# Use xcdat.tutorial.open_dataset for example datasets
# See: https://github.com/xCDAT/xcdat/blob/main/xcdat/tutorial.py
INPUT_DIR = Path("data/input/")
OUTPUT_DIR = Path("data/output/")
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ACCEPTANCE_JSON = OUTPUT_DIR / f"acceptance_summary_{TIMESTAMP}.json"

# Default dataset-variable map. Filepath indicates the full path to the NetCDF file.
# The user can pass use_subset to use xcdat tutorial datasets instead, which
# are smaller subsets suitable for testing.
DATASET_MAP = {
    "pr_amon_access": {
        "var_name": "pr",
        "filename": "pr_Amon_ACCESS-ESM1-5_historical_r10i1p1f1_gn_185001-201412.nc",
    },
    "tas_amon_access": {
        "var_name": "tas",
        "filename": "tas_Amon_ACCESS-ESM1-5_historical_r10i1p1f1_gn_185001-201412.nc",
    },
    "tas_3hr_access": {
        "var_name": "tas",
        "filename": "tas_3hr_ACCESS-ESM1-5_historical_r10i1p1f1_gn_201001010300-201501010000.nc",
    },
    "tas_amon_canesm5": {
        "var_name": "tas",
        "filename": "tas_Amon_CanESM5_historical_r13i1p1f1_gn_185001-201412.nc",
    },
    "so_omon_cesm2": {
        "var_name": "so",
        "filename": "so_Omon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc",
    },
    "thetao_omon_cesm2": {
        "var_name": "thetao",
        "filename": "thetao_Omon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc",
    },
    "cl_amon_e3sm2": {
        "var_name": "cl",
        "filename": "cl_Amon_E3SM-2-0_historical_r1i1p1f1_gr_185001-189912.nc",
    },
    "ta_amon_e3sm2": {
        "var_name": "ta",
        "filename": "ta_Amon_E3SM-2-0_historical_r1i1p1f1_gr_185001-189912.nc",
    },
}


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


def main(use_subset: bool = True) -> None:
    print("=== Compression agent starting ===")

    summary_results = []

    if use_subset:
        print("🧪 Using subset datasets from xcdat.tutorial for testing.\n")
    else:
        print("📂 Using full datasets from specified filepaths.\n")
        print("🔎 Checking that all dataset filepaths are valid...\n")
        _check_valid_filepaths(DATASET_MAP)

    for dataset_key, dataset_info in DATASET_MAP.items():
        var = dataset_info["var_name"]
        input_filename = dataset_info.get("filename")

        print(f"\n=== Dataset: {dataset_key} | Variable: {var} ===")

        # 1. Open the input dataset
        # -----------------------------
        if not use_subset:
            input_filepath = INPUT_DIR / input_filename
            ds = xr.open_dataset(input_filepath, decode_times=False)
        else:
            input_filename = f"{input_filename.replace('.nc', '_subset.nc')}"
            input_filepath = INPUT_DIR / input_filename

            try:
                ds = xr.open_dataset(input_filepath, decode_times=False)
            except FileNotFoundError:
                ds = open_dataset(dataset_key)
                ds.to_netcdf(input_filepath)

        # 2. Inspect the dataset and request a compression plan from the agent
        # ---------------------------------------------------------------------
        print("Inspecting dataset...")
        summary = _inspect_dataset(ds)

        print("Requesting compression plan from agent...")
        plan = _propose_compression_plan(summary)
        print(plan)

        # 3. Attempt compression according to the plan and evaluate
        # --------------------------------------------------------------
        accepted = False
        candidate_used = None

        # --- Candidate 1: plan-based compression ---
        print("\n--- Candidate 1: plan-based compression ---")
        print("Building dataset encoding from plan...")
        encoding = _build_encoding_from_plan(ds, plan)

        print("Applying compression to dataset and writing to disk...")
        compressed_path = OUTPUT_DIR / f"compressed_plan_{input_filename}"
        ds.to_netcdf(compressed_path, encoding=encoding)

        print("Evaluating compressed dataset against safety thresholds...")
        accepted = _evaluate_and_accept_plan(
            original_path=input_filepath,
            compressed_path=str(compressed_path),
            dataset=input_filename,
            variable=var,
            thresholds=SAFETY_THRESHOLDS,
            candidate_desc={"description": "plan-based compression"},
            output_json=OUTPUT_ACCEPTANCE_JSON,
            encoding=encoding,
        )

        if accepted:
            print(f"Plan-based compression accepted: {accepted}")
            candidate_used = "plan-based compression"
        # --- Fallback: lossless compression (safety net) ---
        if not accepted:
            print(
                "\nNo candidate met safety thresholds. Falling back to lossless "
                "compression."
            )

            encoding = _lossless_encoding(ds)
            _accept_plan(
                encoding=encoding,
                evaluation=_fallback_result(input_filename, var),
                output_json=OUTPUT_ACCEPTANCE_JSON,
                agent_opinion=None,
            )
            candidate_used = "lossless compression"

        # Collect summary info for this dataset
        orig_size_mb = input_filepath.stat().st_size / (1024 * 1024)
        comp_size_mb = (
            Path(compressed_path).stat().st_size / (1024 * 1024)
            if compressed_path and Path(compressed_path).exists()
            else None
        )
        summary_results.append(
            {
                "dataset": dataset_key,
                "variable": var,
                "plan": plan,
                "candidate": candidate_used,
                "encoding": encoding,
                "original_size_mb": orig_size_mb,
                "compressed_size_mb": comp_size_mb,
            }
        )

    print("=== Batch Compression Agent Finished ===")
    _print_batch_summary(summary_results)


def _check_valid_filepaths(dataset_map: dict[str, dict[str, str]]) -> None:
    for dataset_name, info in dataset_map.items():
        filename = info["filename"]
        filepath = INPUT_DIR / filename

        if not Path(filepath).exists():
            raise ValueError(
                f"Dataset {dataset_name} requires a valid 'filepath' when use_subset=False"
            )

    print("All dataset filepaths are valid.\n")


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


if __name__ == "__main__":
    main(use_subset=False)
