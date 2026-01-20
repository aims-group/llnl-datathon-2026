# %%
from pathlib import Path

import xarray as xr
import xbitinfo as xb

from agentic.workflows.compression import (
    inspect_dataset,
    print_before_after_comparison,
    print_evaluation,
    propose_compression_plan,
    build_encoding_from_plan,
    write_compressed,
    evaluate,
    agent_assess,
    accept,
    fallback_result,
    lossless_encoding,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ORIGINAL_PATH = "original.nc"
WORK_DIR = Path("outputs/work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET_VARIABLE = "pr"

THRESHOLDS = {
    "rmse": 0.05,
    "max_abs": 0.2,
    "mean_abs": 0.02,
}

# -----------------------------------------------------------------------------
# Main compression agent loop (with get_keepbits)
# -----------------------------------------------------------------------------


def main() -> None:
    print("=== Compression agent starting ===")

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = xr.open_dataset(ORIGINAL_PATH)

    # -------------------------------------------------------------------------
    # Inspect dataset (deterministic evidence)
    # -------------------------------------------------------------------------
    print("Inspecting dataset...")
    summary = inspect_dataset(ds)

    # -------------------------------------------------------------------------
    # Ask agent to propose a semantic compression plan
    # -------------------------------------------------------------------------
    print("Requesting compression plan from agent...")
    plan = propose_compression_plan(summary)
    print(plan)

    accepted = False

    # =====================================================================
    # Candidate 1: plan-based compression (baseline, safest)
    #
    # Plan-based candidates encode *semantic intent* about the data.
    # Variable roles (e.g., state, diagnostic, index) are inferred by the
    # agent and mapped to conservative, rule-based encodings. This approach
    # prioritizes scientific meaning and interpretability over maximal
    # compression and serves as the safe baseline against which more
    # aggressive strategies are evaluated.
    # =====================================================================

    print("\n--- Candidate 1: plan-based compression ---")

    encoding = build_encoding_from_plan(ds, plan)
    compressed_path = WORK_DIR / "compressed_plan.nc"

    write_compressed(ds, encoding, str(compressed_path))

    evaluation = evaluate(
        original_path=ORIGINAL_PATH,
        compressed_path=str(compressed_path),
        variable=TARGET_VARIABLE,
        thresholds=THRESHOLDS,
    )

    print_evaluation(evaluation)
    print_before_after_comparison(ORIGINAL_PATH, str(compressed_path), TARGET_VARIABLE)

    agent_opinion = agent_assess(
        candidate={"description": "plan-based compression"},
        evaluation=evaluation,
    )

    print("\n=== AGENT ASSESSMENT ===")
    print(agent_opinion["agent_opinion"])
    print()

    if evaluation["verdict"] == "safe":
        print("Decision trace: metrics → SAFE → accept plan-based encoding")
        accept(
            encoding=encoding,
            evaluation=evaluation,
            agent_opinion=agent_opinion,
        )
        accepted = True

    # =====================================================================
    # Candidate 2: xbitinfo bitround (only if needed)
    # xbitinfo-based candidates encode numerical information content.
    # xbitinfo provides information-theoretic analysis of floating-point datasets by quantifying how much real information is carried in individual mantissa bits. In our workflow, xbitinfo is used as an evidence generator that proposes precision-reduction candidates (e.g., via get_keepbits), which are then evaluated and conditionally accepted or rejected by an agent under explicit scientific accuracy constraints. This separates numerical information analysis from decision-making and preserves scientific guardrails.
    # Short:
    # xbitinfo determines how much numerical precision may be redundant; the agent determines whether reducing that precision is scientifically acceptable.
    # =====================================================================
    if not accepted:
        print("\n--- Candidate 2: xbitinfo bitround ---")

        # Compute bit information for target variable only
        bitinfo = xb.get_bitinformation(
            ds[[TARGET_VARIABLE]],
            dim="time",
            implementation="python",
        )

        keepbits = xb.get_keepbits(bitinfo, inflevel=0.99)

        print(f"xbitinfo keepbits for {TARGET_VARIABLE}: {keepbits}")

        # Apply bitrounding only to the target variable
        ds_bitrounded = ds.copy()
        ds_bitrounded[TARGET_VARIABLE] = xb.xr_bitround(
            ds_bitrounded[[TARGET_VARIABLE]],
            keepbits,
        )[TARGET_VARIABLE]

        compressed_path = WORK_DIR / "compressed_xbitinfo.nc"
        ds_bitrounded.to_netcdf(compressed_path)

        evaluation = evaluate(
            original_path=ORIGINAL_PATH,
            compressed_path=str(compressed_path),
            variable=TARGET_VARIABLE,
            thresholds=THRESHOLDS,
        )

        print_evaluation(evaluation)
        print_before_after_comparison(
            ORIGINAL_PATH, str(compressed_path), TARGET_VARIABLE
        )

        agent_opinion = agent_assess(
            candidate={
                "description": "xbitinfo bitround (99% information)",
                "keepbits": keepbits,
            },
            evaluation=evaluation,
        )

        print("\n=== AGENT ASSESSMENT ===")
        print(agent_opinion["agent_opinion"])
        print()

        if evaluation["verdict"] == "safe":
            print("Decision trace: metrics → SAFE → accept xbitinfo encoding")
            accept(
                encoding={
                    "method": "xbitinfo_bitround",
                    "keepbits": keepbits,
                },
                evaluation=evaluation,
                agent_opinion=agent_opinion,
            )
            accepted = True

    # =====================================================================
    # Fallback: lossless compression (safety net)
    #
    # The fallback path enforces a strict safety guarantee. If no candidate
    # compression strategy satisfies the predefined scientific accuracy
    # thresholds, the system reverts to a fully lossless encoding. This ensures
    # that scientific fidelity is preserved even when more aggressive or
    # information-theoretic candidates are rejected, and makes failure modes
    # explicit and auditable.
    # =====================================================================

    if not accepted:
        print(
            "\nNo candidate met safety thresholds. "
            "Falling back to lossless compression."
        )

        accept(
            encoding=lossless_encoding(ds),
            evaluation=fallback_result(TARGET_VARIABLE),
            agent_opinion=None,
        )

    print("=== Compression agent finished ===")


if __name__ == "__main__":
    main()

# %%
