"""
This code is agentic because it delegates judgment and planning to a model, uses tools to gather evidence, and then makes a conditional decision that changes the system state based on that judgment—while enforcing hard safety boundaries in code.

It is not agentic because it is async, looping, or uses an “agent framework.”
It is agentic because of where cognition lives and how decisions are made.

Your code has an implicit, persistent goal:

“Compress this dataset as much as possible without violating scientific accuracy thresholds.”

How is this agentic?
“The system delegates semantic planning and risk assessment to a model, executes plans through tools, evaluates real-world consequences, and conditionally commits actions based on evidence—while enforcing hard safety boundaries in code.”

"""

# %%
from pathlib import Path

import xarray as xr

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

# %%

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
# Main compression agent loop
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
    # (variable classes only — no encodings)
    # -------------------------------------------------------------------------
    print("Requesting compression plan from agent...")
    plan = propose_compression_plan(summary)

    # -------------------------------------------------------------------------
    # Build encoding from plan (authoritative)
    # -------------------------------------------------------------------------
    print("Building encoding from plan...")
    encoding = build_encoding_from_plan(ds, plan)

    # -------------------------------------------------------------------------
    # Apply compression
    # -------------------------------------------------------------------------
    compressed_path = WORK_DIR / "compressed.nc"
    print(f"Writing compressed dataset to {compressed_path}...")
    write_compressed(ds, encoding, str(compressed_path))

    # -------------------------------------------------------------------------
    # Evaluate scientific impact (numeric truth)
    # -------------------------------------------------------------------------
    print("Evaluating compression accuracy...")
    evaluation = evaluate(
        original_path=ORIGINAL_PATH,
        compressed_path=str(compressed_path),
        variable=TARGET_VARIABLE,
        thresholds=THRESHOLDS,
    )

    print_evaluation(evaluation)

    print_before_after_comparison(
        original_path=ORIGINAL_PATH,
        compressed_path=str(compressed_path),
        variable=TARGET_VARIABLE,
    )

    # -------------------------------------------------------------------------
    # Ask agent to assess the result (advisory only)
    # -------------------------------------------------------------------------
    print("Requesting agent assessment...")
    agent_opinion = agent_assess(
        candidate={"description": "plan-based compression"},
        evaluation=evaluation,
    )

    print("\n=== AGENT ASSESSMENT ===")
    print(agent_opinion["agent_opinion"])
    print()

    # -------------------------------------------------------------------------
    # Accept or fallback
    # -------------------------------------------------------------------------
    if evaluation["verdict"] == "safe":
        print("Compression deemed SAFE. Accepting.")
        print(
            f"Decision trace: metrics → {evaluation['verdict']} → "
            f"{'accept encoding' if evaluation['verdict'] == 'safe' else 'fallback'}"
        )

        accept(
            encoding=encoding,
            evaluation=evaluation,
            agent_opinion=agent_opinion,
        )
    else:
        print(
            f"Compression verdict = {evaluation['verdict'].upper()}. "
            "Falling back to lossless compression."
        )
        fallback_encoding = lossless_encoding(ds)

        accept(
            encoding=fallback_encoding,
            evaluation=fallback_result(TARGET_VARIABLE),
            agent_opinion=agent_opinion,
        )

    print("=== Compression agent finished ===")


if __name__ == "__main__":
    main()

# %%
