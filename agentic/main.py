from agentic.llm import call_llm, system_message


def main() -> None:
    messages = [
        system_message(),
        {
            "role": "user",
            "content": (
                "Given this dataset summary, propose a safe compression strategy. "
                "Focus on conservative choices appropriate for scientific data."
            ),
        },
    ]

    result = call_llm(messages)

    print("=== LLM RESPONSE ===")
    print(result["content"])
    print()
    print("=== METADATA ===")
    print(f"Backend: {result.get('backend')}")
    print(f"Model:   {result.get('model')}")
    if "usage" in result and result["usage"] is not None:
        print(f"Usage:   {result['usage']}")


if __name__ == "__main__":
    main()
