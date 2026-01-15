from agentic.llm import call_llm, system_message

messages = [
    system_message(),
    {
        "role": "user",
        "content": "Given this dataset summary, propose a safe compression strategy.",
    },
]

result = call_llm(messages)
print(result["content"])
