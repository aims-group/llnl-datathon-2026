# llm.py
def propose_compression(summary):
    prompt = f"""
You are assisting with Earth system model data compression.

Dataset summary:
{summary}

Propose 2–3 reasonable compression strategies.
Focus on safety over aggressiveness.
Return JSON.
"""
