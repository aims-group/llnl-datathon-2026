from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


# -------------------
# 1. Define agent state
# -------------------
class AgentState(TypedDict):
    question: str
    answer: str


# -------------------
# 2. Hosted LLM
# -------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",  # or gpt-4.1, etc.
    temperature=0,
)


# -------------------
# 3. Node function
# -------------------
def answer_question(state: AgentState) -> AgentState:
    response = llm.invoke([HumanMessage(content=state["question"])])
    return {"question": state["question"], "answer": response.content}


# -------------------
# 4. Build graph
# -------------------
graph = StateGraph(AgentState)
graph.add_node("llm_answer", answer_question)
graph.set_entry_point("llm_answer")
graph.add_edge("llm_answer", END)

agent = graph.compile()

# -------------------
# 5. Run
# -------------------
result = agent.invoke({"question": "What does chunking do in NetCDF compression?"})

print(result["answer"])
