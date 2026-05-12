from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from chains import first_responder, revisor
from tool_executor import execute_tools

MAX_ITERATIONS = 2


def draft_node(state: MessagesState) -> MessagesState:
    """Draft the initial response"""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState) -> MessagesState:
    "Revise the answer based on tool results"
    response = revisor.invoke({"messages": state["messages"]})

    return {"messages": response}


def should_continue(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count"""
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if count_tool_visits >= MAX_ITERATIONS:
        return END
    return "execute_tools"


graph = StateGraph(MessagesState)
graph.add_node("draft", draft_node)
graph.add_node("revise", revise_node)
graph.add_node("execute_tools", execute_tools)

graph.add_edge(START, "draft")
graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revise")

graph.add_conditional_edges("revise", should_continue, ["execute_tools", END])

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="reflexion.png")


if __name__ == "__main__":
    res = app.invoke(
        {
            "messages": HumanMessage(
                content=(
                    "write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
                )
            )
        }
    )

    last_msg = res["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        print(last_msg.tool_calls[0]["args"]["answer"])