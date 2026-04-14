from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from .llm import llm  # noqa: F401 — needed to ensure tools are bound
from .nodes import parse_papers_node, reason_node, write_report_node
from .state import AgentState
from .tools import TOOLS

# ── Tool node ─────────────────────────────────────────────────────────────────

TOOL_NODE = ToolNode(TOOLS)


# ── Routing logic ─────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route after reason_node: tools, write_report, or loop back to reason."""
    last = state["messages"][-1]

    # If the LLM requested a tool call, execute it
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # If the agent output the PAPERS_COMPLETE sentinel, write the report
    content = getattr(last, "content", "") or ""
    if "PAPERS_COMPLETE" in content:
        return "write_report"

    # If we have enough confirmed papers, write the report
    papers = state.get("papers", [])
    if len(papers) >= 5:
        return "write_report"

    # Safety valve — prevent infinite loops
    if state.get("iteration", 0) >= 30:
        return "write_report"

    # Otherwise, keep reasoning
    return "reason"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("reason", reason_node)
    builder.add_node("tools", TOOL_NODE)
    builder.add_node("parse_papers", parse_papers_node)
    builder.add_node("write_report", write_report_node)

    builder.set_entry_point("reason")

    builder.add_conditional_edges(
        "reason",
        should_continue,
        {
            "tools": "tools",
            "write_report": "write_report",
            "reason": "reason",
        },
    )
    builder.add_edge("tools", "parse_papers")
    builder.add_edge("parse_papers", "reason")
    builder.add_edge("write_report", END)

    return builder.compile()


graph = build_graph()
