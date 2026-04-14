import asyncio
import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import graph
from agent.state import AgentState

load_dotenv()

INITIAL_MESSAGE = (
    "Find 5 position-independent KV cache papers from top-tier ML/NLP conferences "
    "(NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR, ECCV, NAACL, COLM) published between "
    "2022 and 2025. For each paper extract: problem, motivation, issues, approach, "
    "and key_findings. When done, output PAPERS_COMPLETE followed by the JSON list."
)


async def main():
    os.makedirs("outputs", exist_ok=True)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=INITIAL_MESSAGE)],
        "papers": [],
        "search_queries_used": [],
        "iteration": 0,
        "final_report_path": "",
    }

    print("Starting ReAct agent...\n", flush=True)

    final_state = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": 50},
    )

    path = final_state.get("final_report_path", "")
    papers = final_state.get("papers", [])

    if path:
        print(f"\nReport written to: {path}")
        print(f"Papers found: {len(papers)}")
    else:
        print("\nAgent finished without generating a report.")
        print(json.dumps(papers, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
