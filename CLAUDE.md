# CLAUDE.md — ReAct Agent: Position-Independent KV Paper Search & PDF Report Generator

## Project Overview

This project implements a **LangGraph-based ReAct agent** that:
1. Uses a web search tool to discover **5 position-independent KV cache papers** from top-tier ML/NLP conferences (NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR, ECCV — published 2022–2025)
2. Extracts structured metadata per paper (problem, motivation, issues, approach, key findings)
3. Generates a **10-page professional PDF summary report** via `reportlab`

---

## Repository Layout

```
react_kv_agent/
├── CLAUDE.md                  # ← this file
├── README.md
├── requirements.txt
├── .env                       # TAVILY_API_KEY, ANTHROPIC_API_KEY
├── agent/
│   ├── __init__.py
│   ├── graph.py               # LangGraph ReAct graph definition
│   ├── nodes.py               # Node functions (reason, act, observe)
│   ├── state.py               # AgentState TypedDict
│   └── tools.py               # Web search + paper-fetch tools
├── report/
│   ├── __init__.py
│   ├── builder.py             # PDF generation with reportlab
│   └── templates.py           # ReportLab styles and layout helpers
├── prompts/
│   └── system_prompt.txt      # ReAct system prompt for the agent
├── run.py                     # Entry-point CLI
└── outputs/
    └── kv_papers_report.pdf   # Final output
```

---

## Environment Setup

### 1. Python Version
Requires **Python ≥ 3.10**.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```
langgraph>=0.2.0
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-community>=0.3.0
tavily-python>=0.3.0
reportlab>=4.0.0
python-dotenv>=1.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
arxiv>=2.1.0
```

### 3. Environment Variables

Create `.env` at project root:
```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

Get Tavily API key at https://tavily.com (free tier: 1 000 searches/month).

---

## Architecture

### ReAct Loop (LangGraph)

```
 ┌──────────┐     ┌──────────┐     ┌──────────────┐
 │  REASON  │────▶│   ACT    │────▶│   OBSERVE    │
 │ (Claude) │     │ (Tools)  │     │ (Tool result)│
 └──────────┘     └──────────┘     └──────┬───────┘
      ▲                                    │
      └────────────────────────────────────┘
                (loop until 5 papers found)
```

**Stop condition**: `len(state["papers"]) >= 5 AND all papers have all 5 required fields`.

### State Schema (`agent/state.py`)

```python
from typing import TypedDict, Annotated, List
import operator

class Paper(TypedDict):
    title: str
    authors: str
    venue: str          # e.g. "NeurIPS 2024"
    year: int
    url: str
    problem: str        # ~150 words
    motivation: str     # ~150 words
    issues: str         # ~150 words
    approach: str       # ~200 words
    key_findings: str   # ~200 words

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]   # full message history
    papers: List[Paper]                        # accumulated papers
    search_queries_used: List[str]             # dedup guard
    iteration: int                             # safety counter
    final_report_path: str                     # set when PDF is written
```

---

## Tool Specifications (`agent/tools.py`)

### Tool 1 — `web_search`

```python
from langchain_community.tools.tavily_search import TavilySearchResults

web_search = TavilySearchResults(
    max_results=8,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
)
```

**Usage by agent**: Query like `"position-independent KV cache NeurIPS 2024 site:arxiv.org OR site:openreview.net"`.

---

### Tool 2 — `fetch_paper_details`

```python
@tool
def fetch_paper_details(arxiv_id_or_url: str) -> str:
    """
    Fetch abstract and metadata from an arXiv paper.
    Accepts either a full arXiv URL or a bare ID like '2405.12345'.
    Returns JSON with: title, authors, published, abstract, pdf_url.
    """
```

Implementation uses the `arxiv` Python library:
```python
import arxiv, json, re

def _fetch(arxiv_id_or_url: str) -> str:
    # normalise to bare ID
    match = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", arxiv_id_or_url)
    if not match:
        return json.dumps({"error": "Could not parse arXiv ID"})
    paper_id = match.group(1)
    client = arxiv.Client()
    results = list(client.results(arxiv.Search(id_list=[paper_id])))
    if not results:
        return json.dumps({"error": "Paper not found"})
    p = results[0]
    return json.dumps({
        "title": p.title,
        "authors": ", ".join(a.name for a in p.authors[:6]),
        "published": str(p.published.date()),
        "abstract": p.summary,
        "pdf_url": p.pdf_url,
    })
```

---

## Graph Definition (`agent/graph.py`)

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from .state import AgentState
from .tools import web_search, fetch_paper_details

TOOLS = [web_search, fetch_paper_details]
TOOL_NODE = ToolNode(TOOLS)

llm = ChatAnthropic(
    model="claude-opus-4-5",
    temperature=0,
    max_tokens=4096,
).bind_tools(TOOLS)

def reason_node(state: AgentState) -> dict:
    """LLM decides what to search or extract next."""
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1,
    }

def should_continue(state: AgentState) -> str:
    """Route: call tools, extract papers, or finish."""
    last = state["messages"][-1]
    # if LLM requested a tool call
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    # if enough structured papers collected
    papers = state.get("papers", [])
    if len(papers) >= 5:
        return "write_report"
    # safety valve
    if state["iteration"] >= 30:
        return "write_report"
    return "reason"

def parse_papers_node(state: AgentState) -> dict:
    """
    After each tool observation, ask Claude to extract any NEW paper
    entries from the latest tool result and append to state["papers"].
    Keep only papers that satisfy:
      - venue in {NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR, ECCV, NAACL, COLM}
      - year >= 2022
      - topic: position-independent / position-agnostic KV cache / attention
    """
    # (Implementation: call llm with structured extraction prompt,
    #  parse JSON response, merge with existing state["papers"])
    ...

def write_report_node(state: AgentState) -> dict:
    """Generate PDF and return its path."""
    from report.builder import build_pdf
    path = build_pdf(state["papers"])
    return {"final_report_path": path}

# Build graph
builder = StateGraph(AgentState)
builder.add_node("reason", reason_node)
builder.add_node("tools", TOOL_NODE)
builder.add_node("parse_papers", parse_papers_node)
builder.add_node("write_report", write_report_node)

builder.set_entry_point("reason")
builder.add_conditional_edges("reason", should_continue, {
    "tools": "tools",
    "write_report": "write_report",
    "reason": "reason",
})
builder.add_edge("tools", "parse_papers")
builder.add_edge("parse_papers", "reason")
builder.add_edge("write_report", END)

graph = builder.compile()
```

---

## System Prompt (`prompts/system_prompt.txt`)

```
You are a research assistant agent operating in a ReAct loop.

GOAL
----
Find exactly 5 peer-reviewed papers about **position-independent** (or position-agnostic)
**key-value (KV) cache** mechanisms in transformer models.

CONSTRAINTS
-----------
• Papers must appear in top-tier venues:
  NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, CVPR, ECCV, COLM
• Publication year: 2022, 2023, 2024, or 2025
• Must be directly about position-independent KV representations, not merely
  mentioning KV caches in passing
• No duplicates

SEARCH STRATEGY
---------------
1. Start with targeted queries like:
   - "position-independent key-value cache transformer NeurIPS 2024"
   - "position-agnostic KV cache ICLR 2023 arxiv"
   - "relative position KV compression attention ICML 2024"
2. Use fetch_paper_details to verify venue, year, and relevance from the abstract.
3. For each confirmed paper, extract ALL FIVE fields:
   problem, motivation, issues, approach, key_findings
   (each ≥ 100 words, based on abstract + your knowledge).
4. When you have 5 confirmed papers with all fields filled, output ONLY:

PAPERS_COMPLETE
[{"title":..., "authors":..., "venue":..., "year":..., "url":...,
  "problem":..., "motivation":..., "issues":..., "approach":...,
  "key_findings":...}, ...]

Do NOT hallucinate paper titles or venues.
```

---

## PDF Report Builder (`report/builder.py`)

### Target Structure (10 pages)

| Page(s) | Section |
|---------|---------|
| 1 | Cover Page — title, date, agent info |
| 2 | Table of Contents |
| 3 | Executive Summary (~400 words) |
| 4–9 | One paper per page (6 papers' worth of space; 5 papers + breathing room) |
| 10 | Conclusion & Comparative Analysis |

### Implementation Skeleton

```python
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import date

def build_pdf(papers: list, output_path: str = "outputs/kv_papers_report.pdf") -> str:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=LETTER,
        leftMargin=1*inch, rightMargin=1*inch,
        topMargin=1*inch,  bottomMargin=1*inch,
    )

    styles = _build_styles()
    story = []

    # ── Cover ──────────────────────────────────────────────────────
    story += _cover_page(styles)
    story.append(PageBreak())

    # ── Table of Contents ──────────────────────────────────────────
    story += _toc(papers, styles)
    story.append(PageBreak())

    # ── Executive Summary ──────────────────────────────────────────
    story += _executive_summary(papers, styles)
    story.append(PageBreak())

    # ── Per-Paper Sections ─────────────────────────────────────────
    for i, paper in enumerate(papers, 1):
        story += _paper_section(i, paper, styles)
        story.append(PageBreak())

    # ── Conclusion ─────────────────────────────────────────────────
    story += _conclusion(papers, styles)

    doc.build(story)
    return output_path


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "CoverTitle",
        fontSize=26, leading=32, alignment=TA_CENTER,
        textColor=colors.HexColor("#1a1a2e"), spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        "SectionHeader",
        fontSize=14, leading=18, textColor=colors.HexColor("#16213e"),
        spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "FieldLabel",
        fontSize=10, leading=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f3460"), spaceBefore=8, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        "Body",
        fontSize=10, leading=14, alignment=TA_JUSTIFY,
        fontName="Helvetica",
    ))
    return styles


def _paper_section(index: int, paper: dict, styles) -> list:
    """Renders one paper as a structured sub-report (~1 page)."""
    elems = []
    # Metadata header table
    header_data = [
        ["Paper", f"{index} of 5"],
        ["Title", paper["title"]],
        ["Authors", paper["authors"]],
        ["Venue / Year", f"{paper['venue']} {paper['year']}"],
        ["URL", paper["url"]],
    ]
    t = Table(header_data, colWidths=[1.2*inch, 5.3*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f4f8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#f0f8ff"), colors.white]),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 10))

    # Five structured fields
    fields = [
        ("Problem Description", "problem"),
        ("Motivation", "motivation"),
        ("Issues & Challenges", "issues"),
        ("Proposed Approach", "approach"),
        ("Key Experimental Findings", "key_findings"),
    ]
    for label, key in fields:
        elems.append(Paragraph(label, styles["FieldLabel"]))
        elems.append(Paragraph(paper.get(key, "N/A"), styles["Body"]))
        elems.append(HRFlowable(width="100%", thickness=0.3,
                                color=colors.lightgrey, spaceAfter=4))
    return elems
```

> **Note**: `_cover_page`, `_toc`, `_executive_summary`, and `_conclusion` follow the same pattern — compose `Paragraph`, `Spacer`, `Table`, and `HRFlowable` elements from `reportlab.platypus`.

---

## Entry Point (`run.py`)

```python
import asyncio, json, os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import graph
from agent.state import AgentState

load_dotenv()

INITIAL_MESSAGE = (
    "Find 5 position-independent KV cache papers from top-tier ML/NLP conferences "
    "(NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR, ECCV) published between 2022 and 2025. "
    "For each paper extract: problem, motivation, issues, approach, and key_findings. "
    "When done, output PAPERS_COMPLETE followed by the JSON list."
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

    print("🚀  Starting ReAct agent …\n")
    final_state = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": 50},
    )

    path = final_state.get("final_report_path", "")
    if path:
        print(f"\n✅  Report written to: {path}")
        print(f"    Papers found: {len(final_state['papers'])}")
    else:
        print("⚠️  Agent finished without generating a report.")
        print(json.dumps(final_state["papers"], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Running the Agent

```bash
# 1. Clone / create project structure
mkdir react_kv_agent && cd react_kv_agent

# 2. Create virtualenv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env with your keys

# 5. Run
python run.py
```

Expected output:
```
🚀  Starting ReAct agent …

[Iteration 1] Searching: "position-independent KV cache NeurIPS 2024"
[Iteration 2] Fetching paper: arxiv:2405.18747 …
...
[Iteration 14] 5 papers confirmed. Writing PDF …

✅  Report written to: outputs/kv_papers_report.pdf
    Papers found: 5
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph** over plain LangChain | Explicit state machine; stop condition and routing are transparent and testable |
| **Tavily** for web search | Returns structured JSON snippets; better than raw SerpAPI for academic queries |
| **`arxiv` Python library** for paper fetch | Authoritative metadata; avoids scraping |
| **`reportlab` Platypus** for PDF | Flow-based layout auto-handles page breaks; no headless browser required |
| **Structured extraction in `parse_papers_node`** | Separates tool calling from LLM reasoning; makes state transitions deterministic |
| **`recursion_limit=50`** | Prevents infinite loops; 30 iterations of logic + safety margin |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `TavilyAPIError: 401` | Check `TAVILY_API_KEY` in `.env` |
| `AnthropicAuthenticationError` | Check `ANTHROPIC_API_KEY` |
| Agent loops without finding 5 papers | Broaden queries in system prompt; lower venue constraints temporarily |
| PDF fonts render as boxes | Do **not** use Unicode sub/superscript characters in ReportLab — use `<sub>` and `<super>` XML tags in `Paragraph` objects |
| `recursion_limit` hit | Increase limit in `graph.ainvoke(config={"recursion_limit": 60})` |
| arXiv paper not found | Paper may use a non-standard ID format; pass the full URL instead |

---

## Extending the Project

- **Add more tools**: `semantic_scholar_search`, `openreview_fetch` for richer venue filtering
- **Async parallel search**: Use `asyncio.gather` to fire 3 searches simultaneously after the first round
- **Streaming progress**: Use `graph.astream_events` to print live iteration updates
- **Citation graph**: Add a `CrossRef` tool to pull paper citations and identify the most influential works
- **Evaluation harness**: Log all 5 extracted papers to a JSON file and run a Claude-as-judge scorer on factual accuracy

---

*Generated for use with Claude Code. Model: `claude-opus-4-5`. LangGraph ≥ 0.2. Python ≥ 3.10.*
