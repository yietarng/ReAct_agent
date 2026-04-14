import json
import pathlib
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import extractor_llm, llm
from .state import AgentState, Paper

# Load system prompt once at module import time
_SYSTEM_PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text()

# Allowed venues for filtering
ALLOWED_VENUES = {
    "neurips", "nips", "icml", "iclr", "acl", "emnlp",
    "naacl", "cvpr", "eccv", "colm",
}

EXTRACTION_PROMPT = """\
You are a paper extraction assistant. Given the tool result below, extract any papers
about position-independent or position-agnostic KV cache mechanisms in transformers.

REQUIREMENTS:
- Only include papers published at: NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, CVPR, ECCV, COLM
- Only include papers from 2022-2025
- Must be directly about position-independent/agnostic KV representations
- Infer the venue from the abstract or title if not explicitly stated
- Synthesize the five analysis fields from the abstract content (100-200 words each)

Return ONLY a valid JSON array. Each object must have ALL of these keys:
title, authors, venue, year, url, problem, motivation, issues, approach, key_findings

- venue: short conference name (e.g. "NeurIPS 2024", "ICLR 2023")
- year: integer (e.g. 2024)
- url: arxiv URL or paper URL
- problem: ~150 words describing the specific problem addressed
- motivation: ~150 words on why this problem matters
- issues: ~150 words on key technical challenges
- approach: ~200 words describing the proposed method
- key_findings: ~200 words on main experimental results

If no qualifying papers are found, return an empty array: []

Tool result:
{tool_result}
"""


# ── reason_node ───────────────────────────────────────────────────────────────

def reason_node(state: AgentState) -> dict:
    """LLM reasoning step — decides what to search or extract next."""
    messages = state["messages"]

    # Prepend system prompt on first call only
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm.invoke(messages)
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1,
    }


# ── parse_papers_node ─────────────────────────────────────────────────────────

def parse_papers_node(state: AgentState) -> dict:
    """After each tool call, extract any new papers from the tool result."""
    # Find the most recent ToolMessage
    last_tool_msg = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "tool":
            last_tool_msg = msg
            break

    if last_tool_msg is None:
        return {"papers": state.get("papers", [])}

    # Truncate to avoid token limits
    tool_content = str(last_tool_msg.content)[:8000]
    prompt = EXTRACTION_PROMPT.format(tool_result=tool_content)

    response = extractor_llm.invoke([HumanMessage(content=prompt)])
    text = response.content if hasattr(response, "content") else str(response)

    # Strip markdown code fences if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        new_papers: List[dict] = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return {"papers": state.get("papers", [])}

    if not isinstance(new_papers, list):
        return {"papers": state.get("papers", [])}

    # Merge with existing papers, deduplicating by normalized title
    existing = list(state.get("papers", []))
    existing_titles = {p.get("title", "").lower().strip() for p in existing}

    required_keys = {
        "title", "authors", "venue", "year", "url",
        "problem", "motivation", "issues", "approach", "key_findings",
    }

    for p in new_papers:
        if not isinstance(p, dict):
            continue
        # Check all required fields are present and non-empty
        if not all(p.get(k) for k in required_keys):
            continue
        title_norm = p.get("title", "").lower().strip()
        if title_norm in existing_titles:
            continue
        # Validate venue is allowed
        venue_lower = p.get("venue", "").lower()
        if not any(v in venue_lower for v in ALLOWED_VENUES):
            continue
        # Validate year
        try:
            year = int(p.get("year", 0))
        except (ValueError, TypeError):
            continue
        if year < 2022 or year > 2025:
            continue
        existing.append(p)
        existing_titles.add(title_norm)

    return {"papers": existing}


# ── write_report_node ─────────────────────────────────────────────────────────

def write_report_node(state: AgentState) -> dict:
    """Generate the PDF report from collected papers."""
    papers = list(state.get("papers", []))

    # Fallback: parse PAPERS_COMPLETE sentinel from messages if papers list is empty
    if not papers:
        for msg in reversed(state["messages"]):
            content = getattr(msg, "content", "") or ""
            if "PAPERS_COMPLETE" in content:
                try:
                    json_str = content.split("PAPERS_COMPLETE")[1].strip()
                    # Strip code fences if present
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0]
                    papers = json.loads(json_str.strip())
                    break
                except (json.JSONDecodeError, IndexError, ValueError):
                    pass

    from report.builder import build_pdf
    path = build_pdf(papers)
    return {"final_report_path": path}
