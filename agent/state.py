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
