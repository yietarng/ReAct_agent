import json
import re

import arxiv
from langchain_core.tools import tool

# ── Web search ──────────────────────────────────────────────────────────────

# Lazily initialized to avoid requiring TAVILY_API_KEY at import time
_tavily = None


def _get_tavily():
    """Return a cached TavilySearch instance, initializing on first call."""
    global _tavily
    if _tavily is None:
        try:
            from langchain_tavily import TavilySearch
            _tavily = TavilySearch(
                max_results=8,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
            )
        except ImportError:
            from langchain_community.tools.tavily_search import TavilySearchResults
            _tavily = TavilySearchResults(
                max_results=8,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
            )
    return _tavily


@tool
def web_search(query: str) -> str:
    """Search the web for academic papers about position-independent KV cache.

    Use queries like:
      'position-independent KV cache NeurIPS 2024 site:arxiv.org'
      'position-agnostic key-value attention ICLR 2023'

    Returns structured search results with URLs and snippets.
    """
    results = _get_tavily().invoke(query)
    return str(results)


# ── Fetch paper details ──────────────────────────────────────────────────────

@tool
def fetch_paper_details(arxiv_id_or_url: str) -> str:
    """Fetch abstract and metadata from an arXiv paper.

    Accepts either a full arXiv URL or a bare ID like '2405.12345'.
    Returns JSON with: title, authors, published, abstract, pdf_url.
    """
    match = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", arxiv_id_or_url)
    if not match:
        return json.dumps({"error": "Could not parse arXiv ID from input"})
    paper_id = match.group(1)
    client = arxiv.Client()
    results = list(client.results(arxiv.Search(id_list=[paper_id])))
    if not results:
        return json.dumps({"error": f"Paper not found for ID: {paper_id}"})
    p = results[0]
    return json.dumps({
        "title": p.title,
        "authors": ", ".join(a.name for a in p.authors[:6]),
        "published": str(p.published.date()),
        "abstract": p.summary,
        "pdf_url": p.pdf_url,
        "categories": p.categories,
    })


# ── Tool list (used by ToolNode and llm.bind_tools) ─────────────────────────

TOOLS = [web_search, fetch_paper_details]
