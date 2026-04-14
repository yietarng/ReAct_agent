from langchain_anthropic import ChatAnthropic

from .tools import TOOLS

# Tool-bound LLM used by reason_node for the main ReAct loop
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    max_tokens=4096,
).bind_tools(TOOLS)

# Bare LLM (no tools) used by parse_papers_node for structured extraction
extractor_llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    max_tokens=4096,
)
