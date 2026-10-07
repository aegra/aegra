"""Unified factory example — typed context, user-aware tools, and MCP lifecycle.

Demonstrates all factory capabilities in a single graph:

- **Graph structure changes** via ``FactoryContext.enable_search`` — the factory
  conditionally includes/excludes the ``search_web`` tool and the ``tools`` node
  (read from ``ServerRuntime[FactoryContext]`` at factory time)
- **User-aware tool filtering** via ``runtime.user`` — admin users get
  ``delete_user`` (read from ``ServerRuntime`` at factory time)
- **MCP lifecycle** via ``FactoryContext.enable_mcp`` — the async context manager
  spins up MCP tool servers when enabled (factory time)
- **Model selection** via ``FactoryContext.model`` — read inside the node via
  ``Runtime[FactoryContext]`` at execution time
- **System prompt** via ``FactoryContext.system_prompt`` — read inside the node
  via ``Runtime[FactoryContext]`` at execution time

Usage in aegra.json::

    {
      "graphs": {
        "factory": "./examples/factory/graph.py:graph"
      }
    }
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, cast

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph_sdk.runtime import ServerRuntime
from typing_extensions import TypedDict

from factory.context import FactoryContext
from factory.tools import get_tools
from factory.utils import load_chat_model

# Optional MCP dependency — the example works without it
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False

# ---------------------------------------------------------------------------
# MCP server configuration (used when enable_mcp=True)
# ---------------------------------------------------------------------------

MCP_SERVERS: dict[str, dict[str, Any]] = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],  # nosec B108
        "transport": "stdio",
    },
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class State(TypedDict):
    """Minimal chat state."""

    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _build_graph(tools: list[BaseTool]) -> Any:
    """Build and compile the graph, adapting structure to the tool list.

    When *tools* is non-empty the graph includes a ``tools`` node and a
    conditional edge from ``call_model`` that routes tool-call responses
    to it. When *tools* is empty, the graph is a simple
    ``__start__ -> call_model -> __end__`` chain.

    Uses ``context_schema=FactoryContext`` so that nodes receive typed
    context via ``Runtime[FactoryContext]`` injection at execution time.
    """

    # The node closes over `tools` (a structural decision from the factory)
    # but reads model/prompt from Runtime[FactoryContext] (execution-time context).
    async def call_model(state: State, runtime: Runtime[FactoryContext]) -> dict[str, list[AIMessage]]:
        """Call the LLM using execution-time context for model and prompt."""
        ctx = runtime.context
        model = load_chat_model(ctx.model)

        if tools:
            model = model.bind_tools(tools)

        existing = list(state.get("messages", []))
        system_msg = ctx.system_prompt
        if existing and isinstance(existing[0], SystemMessage) and existing[0].content == system_msg:
            messages = existing
        else:
            messages = [SystemMessage(content=system_msg), *existing]

        response = cast("AIMessage", await model.ainvoke(messages))
        return {"messages": [response]}

    def route_output(state: State) -> Literal["__end__", "tools"]:
        """Route to tools or end."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    builder = StateGraph(State, context_schema=FactoryContext)
    builder.add_node("call_model", call_model)
    builder.add_edge("__start__", "call_model")

    if tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges("call_model", route_output)
        builder.add_edge("tools", "call_model")
    else:
        builder.add_edge("call_model", "__end__")

    return builder.compile(name="Factory Agent")


# ---------------------------------------------------------------------------
# Factory entry point
# ---------------------------------------------------------------------------


@asynccontextmanager
async def graph(config: dict[str, Any], runtime: ServerRuntime[FactoryContext]) -> AsyncIterator[Any]:
    """Unified factory — 2-param async context manager.

    Uses ``ServerRuntime[FactoryContext]`` for **structural** decisions only:

    - ``enable_search`` → include or exclude the tools node
    - ``enable_mcp`` → spin up MCP server connections
    - ``runtime.user`` → grant admin-only tools

    Execution-time values (``model``, ``system_prompt``) are read inside
    nodes via ``Runtime[FactoryContext]`` — LangGraph's standard context
    injection — not from factory closures.
    """
    ert = runtime.execution_runtime
    if ert:
        raw = ert.context
        if isinstance(raw, FactoryContext):
            ctx = raw
        elif isinstance(raw, dict):
            ctx = FactoryContext(**raw)
        else:
            ctx = FactoryContext()
    else:
        ctx = FactoryContext()

    # Assemble tools based on context + user permissions (structural decision)
    tools = list(get_tools(ctx, runtime.user))

    # Optionally add MCP tools (only in execution context)
    mcp_client = None
    if ctx.enable_mcp and ert:
        if not _HAS_MCP:
            raise RuntimeError(
                "enable_mcp=True but langchain-mcp-adapters is not installed. "
                "Install it with: pip install langchain-mcp-adapters"
            )
        client = MultiServerMCPClient(MCP_SERVERS)
        await client.__aenter__()
        mcp_client = client  # only set after successful entry
        mcp_tools: list[BaseTool] = mcp_client.get_tools()
        tools.extend(mcp_tools)

    compiled = _build_graph(tools)

    try:
        yield compiled
    finally:
        if mcp_client is not None:
            await mcp_client.__aexit__(None, None, None)
