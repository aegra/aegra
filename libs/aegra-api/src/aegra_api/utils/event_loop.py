"""Event loop factory for uvicorn's ``--loop`` option."""

import asyncio


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Selector event loop for uvicorn ``--loop``; works on every platform.

    Windows uvicorn defaults to the Proactor loop without --reload, and the
    LangGraph psycopg pool cannot connect on it (#513).
    """
    return asyncio.SelectorEventLoop()
