"""Event loop factory for uvicorn's ``--loop`` option."""

import asyncio


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Selector event loop, importable via ``--loop aegra_api.utils.event_loop:selector_loop_factory``.

    Windows uvicorn defaults to the Proactor loop when run without --reload,
    and the LangGraph psycopg pool cannot connect on it (#513). The selector
    loop works on every platform.
    """
    return asyncio.SelectorEventLoop()
