"""The uvicorn --loop factory must produce a psycopg-compatible loop (#513)."""

import asyncio

from aegra_api.utils.event_loop import selector_loop_factory


def test_factory_returns_a_selector_event_loop() -> None:
    loop = selector_loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_factory_returns_a_fresh_loop_each_call() -> None:
    first = selector_loop_factory()
    second = selector_loop_factory()
    try:
        assert first is not second
    finally:
        first.close()
        second.close()
