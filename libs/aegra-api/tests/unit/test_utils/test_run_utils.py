import pytest
from langgraph.types import Send

from aegra_api.utils.run_utils import map_command_to_langgraph


def test_merge_jsonb_and_should_skip_event():
    # Import inside test to ensure package import resolution in test env
    from aegra_api.utils.run_utils import _merge_jsonb, _should_skip_event

    # _merge_jsonb should merge dicts and ignore None
    a = {"x": 1, "y": {"a": 2}}
    b = {"y": {"b": 3}, "z": 4}
    merged = _merge_jsonb(a, None, b)
    assert merged["x"] == 1
    assert merged["z"] == 4
    # b should override a for top-level keys
    assert merged["y"] == {"b": 3}

    # _should_skip_event: tuple with last element being (something, metadata_dict)
    raw_event = ("values", {"foo": "bar"}, ("meta", {"tags": ["langsmith:nostream"]}))
    assert _should_skip_event(raw_event) is True

    # Other shapes should not be skipped
    assert _should_skip_event(("values", {"foo": "bar"})) is False
    assert _should_skip_event("just-a-string") is False


class DummyLogger:
    def __init__(self):
        self.calls = []

    async def adebug(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_filter_no_schema_returns_same():
    from aegra_api.utils import run_utils

    dummy = DummyLogger()
    # inject dummy logger to capture adebug
    run_utils.logger = dummy

    context = {"a": 1}
    result = await run_utils._filter_context_by_schema(context, None)
    # should return the context unchanged when no schema provided
    assert result is context


@pytest.mark.unit
@pytest.mark.asyncio
async def test_filter_with_schema_filters_keys():
    from aegra_api.utils import run_utils

    dummy = DummyLogger()
    run_utils.logger = dummy

    context = {"a": 1, "b": 2}
    schema = {"properties": {"a": {}}}

    filtered = await run_utils._filter_context_by_schema(context, schema)
    assert filtered == {"a": 1}

    # ensure adebug was called for the filtered-out key
    assert len(dummy.calls) == 1
    _, kw = dummy.calls[0]
    assert kw.get("context_key") == "b"
    assert "available_keys" in kw


@pytest.mark.unit
@pytest.mark.asyncio
async def test_filter_with_empty_properties_returns_context():
    from aegra_api.utils import run_utils

    dummy = DummyLogger()
    run_utils.logger = dummy

    context = {"a": 1}
    schema = {"properties": {}}

    result = await run_utils._filter_context_by_schema(context, schema)
    # no properties defined -> do not filter
    assert result == context


class TestMapCommandToLangGraph:
    def test_goto_send_without_input_maps_to_none_input(self) -> None:
        command = map_command_to_langgraph({"resume": 1, "goto": [{"node": "tool"}, "plain"]})

        assert command.resume == 1
        assert command.goto == [Send("tool", None), "plain"]

    def test_goto_send_with_input_keeps_input(self) -> None:
        command = map_command_to_langgraph({"goto": {"node": "tool", "input": {"k": 1}}})

        assert command.goto == [Send("tool", {"k": 1})]
