"""Tests for raw langgraph event → v2 channel event translation."""

from langchain_core.messages import AIMessageChunk, HumanMessage

from aegra_api.services.event_streaming.translator import EventTranslator


def _chunk(text: str, *, msg_id: str = "m1", last: bool = False) -> AIMessageChunk:
    chunk = AIMessageChunk(content=text, id=msg_id)
    if last:
        chunk.chunk_position = "last"
    return chunk


class TestMessageTranslation:
    def test_first_chunk_emits_start_then_delta(self) -> None:
        t = EventTranslator()
        events = t.translate("messages", (_chunk("hello"), {"ls_model_name": "gpt", "ls_provider": "openai"}))
        assert [e[1]["event"] for e in events] == ["message-start", "content-block-delta"]
        start = events[0][1]
        assert start == {
            "event": "message-start",
            "role": "ai",
            "id": "m1",
            "metadata": {"model": "gpt", "provider": "openai"},
        }
        delta = events[1][1]
        assert delta == {"event": "content-block-delta", "index": 0, "delta": {"type": "text-delta", "text": "hello"}}

    def test_subsequent_chunk_emits_only_delta(self) -> None:
        t = EventTranslator()
        t.translate("messages", (_chunk("hello"), {}))
        events = t.translate("messages", (_chunk(" world"), {}))
        assert [e[1]["event"] for e in events] == ["content-block-delta"]
        assert events[0][1]["delta"]["text"] == " world"

    def test_final_chunk_emits_finish(self) -> None:
        t = EventTranslator()
        t.translate("messages", (_chunk("hi"), {}))
        events = t.translate("messages", (_chunk("!", last=True), {}))
        assert [e[1]["event"] for e in events] == ["content-block-delta", "message-finish"]

    def test_empty_text_chunk_skips_delta(self) -> None:
        t = EventTranslator()
        t.translate("messages", (_chunk("first"), {}))
        events = t.translate("messages", (_chunk("", last=True), {}))
        assert [e[1]["event"] for e in events] == ["message-finish"]

    def test_complete_message_emits_start_delta_finish(self) -> None:
        """A whole (non-chunk) message from a non-streaming model finishes immediately."""
        from langchain_core.messages import AIMessage

        t = EventTranslator()
        events = t.translate("messages", (AIMessage(content="done", id="c1"), {}))
        assert [e[1]["event"] for e in events] == ["message-start", "content-block-delta", "message-finish"]

    def test_message_state_is_cleared_after_finish(self) -> None:
        """Per-message state does not accumulate across finished messages."""
        t = EventTranslator()
        t.translate("messages", (_chunk("a", msg_id="x", last=True), {}))
        assert "x" not in t._messages

    def test_human_message_role(self) -> None:
        t = EventTranslator()
        msg = HumanMessage(content="hey", id="h1")
        events = t.translate("messages", (msg, {}))
        assert events[0][1]["role"] == "human"

    def test_channel_is_messages(self) -> None:
        t = EventTranslator()
        events = t.translate("messages", (_chunk("x"), {}))
        assert all(channel == "messages" for channel, _ in events)

    def test_message_without_id_is_skipped(self) -> None:
        t = EventTranslator()
        assert t.translate("messages", (AIMessageChunk(content="x", id=None), {})) == []

    def test_malformed_message_payload_skipped(self) -> None:
        t = EventTranslator()
        assert t.translate("messages", "not a tuple") == []


class TestOtherChannels:
    def test_values_carries_state_dict_directly(self) -> None:
        t = EventTranslator()
        assert t.translate("values", {"count": 1}) == [("values", {"count": 1})]

    def test_updates_one_event_per_node(self) -> None:
        t = EventTranslator()
        events = t.translate("updates", {"node_a": {"x": 1}, "node_b": {"y": 2}})
        assert ("updates", {"node": "node_a", "values": {"x": 1}}) in events
        assert ("updates", {"node": "node_b", "values": {"y": 2}}) in events

    def test_custom(self) -> None:
        t = EventTranslator()
        assert t.translate("custom", {"foo": "bar"}) == [("custom", {"payload": {"foo": "bar"}})]

    def test_unhandled_modes_return_nothing(self) -> None:
        t = EventTranslator()
        for mode in ("metadata", "debug", "end", "error"):
            assert t.translate(mode, {"anything": True}) == []
