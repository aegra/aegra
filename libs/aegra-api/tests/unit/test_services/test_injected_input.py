"""The injected-input reader keys a claim by the node that made it, as every try of that node sees it."""

import pytest
from langchain_core.runnables import RunnableLambda

from aegra_api.services.injected_input import _current_task


def _key_in_node(checkpoint_ns: str) -> str | None:
    """What the reader records for a claim made under this LangGraph checkpoint namespace."""
    runnable: RunnableLambda[None, str | None] = RunnableLambda(lambda _: _current_task())
    return runnable.invoke(None, config={"configurable": {"checkpoint_ns": checkpoint_ns}})


class TestClaimKey:
    def test_outside_a_graph_node_there_is_no_key(self) -> None:
        assert _current_task() is None

    @pytest.mark.parametrize(
        ("first_try", "retry"),
        [
            pytest.param("call_model:t1", "call_model:t1", id="node"),
            pytest.param("agent:p|model:t1", "agent:p|1|model:t2", id="subgraph node"),
            pytest.param("agent:p|model:t1", "agent:p|2|model:t3", id="subgraph node, second retry"),
            pytest.param("outer:p|agent:q|model:t1", "outer:p|1|agent:r|model:t2", id="outer of nested subgraphs"),
            pytest.param("outer:p|agent:q|model:t1", "outer:p|agent:q|1|model:t2", id="inner of nested subgraphs"),
        ],
    )
    def test_a_retry_keys_its_claims_as_its_failed_try_did(self, first_try: str, retry: str) -> None:
        assert _key_in_node(retry) == _key_in_node(first_try)

    @pytest.mark.parametrize(
        ("one", "other"),
        [
            pytest.param("first:t1", "later:t2", id="later node"),
            pytest.param("worker:t1", "worker:t2", id="parallel Send tasks"),
            pytest.param("agent:p|first:t1", "agent:p|later:t2", id="later node of a subgraph"),
            pytest.param("agent:p|model:t1", "agent:q|model:t2", id="the subgraph run again as another task"),
        ],
    )
    def test_other_nodes_key_their_claims_apart(self, one: str, other: str) -> None:
        assert _key_in_node(one) != _key_in_node(other)
