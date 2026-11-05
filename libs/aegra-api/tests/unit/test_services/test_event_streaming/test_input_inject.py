"""Tests for the v2 ``input.inject`` command (validation, admission, storage, auth result)."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aegra_api.models import User
from aegra_api.services.event_streaming import commands as cmd
from aegra_api.services.event_streaming import inject


@pytest.fixture
def user() -> User:
    return User(identity="u1")


@pytest.fixture
def admission(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value="accepting")
    monkeypatch.setattr(inject, "thread_admission", mock)
    return mock


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=True)
    monkeypatch.setattr(inject, "store_message", mock)
    return mock


def _inject(params: dict[str, Any]) -> dict[str, Any]:
    return {"id": 4, "method": "input.inject", "params": params}


async def _dispatch(
    payload: dict[str, Any], user: User, *, authorized: dict[str, Any] | None = None
) -> tuple[dict[str, Any], str | None]:
    session = AsyncMock()
    return await cmd.handle_command(payload, session=session, thread_id="t1", user=user, authorized=authorized)


def _stored_message(stored: AsyncMock) -> dict[str, Any]:
    return stored.call_args.kwargs["message"]


class TestInjectSuccess:
    async def test_returns_empty_result_and_starts_no_run(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        resp, run_id = await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "also check the tests", "id": "m1"}}),
            user,
        )

        assert resp == {"type": "success", "id": 4, "result": {}, "meta": {"applied_through_seq": 0}}
        assert run_id is None

    async def test_stores_a_langchain_message_dict_with_the_client_id(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "hi", "id": "m1", "name": "alice"}}),
            user,
        )

        assert stored.call_args.kwargs["thread_id"] == "t1"
        assert stored.call_args.kwargs["user_id"] == "u1"
        assert stored.call_args.kwargs["message_id"] == "m1"
        assert _stored_message(stored) == {
            "role": "user",
            "content": "hi",
            "id": "m1",
            "name": "alice",
            "additional_kwargs": {"injected_by": "u1"},
        }

    async def test_extensible_fields_land_in_additional_kwargs(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        await _dispatch(
            _inject(
                {
                    "namespace": [],
                    "message": {"role": "system", "content": "be brief", "id": "m2", "lang": "en", "source": "slack"},
                }
            ),
            user,
        )

        message = _stored_message(stored)
        assert message["role"] == "system"
        assert message["additional_kwargs"] == {"lang": "en", "source": "slack", "injected_by": "u1"}
        assert "lang" not in message

    async def test_server_stamps_the_identity_over_a_client_value(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "hi", "injected_by": "someone-else"}}),
            user,
        )

        assert _stored_message(stored)["additional_kwargs"]["injected_by"] == "u1"

    async def test_message_without_id_gets_a_server_id(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        await _dispatch(_inject({"namespace": [], "message": {"role": "user", "content": "hi"}}), user)

        message_id = stored.call_args.kwargs["message_id"]
        assert isinstance(message_id, str) and message_id
        assert _stored_message(stored)["id"] == message_id

    async def test_omitted_namespace_targets_the_root_graph(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        resp, _ = await _dispatch(_inject({"message": {"role": "user", "content": "hi"}}), user)

        assert resp["type"] == "success"

    async def test_identical_repeat_succeeds(self, admission: AsyncMock, stored: AsyncMock, user: User) -> None:
        stored.return_value = True

        resp, _ = await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "hi", "id": "m1"}}), user
        )

        assert resp["type"] == "success"


class TestInjectAuthResult:
    """The route's ``threads.create_run`` handler result is honored like on v1 run creation."""

    async def test_in_place_mutation_of_the_message_is_kept(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        authorized = {
            "id": 4,
            "method": "input.inject",
            "params": {"namespace": [], "message": {"role": "user", "content": "[redacted]", "id": "m1", "team": "a"}},
            "thread_id": "t1",
        }

        await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "secret", "id": "m1"}}),
            user,
            authorized=authorized,
        )

        message = _stored_message(stored)
        assert message["content"] == "[redacted]"
        assert message["additional_kwargs"] == {"team": "a", "injected_by": "u1"}

    async def test_returned_message_fields_merge_over_the_client_message(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "hi", "id": "m1"}}),
            user,
            authorized={"params": {"message": {"priority": "high"}}},
        )

        message = _stored_message(stored)
        assert message["content"] == "hi"
        assert message["additional_kwargs"] == {"priority": "high", "injected_by": "u1"}

    async def test_a_result_without_a_message_keeps_the_client_message(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "hi", "id": "m1"}}),
            user,
            authorized={"owner": "u1"},
        )

        assert _stored_message(stored)["content"] == "hi"

    async def test_a_handler_breaking_the_message_is_refused(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        resp, _ = await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "hi"}}),
            user,
            authorized={"params": {"message": {"content": 42}}},
        )

        assert resp["error"] == "invalid_argument"
        stored.assert_not_awaited()


class TestInjectAdmission:
    async def test_no_active_run_is_no_such_run(self, admission: AsyncMock, stored: AsyncMock, user: User) -> None:
        admission.return_value = "no_active_run"

        resp, _ = await _dispatch(_inject({"namespace": [], "message": {"role": "user", "content": "hi"}}), user)

        assert resp["type"] == "error"
        assert resp["error"] == "no_such_run"
        assert "run.start" in resp["message"]
        stored.assert_not_awaited()

    async def test_run_that_did_not_declare_live_input_is_not_supported(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        admission.return_value = "not_accepting"

        resp, _ = await _dispatch(_inject({"namespace": [], "message": {"role": "user", "content": "hi"}}), user)

        assert resp["error"] == "not_supported"
        stored.assert_not_awaited()

    async def test_admission_is_scoped_to_the_caller(self, admission: AsyncMock, stored: AsyncMock, user: User) -> None:
        await _dispatch(_inject({"namespace": [], "message": {"role": "user", "content": "hi"}}), user)

        assert admission.call_args.kwargs == {"thread_id": "t1", "user_id": "u1"}

    async def test_reused_id_with_a_different_message_is_invalid(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        stored.return_value = False

        resp, _ = await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "changed", "id": "m1"}}), user
        )

        assert resp["error"] == "invalid_argument"
        assert "m1" in resp["message"]


class TestInjectValidation:
    @pytest.mark.parametrize(
        "message",
        [
            None,
            "hi",
            {"content": "hi"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "hi"},
            {"role": "user"},
            {"role": "user", "content": ""},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "hi", "name": 7},
            {"role": "user", "content": "hi", "id": ""},
            {"role": "user", "content": "hi", "id": 12},
        ],
    )
    async def test_malformed_message_is_invalid(
        self, admission: AsyncMock, stored: AsyncMock, user: User, message: Any
    ) -> None:
        resp, _ = await _dispatch(_inject({"namespace": [], "message": message}), user)

        assert resp["type"] == "error"
        assert resp["error"] == "invalid_argument"
        admission.assert_not_awaited()
        stored.assert_not_awaited()

    @pytest.mark.parametrize("namespace", ["", "child", {"a": 1}, [1]])
    async def test_malformed_namespace_is_invalid(
        self, admission: AsyncMock, stored: AsyncMock, user: User, namespace: Any
    ) -> None:
        resp, _ = await _dispatch(_inject({"namespace": namespace, "message": {"role": "user", "content": "hi"}}), user)

        assert resp["error"] == "invalid_argument"
        stored.assert_not_awaited()

    async def test_subgraph_namespace_is_not_supported(
        self, admission: AsyncMock, stored: AsyncMock, user: User
    ) -> None:
        resp, _ = await _dispatch(
            _inject({"namespace": ["child:1"], "message": {"role": "user", "content": "hi"}}), user
        )

        assert resp["error"] == "not_supported"
        stored.assert_not_awaited()

    async def test_unknown_param_keys_are_logged(
        self, admission: AsyncMock, stored: AsyncMock, user: User, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        warning = MagicMock()
        monkeypatch.setattr(cmd.logger, "warning", warning)

        resp, _ = await _dispatch(
            _inject({"namespace": [], "message": {"role": "user", "content": "hi"}, "priority": 1}), user
        )

        assert resp["type"] == "success"
        warning.assert_called_once()
        assert warning.call_args.kwargs == {"method": "input.inject", "keys": ["priority"]}
