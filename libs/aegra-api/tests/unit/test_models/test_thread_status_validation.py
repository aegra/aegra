"""Tests for Thread model status validation."""

from datetime import UTC, datetime

import pytest

from aegra_api.models.threads import Thread, ThreadSearchRequest


class TestThreadStatusValidation:
    """Tests for Thread model status field validation."""

    def test_thread_validates_standard_statuses(self):
        """Test that Thread model accepts standard statuses."""
        valid_statuses = ["idle", "busy", "interrupted", "error"]
        for status in valid_statuses:
            thread = Thread(
                thread_id="test-thread-1",
                status=status,
                metadata={},
                user_id="test-user",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            assert thread.status == status

    def test_thread_rejects_invalid_status(self):
        """Test that Thread model rejects invalid status values."""
        with pytest.raises(ValueError, match="Invalid thread status"):
            Thread(
                thread_id="test-thread-1",
                status="invalid_status",
                metadata={},
                user_id="test-user",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )

    def test_thread_rejects_non_string_status(self):
        """Test that Thread model rejects non-string status values."""
        with pytest.raises(ValueError, match="Status must be a string"):
            Thread(
                thread_id="test-thread-1",
                status=123,  # type: ignore
                metadata={},
                user_id="test-user",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )

    def test_thread_from_orm_with_standard_status(self):
        """Test that Thread model accepts standard statuses from ORM."""

        # Simulate ORM object with standard status
        class MockORM:
            thread_id = "test-thread-1"
            status = "busy"  # Standard status
            metadata_json = {}
            user_id = "test-user"
            created_at = datetime.now(UTC)
            updated_at = datetime.now(UTC)

        thread = Thread.model_validate(MockORM())
        assert thread.status == "busy"


class TestThreadDocumentedStateFields:
    """Thread JSON must expose Agent Server / LangGraph SDK state fields."""

    def test_thread_defaults_empty_state_fields(self) -> None:
        thread = Thread(
            thread_id="test-thread-1",
            status="idle",
            metadata={},
            user_id="test-user",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        assert thread.values == {}
        assert thread.interrupts == {}
        assert thread.config == {}
        assert thread.state_updated_at is None

    def test_thread_accepts_populated_state_fields(self) -> None:
        updated = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
        thread = Thread(
            thread_id="test-thread-1",
            status="interrupted",
            metadata={},
            user_id="test-user",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            state_updated_at=updated,
            config={"configurable": {"thread_id": "test-thread-1"}},
            values={"messages": [{"type": "human", "content": "hi"}]},
            interrupts={"task-1": [{"id": "int-1", "value": "approve?"}]},
        )
        assert thread.values["messages"][0]["content"] == "hi"
        assert thread.interrupts["task-1"][0]["id"] == "int-1"
        assert thread.config["configurable"]["thread_id"] == "test-thread-1"
        assert thread.state_updated_at == updated


class TestThreadSearchRequestStatusValidation:
    """Tests for ThreadSearchRequest status filter validation."""

    def test_thread_search_request_validates_standard_statuses(self):
        """Test that ThreadSearchRequest accepts standard statuses."""
        valid_statuses = ["idle", "busy", "interrupted", "error"]
        for status in valid_statuses:
            request = ThreadSearchRequest(status=status)
            assert request.status == status

    def test_thread_search_request_allows_none_status(self):
        """Test that ThreadSearchRequest allows None status."""
        request = ThreadSearchRequest(status=None)
        assert request.status is None

    def test_thread_search_request_rejects_invalid_status(self):
        """Test that ThreadSearchRequest rejects invalid status values."""
        with pytest.raises(ValueError, match="Invalid thread status"):
            ThreadSearchRequest(status="invalid_status")
