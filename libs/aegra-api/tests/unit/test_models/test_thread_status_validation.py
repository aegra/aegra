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


class TestThreadSearchRequestSelectValidation:
    """Tests for unsupported thread search response projection."""

    def test_thread_search_request_allows_omitted_select(self: "TestThreadSearchRequestSelectValidation") -> None:
        request = ThreadSearchRequest()
        assert request.select is None

    def test_thread_search_request_allows_null_select(self: "TestThreadSearchRequestSelectValidation") -> None:
        request = ThreadSearchRequest(select=None)
        assert request.select is None

    def test_thread_search_request_omits_select_from_auth_payload(
        self: "TestThreadSearchRequestSelectValidation",
    ) -> None:
        request = ThreadSearchRequest()
        assert "select" not in request.model_dump()

    @pytest.mark.parametrize("select", [[], ["thread_id"], ["interrupts"]])
    def test_thread_search_request_rejects_non_null_select(
        self: "TestThreadSearchRequestSelectValidation", select: list[str]
    ) -> None:
        with pytest.raises(ValueError, match="select projection is not supported"):
            ThreadSearchRequest(select=select)
