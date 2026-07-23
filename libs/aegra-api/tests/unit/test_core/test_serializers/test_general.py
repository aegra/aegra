"""Unit tests for serializers"""

from collections import namedtuple

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from aegra_api.core.serializers.base import SerializationError
from aegra_api.core.serializers.general import GeneralSerializer
from aegra_api.core.serializers.langgraph import LangGraphSerializer


class PydanticV2Model(BaseModel):
    """Pydantic v2 model with model_dump"""

    name: str
    value: int


class PydanticV1Style:
    """Mock Pydantic v1 style with dict method"""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def dict(self):
        return {"name": self.name, "value": self.value}


class InterruptMock:
    """Mock LangGraph Interrupt object"""

    def __init__(self, value, interrupt_id):
        self.value = value
        self.id = interrupt_id

    @property
    def __class__(self):
        class MockClass:
            __name__ = "Interrupt"

        return MockClass()


class TestGeneralSerializer:
    """Test GeneralSerializer class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.serializer = GeneralSerializer()

    def test_serialize_pydantic_v2_model(self):
        """Test serialization of Pydantic v2 model"""
        model = PydanticV2Model(name="test", value=42)
        result = self.serializer.serialize(model)

        assert result == {"name": "test", "value": 42}

    def test_serialize_pydantic_v1_style(self):
        """Test serialization of Pydantic v1 style object"""
        obj = PydanticV1Style(name="test", value=42)
        result = self.serializer.serialize(obj)

        assert result == {"name": "test", "value": 42}

    def test_serialize_interrupt_object(self):
        """Test serialization of LangGraph Interrupt object"""
        interrupt = InterruptMock(value={"data": "test"}, interrupt_id="int-123")
        result = self.serializer.serialize(interrupt)

        assert result == {"value": {"data": "test"}, "id": "int-123"}

    def test_serialize_namedtuple(self):
        """Test serialization of NamedTuple"""
        Task = namedtuple("Task", ["id", "name", "status"])
        task = Task(id=1, name="test_task", status="pending")

        result = self.serializer.serialize(task)

        assert result == {"id": 1, "name": "test_task", "status": "pending"}

    def test_serialize_set(self):
        """Test serialization of set"""
        data = {1, 2, 3}
        result = self.serializer.serialize(data)

        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_serialize_frozenset(self):
        """Test serialization of frozenset"""
        data = frozenset([1, 2, 3])
        result = self.serializer.serialize(data)

        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_serialize_tuple(self):
        """Test serialization of tuple"""
        data = (1, "two", 3.0)
        result = self.serializer.serialize(data)

        assert result == [1, "two", 3.0]

    def test_serialize_list(self):
        """Test serialization of list"""
        data = [1, "two", 3.0]
        result = self.serializer.serialize(data)

        assert result == [1, "two", 3.0]

    def test_serialize_nested_list(self):
        """Test serialization of nested list"""
        data = [1, [2, [3, 4]], 5]
        result = self.serializer.serialize(data)

        assert result == [1, [2, [3, 4]], 5]

    def test_serialize_dict(self):
        """Test serialization of dictionary"""
        data = {"key1": "value1", "key2": 42}
        result = self.serializer.serialize(data)

        assert result == {"key1": "value1", "key2": 42}

    def test_serialize_nested_dict(self):
        """Test serialization of nested dictionary"""
        data = {"outer": {"inner": {"deep": "value"}}}
        result = self.serializer.serialize(data)

        assert result == {"outer": {"inner": {"deep": "value"}}}

    def test_serialize_string(self):
        """Test serialization of string"""
        result = self.serializer.serialize("test string")
        assert result == "test string"

    def test_serialize_int(self):
        """Test serialization of integer"""
        result = self.serializer.serialize(42)
        assert result == 42

    def test_serialize_float(self):
        """Test serialization of float"""
        result = self.serializer.serialize(3.14)
        assert result == 3.14

    def test_serialize_bool_true(self):
        """Test serialization of boolean True"""
        result = self.serializer.serialize(True)
        assert result is True

    def test_serialize_bool_false(self):
        """Test serialization of boolean False"""
        result = self.serializer.serialize(False)
        assert result is False

    def test_serialize_none(self):
        """Test serialization of None"""
        result = self.serializer.serialize(None)
        assert result is None

    def test_serialize_complex_nested_structure(self):
        """Test serialization of complex nested structure"""
        Task = namedtuple("Task", ["id", "data"])
        data = {
            "tasks": [
                Task(id=1, data={"status": "pending"}),
                Task(id=2, data={"status": "completed"}),
            ],
            "metadata": {"count": 2, "tags": {1, 2, 3}},
            "nested": [{"deep": {"deeper": "value"}}],
        }

        result = self.serializer.serialize(data)

        assert result["tasks"][0] == {"id": 1, "data": {"status": "pending"}}
        assert result["tasks"][1] == {"id": 2, "data": {"status": "completed"}}
        assert result["metadata"]["count"] == 2
        assert set(result["metadata"]["tags"]) == {1, 2, 3}
        assert result["nested"][0]["deep"]["deeper"] == "value"

    def test_serialize_custom_object_fallback(self):
        """Test serialization of unknown object type (fallback to string)"""

        class CustomObject:
            def __repr__(self):
                return "CustomObject(test)"

        obj = CustomObject()
        result = self.serializer.serialize(obj)

        assert isinstance(result, str)
        assert "CustomObject" in result

    def test_serialize_command_structurally(self):
        """A LangGraph Command (returned by state-updating tools like
        write_todos) must serialize to its structural dict, NOT the str()
        fallback. Command is a dataclass with no model_dump/.dict()/_asdict, so
        without a dedicated branch it collapses to a repr string on the wire,
        and stream consumers (e.g. @ag-ui/langgraph's on_tool_end handler, which
        reads output.update.messages) can't recover the tool result — they read
        an undefined tool_call_id off a string and emit an invalid event.

        All four dataclass fields are emitted (graph/update/resume/goto), matching
        orjson's native dataclass output on LangGraph Platform so consumers read
        the Command byte-for-byte as they do there."""
        command = Command(
            update={
                "todos": [{"content": "Look up customer", "status": "in_progress"}],
                "messages": [
                    ToolMessage(
                        "Updated todo list",
                        tool_call_id="call_abc",
                        name="write_todos",
                        id="msg_1",
                    )
                ],
            }
        )
        result = self.serializer.serialize(command)

        assert isinstance(result, dict), "Command must not fall back to str()"
        assert set(result) == {"graph", "update", "resume", "goto"}, "all fields emitted (Platform parity)"
        assert result["graph"] is None
        assert result["resume"] is None
        assert result["goto"] == []  # unset goto defaults to () → []
        message = result["update"]["messages"][0]
        assert message["type"] == "tool"
        assert message["tool_call_id"] == "call_abc"
        assert message["name"] == "write_todos"
        assert result["update"]["todos"][0]["status"] == "in_progress"

    def test_serialize_command_preserves_falsy_resume(self):
        """A falsy resume value (False/0/"") is a legitimate resume payload and
        must survive serialization. Emitting all fields preserves it; a truthy
        filter (``if getattr(obj, name)``) would silently drop it."""
        assert self.serializer.serialize(Command(resume=False))["resume"] is False
        assert self.serializer.serialize(Command(resume=0))["resume"] == 0
        assert self.serializer.serialize(Command(resume=""))["resume"] == ""

    def test_serialize_pydantic_model_with_nested_data(self):
        """Test serialization of Pydantic model with nested structures"""

        class NestedModel(BaseModel):
            items: list[int]
            metadata: dict[str, str]

        model = NestedModel(items=[1, 2, 3], metadata={"key": "value"})
        result = self.serializer.serialize(model)

        assert result == {"items": [1, 2, 3], "metadata": {"key": "value"}}

    def test_serialize_mixed_types_in_list(self):
        """Test serialization of list with mixed types"""
        data = [1, "string", 3.14, True, None, {"key": "value"}, [1, 2]]
        result = self.serializer.serialize(data)

        assert result == [1, "string", 3.14, True, None, {"key": "value"}, [1, 2]]

    def test_serialize_empty_containers(self):
        """Test serialization of empty containers"""
        assert self.serializer.serialize([]) == []
        assert self.serializer.serialize({}) == {}
        assert self.serializer.serialize(set()) == []
        assert self.serializer.serialize(()) == []

    def test_serialize_unicode_string(self):
        """Test serialization of Unicode string"""
        result = self.serializer.serialize("Hello 世界 🌍")
        assert result == "Hello 世界 🌍"

    def test_serialize_large_number(self):
        """Test serialization of large numbers"""
        large_int = 999999999999999999
        large_float = 1.7976931348623157e308

        assert self.serializer.serialize(large_int) == large_int
        assert self.serializer.serialize(large_float) == large_float

    def test_serialize_negative_numbers(self):
        """Test serialization of negative numbers"""
        assert self.serializer.serialize(-42) == -42
        assert self.serializer.serialize(-3.14) == -3.14

    def test_serialize_pydantic_class_not_instance(self):
        """Regression for #356: passing a Pydantic class (not an instance) must
        not invoke unbound model_dump(). LangChain's with_structured_output
        stores the class reference inside the runnable chain, and LangGraph
        checkpointing then walks it through this serializer."""
        result = self.serializer.serialize(PydanticV2Model)
        assert isinstance(result, str)
        assert "PydanticV2Model" in result

    def test_serialize_pydantic_v1_class_not_instance(self):
        """Same defect on the .dict() branch — classes with a dict method
        (Pydantic v1 style) would also crash without the isclass guard."""
        result = self.serializer.serialize(PydanticV1Style)
        assert isinstance(result, str)
        assert "PydanticV1Style" in result

    def test_serialize_builtin_class_passthrough(self):
        """Builtin types (dict, list, str) hit the same path; render as str
        rather than blowing up on missing __module__/__qualname__."""
        result = self.serializer.serialize(dict)
        assert isinstance(result, str)
        assert "dict" in result


class MockTask:
    """Mock LangGraph task"""

    def __init__(self, task_id, name, error=None, result=None, interrupts=None):
        self.id = task_id
        self.name = name
        self.error = error
        self.result = result
        self.interrupts = interrupts or []


class MockSnapshot:
    """Mock LangGraph snapshot"""

    def __init__(self, tasks=None, interrupts=None):
        self.tasks = tasks or []
        self.interrupts = interrupts or []


class TestLangGraphSerializer:
    """Test LangGraphSerializer class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.serializer = LangGraphSerializer()

    def test_serialize_simple_dict(self):
        """Test basic serialization"""
        data = {"key": "value", "number": 42}
        result = self.serializer.serialize(data)

        assert result == {"key": "value", "number": 42}

    def test_serialize_nested_structure(self):
        """Test serialization of nested structures"""
        data = {"outer": {"inner": [1, 2, 3]}}
        result = self.serializer.serialize(data)

        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_serialize_task_with_id_and_name(self):
        """Test serialization of proper task object"""
        task = MockTask(task_id="task-123", name="test_task", error=None, result={"data": "value"})
        result = self.serializer.serialize_task(task)

        assert result["id"] == "task-123"
        assert result["name"] == "test_task"
        assert result["error"] is None
        assert result["result"] == {"data": "value"}
        assert result["interrupts"] == []
        assert result["checkpoint"] is None
        assert result["state"] is None

    def test_serialize_task_with_error(self):
        """Test serialization of task with error"""
        task = MockTask(task_id="task-456", name="failed_task", error="Something went wrong")
        result = self.serializer.serialize_task(task)

        assert result["id"] == "task-456"
        assert result["name"] == "failed_task"
        assert result["error"] == "Something went wrong"

    def test_serialize_task_with_interrupts(self):
        """Test serialization of task with interrupts"""
        task = MockTask(
            task_id="task-789",
            name="interrupted_task",
            interrupts=[{"type": "user_input", "value": "pause"}],
        )
        result = self.serializer.serialize_task(task)

        assert result["id"] == "task-789"
        assert result["interrupts"] == [{"type": "user_input", "value": "pause"}]

    def test_serialize_task_dict_format(self):
        """Test serialization of raw task dict"""
        task_dict = {"id": "raw-task", "name": "raw", "status": "pending"}
        result = self.serializer.serialize_task(task_dict)

        assert result == {"id": "raw-task", "name": "raw", "status": "pending"}

    def test_serialize_task_invalid_non_dict(self):
        """Test serialization error for non-dict task result"""
        with pytest.raises(SerializationError) as exc_info:
            self.serializer.serialize_task("invalid_task_string")

        assert "non-dict" in str(exc_info.value)

    def test_serialize_interrupt(self):
        """Test interrupt serialization"""
        interrupt = {"type": "user_input", "value": "test", "id": "int-123"}
        result = self.serializer.serialize_interrupt(interrupt)

        assert result == {"type": "user_input", "value": "test", "id": "int-123"}

    def test_extract_tasks_from_snapshot_with_tasks(self):
        """Test extracting tasks from snapshot"""
        tasks = [
            MockTask(task_id="task-1", name="task_one"),
            MockTask(task_id="task-2", name="task_two"),
        ]
        snapshot = MockSnapshot(tasks=tasks)

        result = self.serializer.extract_tasks_from_snapshot(snapshot)

        assert len(result) == 2
        assert result[0]["id"] == "task-1"
        assert result[1]["id"] == "task-2"

    def test_extract_tasks_from_snapshot_no_tasks(self):
        """Test extracting tasks from snapshot with no tasks"""
        snapshot = MockSnapshot(tasks=[])
        result = self.serializer.extract_tasks_from_snapshot(snapshot)

        assert result == []

    def test_extract_tasks_from_snapshot_no_tasks_attribute(self):
        """Test extracting tasks from snapshot without tasks attribute"""
        snapshot = type("Snapshot", (), {})()
        result = self.serializer.extract_tasks_from_snapshot(snapshot)

        assert result == []

    def test_extract_tasks_from_snapshot_with_invalid_task(self):
        """Test extracting tasks skips invalid tasks"""

        class InvalidTask:
            pass

        tasks = [
            MockTask(task_id="task-1", name="valid_task"),
            InvalidTask(),  # This should be skipped
        ]
        snapshot = MockSnapshot(tasks=tasks)

        result = self.serializer.extract_tasks_from_snapshot(snapshot)

        assert len(result) == 1
        assert result[0]["id"] == "task-1"

    def test_extract_interrupts_from_snapshot_with_interrupts(self):
        """Test extracting interrupts from snapshot"""
        interrupts = [
            {"type": "user_input", "id": "int-1"},
            {"type": "approval", "id": "int-2"},
        ]
        snapshot = MockSnapshot(interrupts=interrupts)

        result = self.serializer.extract_interrupts_from_snapshot(snapshot)

        assert len(result) == 2
        assert result[0]["type"] == "user_input"
        assert result[1]["type"] == "approval"

    def test_extract_interrupts_from_snapshot_no_interrupts(self):
        """Test extracting interrupts from snapshot with no interrupts"""
        snapshot = MockSnapshot(interrupts=[])
        result = self.serializer.extract_interrupts_from_snapshot(snapshot)

        assert result == []

    def test_extract_interrupts_from_snapshot_no_attribute(self):
        """Test extracting interrupts from snapshot without interrupts attribute"""
        snapshot = type("Snapshot", (), {})()
        result = self.serializer.extract_interrupts_from_snapshot(snapshot)

        assert result == []
