"""General-purpose object serialization for complex objects"""

import dataclasses
import inspect
from typing import Any

from aegra_api.core.serializers.base import SerializationError, Serializer


class GeneralSerializer(Serializer):
    """Simple object serializer for complex Python objects"""

    def serialize(self, obj: Any) -> Any:
        """Serialize any object to JSON-compatible format"""
        try:
            return self._serialize_object(obj)
        except Exception as e:
            raise SerializationError(f"Failed to serialize object: {str(e)}", obj.__class__.__name__, e) from e

    def _serialize_object(self, obj: Any) -> Any:
        """Core serialization logic for Python objects"""
        # Class objects (e.g. a Pydantic class passed to with_structured_output)
        # carry bound-method descriptors but cannot be dump()'d without an
        # instance. Render them by qualname so duck-typed checks below don't
        # invoke unbound methods.
        if inspect.isclass(obj):
            return f"{obj.__module__}.{obj.__qualname__}"

        # Handle Pydantic v2 models (model_dump method)
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            return obj.model_dump()

        # Handle LangChain objects and Pydantic v1 models (dict method)
        elif hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()

        # Handle LangGraph Interrupt objects (they don't have .dict() method)
        elif obj.__class__.__name__ == "Interrupt" and hasattr(obj, "value") and hasattr(obj, "id"):
            return {"value": self._serialize_object(obj.value), "id": obj.id}

        # Command (from tools like write_todos) is a dataclass with no model_dump/
        # .dict()/_asdict; it would otherwise hit str() and reach consumers as an
        # unparseable repr. Emit all fields to match orjson's native output on Platform.
        elif obj.__class__.__name__ == "Command" and dataclasses.is_dataclass(obj):
            return {field.name: self._serialize_object(getattr(obj, field.name)) for field in dataclasses.fields(obj)}

        # Handle NamedTuples (like PregelTask) - they have _asdict() method
        elif hasattr(obj, "_asdict") and callable(obj._asdict):
            return {k: self._serialize_object(v) for k, v in obj._asdict().items()}

        # Handle sets and frozensets
        elif isinstance(obj, (set, frozenset)):
            return list(obj)

        # Handle tuples and lists recursively
        elif isinstance(obj, (tuple, list)):
            return [self._serialize_object(item) for item in obj]

        # Handle dictionaries recursively
        elif isinstance(obj, dict):
            return {k: self._serialize_object(v) for k, v in obj.items()}

        # Handle basic JSON-serializable types
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # Fallback to string representation for unknown types
        else:
            return str(obj)
