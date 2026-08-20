"""General-purpose object serialization for complex objects"""

import dataclasses
import inspect
from base64 import b64encode
from collections import deque
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import UUID

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
            return self._serialize_object(obj.model_dump())

        # Handle LangChain objects and Pydantic v1 models (dict method)
        elif hasattr(obj, "dict") and callable(obj.dict):
            return self._serialize_object(obj.dict())

        # Handle LangGraph Interrupt objects (they don't have .dict() method)
        elif obj.__class__.__name__ == "Interrupt" and hasattr(obj, "value") and hasattr(obj, "id"):
            return {"value": self._serialize_object(obj.value), "id": obj.id}

        # Dataclasses (including LangGraph Command) have no standard JSON form.
        # Emit all fields recursively so nested values retain their structure.
        elif dataclasses.is_dataclass(obj):
            return {field.name: self._serialize_object(getattr(obj, field.name)) for field in dataclasses.fields(obj)}

        # Handle NamedTuples (like PregelTask) - they have _asdict() method
        elif hasattr(obj, "_asdict") and callable(obj._asdict):
            return {k: self._serialize_object(v) for k, v in obj._asdict().items()}

        # Handle common scalar types that JSON does not encode natively.
        elif isinstance(obj, bytes):
            return b64encode(obj).decode("ascii")
        elif isinstance(obj, Enum):
            return self._serialize_object(obj.value)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, (UUID, Decimal, PurePath)):
            return str(obj)
        elif isinstance(obj, Exception):
            return {"type": obj.__class__.__name__, "message": str(obj)}

        # Handle array-like containers recursively.
        elif isinstance(obj, (set, frozenset, deque, tuple, list)):
            return [self._serialize_object(item) for item in obj]

        # Handle dictionaries recursively
        elif isinstance(obj, dict):
            return {self._serialize_mapping_key(k): self._serialize_object(v) for k, v in obj.items()}

        # Handle basic JSON-serializable types
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # Fallback to string representation for unknown types
        else:
            return str(obj)

    def _serialize_mapping_key(self, key: Any) -> str | int | float | bool | None:
        serialized_key = self._serialize_object(key)
        if isinstance(serialized_key, (str, int, float, bool, type(None))):
            return serialized_key

        return str(serialized_key)
