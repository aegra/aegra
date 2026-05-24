"""Agent Protocol Pydantic models"""

from aegra_api.models.assistants import (
    AgentSchemas,
    Assistant,
    AssistantCreate,
    AssistantList,
    AssistantSearchRequest,
    AssistantUpdate,
)
from aegra_api.models.auth import AuthContext, TokenPayload, User
from aegra_api.models.crons import (
    CronCountRequest,
    CronCreate,
    CronResponse,
    CronSearchRequest,
    CronUpdate,
)
from aegra_api.models.errors import AgentProtocolError, get_error_type
from aegra_api.models.runs import Run, RunCreate, RunStatus
from aegra_api.models.store import (
    StoreDeleteRequest,
    StoreGetResponse,
    StoreItem,
    StoreListNamespacesRequest,
    StoreListNamespacesResponse,
    StorePutRequest,
    StoreSearchRequest,
    StoreSearchResponse,
)
from aegra_api.models.threads import (
    Thread,
    ThreadCheckpoint,
    ThreadCheckpointPostRequest,
    ThreadCreate,
    ThreadHistoryRequest,
    ThreadList,
    ThreadSearchRequest,
    ThreadSearchResponse,
    ThreadState,
    ThreadStateUpdate,
    ThreadStateUpdateResponse,
    ThreadUpdate,
)

__all__ = [
    # Assistants
    "Assistant",
    "AssistantCreate",
    "AssistantList",
    "AssistantSearchRequest",
    "AssistantUpdate",
    "AgentSchemas",
    # Threads
    "Thread",
    "ThreadCreate",
    "ThreadList",
    "ThreadSearchRequest",
    "ThreadSearchResponse",
    "ThreadState",
    "ThreadStateUpdate",
    "ThreadStateUpdateResponse",
    "ThreadCheckpoint",
    "ThreadCheckpointPostRequest",
    "ThreadHistoryRequest",
    # Runs
    "Run",
    "RunCreate",
    "RunStatus",
    # Crons
    "CronCreate",
    "CronResponse",
    "CronUpdate",
    "CronSearchRequest",
    "CronCountRequest",
    # Store
    "StorePutRequest",
    "StoreGetResponse",
    "StoreSearchRequest",
    "StoreSearchResponse",
    "StoreItem",
    "StoreDeleteRequest",
    "StoreListNamespacesRequest",
    "StoreListNamespacesResponse",
    # Errors
    "AgentProtocolError",
    "get_error_type",
    # Auth
    "User",
    "AuthContext",
    "TokenPayload",
    "ThreadUpdate",
]
