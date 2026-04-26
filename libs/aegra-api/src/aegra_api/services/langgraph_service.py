"""LangGraph integration service.

Architecture:
- Base graph definitions are cached (safe, immutable)
- Each request gets a fresh graph copy with checkpointer/store injected
- Factory graphs are rebuilt per-request with the appropriate ServerRuntime
- Thread-safe by design without locks
"""

import asyncio
import copy
import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid5

import structlog
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel
from langgraph_sdk.auth.types import BaseUser

from aegra_api.constants import ASSISTANT_NAMESPACE_UUID
from aegra_api.models.auth import User
from aegra_api.observability.base import (
    get_tracing_callbacks,
    get_tracing_metadata,
)
from aegra_api.services.graph_factory import (
    AccessContext,
    build_server_runtime,
    classify_factory,
    clear_factory_registry,
    coerce_context,
    generate_graph,
    invoke_factory,
    is_factory,
)
from aegra_api.services.js_bridge import JSBridgeError, get_js_bridge
from aegra_api.services.js_graph_wrapper import JSGraphWrapper

State = TypeVar("State")
logger = structlog.get_logger(__name__)


def _module_name_for(graph_id: str) -> str:
    """Return a safe ``sys.modules`` key for a dynamically loaded graph.

    Sanitises the *graph_id* so it cannot shadow real packages (e.g. ``os.path``,
    ``json``) by replacing dots, slashes, and hyphens with underscores and
    placing the result under a private ``aegra_graphs.`` namespace.
    """
    safe_id = graph_id.replace(".", "_").replace("/", "_").replace("-", "_")
    return f"aegra_graphs.{safe_id}"


class LangGraphService:
    """Service to work with LangGraph CLI configuration and graphs.

    Architecture:
    - Caches base graph definitions (raw StateGraph/Pregel before checkpointer)
    - Yields fresh copies per-request with checkpointer/store injected
    - Thread-safe without locks via immutable cached state
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else Path("aegra.json")
        self._explicit_config = config_path is not None
        self.config: dict[str, Any] | None = None
        self._graph_registry: dict[str, Any] = {}
        # Cache for base graph definitions (without checkpointer/store).
        # For factory graphs, this holds the default-compiled graph (for schema extraction).
        self._base_graph_cache: dict[str, Any] = {}
        # Factory callables — stored alongside the base cache so they can be
        # invoked per-request with the appropriate ServerRuntime/config.
        self._graph_factories: dict[str, Callable] = {}

    async def initialize(self) -> None:
        """Load configuration file and setup graph registry.

        Resolution order:
        1) Explicit config_path passed to constructor (if it exists)
        2) Shared resolution (AEGRA_CONFIG env var → aegra.json → langgraph.json)
        """
        from aegra_api.config import _resolve_config_path

        # 1) Explicit path wins if provided and exists
        if self._explicit_config and self.config_path.exists():
            resolved_path = self.config_path
        # 2) Otherwise use shared resolution (warn if explicit path was missing)
        else:
            if self._explicit_config:
                logger.warning(f"Explicit config path '{self.config_path}' not found, falling back to config discovery")
            resolved_path = _resolve_config_path()

        if not resolved_path or not resolved_path.exists():
            raise ValueError(
                "Configuration file not found. Expected one of: AEGRA_CONFIG path, ./aegra.json, or ./langgraph.json"
            )

        self.config_path = resolved_path

        with self.config_path.open() as f:
            self.config = json.load(f)

        # Setup dependency paths before loading graphs
        self._setup_dependencies()

        # Load graph registry from config
        self._load_graph_registry()

        # Eagerly load all graph modules so factory graphs are classified
        # before the first request arrives.  This populates _graph_factories
        # (and _base_graph_cache for static graphs) without calling factory
        # functions with empty user/config — fixing the bug where the first
        # request would fall through to the static path and get a graph
        # compiled with user=None.
        await self._load_all_graph_modules()

        # Pre-register assistants for each graph using deterministic UUIDs so
        # clients can pass graph_id directly.
        await self._ensure_default_assistants()

    def _parse_graph_entry(self, graph_id: str, graph_entry: str | dict[str, Any]) -> dict[str, str]:
        """Parse and validate a graph entry from config.

        Supported formats:
        - Legacy string: ``"./path/to/graph.py:graph"``
        - Object: ``{"runtime": "python" | "langgraphjs", "path": "./path:export"}``
        """
        runtime = "python"
        graph_path: str

        if isinstance(graph_entry, str):
            graph_path = graph_entry
        elif isinstance(graph_entry, dict):
            runtime_value = graph_entry.get("runtime", "python")
            if runtime_value not in {"python", "langgraphjs"}:
                raise ValueError(
                    f"Invalid runtime '{runtime_value}' for graph '{graph_id}'. "
                    "Supported runtimes: 'python', 'langgraphjs'"
                )
            runtime = runtime_value

            raw_path = graph_entry.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(
                    f"Invalid graph config for '{graph_id}'. Object entries must include a non-empty 'path' string"
                )
            graph_path = raw_path
        else:
            raise ValueError(
                f"Invalid graph config for '{graph_id}'. Expected a string or an object with runtime/path fields"
            )

        if ":" not in graph_path:
            raise ValueError(f"Invalid graph path format for '{graph_id}': {graph_path}")

        file_path, export_name = graph_path.rsplit(":", 1)
        if not file_path or not export_name:
            raise ValueError(f"Invalid graph path format for '{graph_id}': {graph_path}")

        return {
            "runtime": runtime,
            "file_path": file_path,
            "export_name": export_name,
        }

    def _load_graph_registry(self) -> None:
        """Load graph definitions from aegra.json"""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        graphs_config = self.config.get("graphs", {})

        for graph_id, graph_entry in graphs_config.items():
            self._graph_registry[graph_id] = self._parse_graph_entry(graph_id, graph_entry)
        # Detect graph_ids that collide after sanitisation (e.g. "a.b" and
        # "a_b" both map to aegra_graphs.a_b in sys.modules).
        # Only applies to Python graphs — JS graphs are not loaded as modules.
        seen_modules: dict[str, str] = {}
        for graph_id in graphs_config:
            entry = self._graph_registry[graph_id]
            if entry.get("runtime") == "langgraphjs":
                continue
            mod_name = _module_name_for(graph_id)
            if mod_name in seen_modules:
                raise ValueError(
                    f"Graph IDs '{seen_modules[mod_name]}' and '{graph_id}' "
                    f"collide after sanitisation (both map to {mod_name})"
                )
            seen_modules[mod_name] = graph_id

    async def _load_all_graph_modules(self) -> None:
        """Eagerly load all graph modules, classifying factories without calling them.

        For static graphs, the compiled graph is cached in ``_base_graph_cache``.
        For factory graphs, the callable is stored in ``_graph_factories`` for
        per-request invocation — the factory is NOT called here, so no graph is
        compiled with ``user=None``.
        """
        for graph_id, graph_info in self._graph_registry.items():
            raw_graph = await self._load_graph_from_file(graph_id, graph_info)
            if raw_graph is not None:
                # Static graph — compile and cache
                if isinstance(raw_graph, StateGraph):
                    logger.info("compiling_graph", graph_id=graph_id)
                    raw_graph = raw_graph.compile()
                self._base_graph_cache[graph_id] = raw_graph

    def _setup_dependencies(self) -> None:
        """Add dependency paths to sys.path for graph imports.

        Supports paths from the 'dependencies' config key.
        Paths are resolved relative to the config file location.
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")
        dependencies = self.config.get("dependencies", [])
        if not dependencies:
            return

        config_dir = self.config_path.parent

        # Iterate in reverse so first dependency in config has highest priority
        for dep in reversed(dependencies):
            dep_path = Path(dep)

            # Resolve relative paths from config directory
            if not dep_path.is_absolute():
                dep_path = (config_dir / dep_path).resolve()
            else:
                dep_path = dep_path.resolve()

            # Add to sys.path if exists and not already present
            path_str = str(dep_path)
            if dep_path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)
                logger.info(f"Added dependency path to sys.path: {path_str}")
            elif not dep_path.exists():
                logger.warning(f"Dependency path does not exist: {path_str}")

    async def _ensure_default_assistants(self) -> None:
        """Create a default assistant per graph with deterministic UUID.

        Uses uuid5 with a fixed namespace so that the same graph_id maps
        to the same assistant_id across restarts. Idempotent.
        """
        from sqlalchemy import select

        from aegra_api.core.orm import Assistant as AssistantORM
        from aegra_api.core.orm import AssistantVersion as AssistantVersionORM
        from aegra_api.core.orm import get_session

        # Fixed namespace used to derive assistant IDs from graph IDs
        NS = ASSISTANT_NAMESPACE_UUID
        session_gen = get_session()
        session = await anext(session_gen)
        try:
            for graph_id in self._graph_registry:
                assistant_id = str(uuid5(NS, graph_id))
                existing = await session.scalar(select(AssistantORM).where(AssistantORM.assistant_id == assistant_id))
                if existing:
                    continue
                session.add(
                    AssistantORM(
                        assistant_id=assistant_id,
                        name=graph_id,
                        description=f"Default assistant for graph '{graph_id}'",
                        graph_id=graph_id,
                        config={},
                        user_id="system",
                        metadata_dict={"created_by": "system"},
                    )
                )
                session.add(
                    AssistantVersionORM(
                        assistant_id=assistant_id,
                        version=1,
                        name=graph_id,
                        description=f"Default assistant for graph '{graph_id}'",
                        graph_id=graph_id,
                        metadata_dict={"created_by": "system"},
                    )
                )
            await session.commit()
        finally:
            await session.close()

    async def _get_base_graph(self, graph_id: str) -> Any:
        """Get the base compiled graph without checkpointer/store.

        Caches the compiled graph structure for reuse. This is safe because
        the base graph is immutable - we create copies with checkpointer/store
        injected per-request.

        For factory graphs whose modules were already loaded by
        ``_load_all_graph_modules``, this lazily calls the factory with
        default args to produce a base graph for schema extraction.

        @param graph_id: The graph identifier from aegra.json
        @returns: Compiled Pregel graph (without checkpointer/store)
        @raises ValueError: If graph_id not found or loading fails
        """
        if graph_id not in self._graph_registry:
            raise ValueError(f"Graph not found: {graph_id}")

        # Return cached base graph if available
        if graph_id in self._base_graph_cache:
            return self._base_graph_cache[graph_id]

        # Factory graph that was classified but never called with defaults yet.
        # Call now to produce a base graph for schema extraction.
        if graph_id in self._graph_factories:
            raw_graph = await self._call_factory_with_defaults(self._graph_factories[graph_id], graph_id)
        else:
            graph_info = self._graph_registry[graph_id]
            # Load graph from file — may return None if the module turns out
            # to be a factory (discovered during lazy loading).
            raw_graph = await self._load_graph_from_file(graph_id, graph_info)

        # _load_graph_from_file returns None for factory graphs that were
        # just discovered and stored in _graph_factories. Retry via the
        # factory path now that the callable is registered.
        if raw_graph is None and graph_id in self._graph_factories:
            raw_graph = await self._call_factory_with_defaults(self._graph_factories[graph_id], graph_id)

        if raw_graph is None:
            raise ValueError(f"Failed to load graph '{graph_id}': module did not produce a graph")

        # Compile if it's a StateGraph
        if isinstance(raw_graph, StateGraph):
            logger.info(f"🔧 Compiling graph '{graph_id}'")
            compiled_graph = raw_graph.compile()
        else:
            compiled_graph = raw_graph

        # Only cache static graphs — factory graphs must be re-invoked
        # each time so their result reflects the current user/config context.
        if graph_id not in self._graph_factories:
            self._base_graph_cache[graph_id] = compiled_graph
        return compiled_graph

    @asynccontextmanager
    async def get_graph(
        self,
        graph_id: str,
        *,
        config: dict[str, Any] | None = None,
        access_context: AccessContext = "threads.create_run",
        user: User | BaseUser | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[Any]:
        """Get a graph instance for execution with checkpointer/store injected.

        For factory graphs, the factory is invoked per-request with the
        appropriate ``ServerRuntime`` and config. For static graphs, the cached
        base graph is returned with checkpointer/store injected.

        This is a context manager that yields a fresh graph copy per-request.
        Thread-safe without locks since each request gets its own instance.

        Usage::

            async with langgraph_service.get_graph(
                "react_agent",
                config=run_config,
                access_context="threads.create_run",
                user=user,
                context=context,
            ) as graph:
                async for event in graph.astream(input, config):
                    ...

        Args:
            graph_id: The graph identifier from aegra.json.
            config: The ``RunnableConfig`` dict for this request.
            access_context: Why the graph is being accessed.
            user: The authenticated user (or ``None`` for anonymous access).
            context: The raw request context dict. For factories with
                ``ServerRuntime[T]``, this is coerced to ``T`` and passed
                to ``_ExecutionRuntime.context``.

        Yields:
            Compiled ``Pregel`` graph with Postgres checkpointer/store attached.

        Raises:
            ValueError: If *graph_id* not found or loading fails.
        """
        from aegra_api.core.database import db_manager

        checkpointer = db_manager.get_checkpointer()
        store = db_manager.get_store()

        # Resolve the factory callable. Normally populated by
        # _load_all_graph_modules() at startup, but may be empty after
        # invalidate_cache(). In that case we re-load the module to
        # re-classify the factory without invoking it with user=None.
        factory = self._graph_factories.get(graph_id)

        if not factory and graph_id not in self._base_graph_cache:
            # Neither factory nor cached static graph — re-load from file
            # (post-invalidation path). This re-classifies factories without
            # calling _call_factory_with_defaults(user=None).
            graph_info = self._graph_registry.get(graph_id)
            if graph_info:
                await self._load_graph_from_file(graph_id, graph_info)
            factory = self._graph_factories.get(graph_id)

        if factory:
            # Factory path — invoke per-request with ServerRuntime
            run_config = config or {"configurable": {}}
            coerced_context = coerce_context(context, graph_id)
            server_runtime = build_server_runtime(
                access_context=access_context,
                store=store,
                user=user,
                context=coerced_context,
            )

            result = invoke_factory(factory, graph_id, run_config, server_runtime)

            async with generate_graph(result, graph_id) as graph_obj:
                if isinstance(graph_obj, StateGraph):
                    graph_obj = graph_obj.compile()

                # Inject checkpointer/store into the factory-produced graph
                try:
                    graph_to_use = graph_obj.copy(update={"checkpointer": checkpointer, "store": store})
                except Exception as exc:
                    logger.warning(
                        "graph_checkpointer_injection_failed",
                        graph_id=graph_id,
                        exc=str(exc),
                        msg="Graph does not support checkpointer injection; running without persistence",
                    )
                    graph_to_use = graph_obj

                yield graph_to_use
        else:
            # Static path — use cached base graph (already loaded above)
            base_graph = await self._get_base_graph(graph_id)

            # Try to create a copy with checkpointer/store injected.
            # NOTE: Do this BEFORE yield to avoid dual-yield when exceptions
            # occur in the context body.
            # Deep-copy the graph's config to prevent astream from mutating
            # the cached base graph's nested dicts (e.g. config.metadata).
            # Pregel.copy() is shallow, so without this, all copies share
            # the same metadata dict, and writes during execution pollute
            # the base graph — breaking subsequent runs.
            try:
                graph_to_use = base_graph.copy(
                    update={
                        "checkpointer": checkpointer,
                        "store": store,
                        "config": copy.deepcopy(base_graph.config),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "graph_checkpointer_injection_failed",
                    graph_id=graph_id,
                    exc=str(exc),
                    msg="Graph does not support checkpointer injection; running without persistence",
                )
                graph_to_use = base_graph

            yield graph_to_use

    async def get_graph_for_validation(
        self,
        graph_id: str,
        *,
        config: dict[str, Any] | None = None,
        access_context: AccessContext = "assistants.read",
        user: User | BaseUser | None = None,
    ) -> Any:
        """Get a graph instance for validation/schema extraction only.

        For factory graphs, the factory is invoked with the given access context
        (default ``assistants.read``) to produce a fresh graph. For static graphs,
        returns the cached base graph.

        Does NOT include checkpointer/store — use ``get_graph()`` for execution.

        Args:
            graph_id: The graph identifier from aegra.json.
            config: Optional ``RunnableConfig`` dict (passed to factory).
            access_context: Why the graph is being accessed (default: schema read).
            user: The authenticated user (or ``None`` for anonymous access).

        Returns:
            Compiled ``Pregel`` graph (without checkpointer/store).

        Raises:
            ValueError: If *graph_id* not found or loading fails.
        """
        if graph_id in self._graph_factories:
            factory = self._graph_factories[graph_id]
            run_config = config or {"configurable": {}}
            server_runtime = build_server_runtime(
                access_context=access_context,
                store=None,
                user=user,
            )

            result = invoke_factory(factory, graph_id, run_config, server_runtime)
            async with generate_graph(result, graph_id) as graph_obj:
                if isinstance(graph_obj, StateGraph):
                    graph_obj = graph_obj.compile()
                # Capture the graph before exiting the context manager.
                # For async-CM-backed factories, __aexit__ fires after this
                # block. Since validation only reads schema metadata, the
                # compiled Pregel is safe to use after CM teardown.
                validated_graph = graph_obj
            return validated_graph

        return await self._get_base_graph(graph_id)

    async def _load_langgraphjs_graph(self, graph_id: str, graph_info: dict[str, str]) -> Any:
        """Load a LangGraphJS graph via the Aegra JS bridge subprocess.

        The bridge spawns a Node.js process that loads and executes
        LangGraph.js graphs, communicating via JSON-RPC over stdin/stdout.

        Falls back to the ``langgraph_api.js`` loader if the bridge is
        not available (e.g. when using the official LangGraph API server
        bridge package).
        """

        raw_path = graph_info["file_path"]
        export_name = graph_info["export_name"]
        file_path = Path(raw_path)

        if not file_path.is_absolute():
            file_path = (self.config_path.parent / file_path).resolve()

        if not file_path.exists():
            raise ValueError(f"LangGraphJS graph file not found: {file_path}")

        # --- Primary: Aegra JS bridge ---
        try:
            bridge = await get_js_bridge()
            graph_info_result = await bridge.load_graph(str(file_path), export_name, graph_id)
            logger.info(
                "Loaded LangGraphJS graph via bridge",
                graph_id=graph_id,
                file_path=str(file_path),
            )
            return JSGraphWrapper(
                bridge,
                graph_id,
                graph_info_result,
                file_path=str(file_path),
                export_name=export_name,
            )

        except JSBridgeError as bridge_exc:
            logger.warning(
                "JS bridge failed, trying fallback loaders",
                graph_id=graph_id,
                error=str(bridge_exc),
            )

        # --- Fallback: langgraph_api.js loader (external bridge package) ---
        loader_candidates: list[tuple[str, list[str]]] = [
            ("langgraph_api.js.loader", ["load_graph", "load_graph_from_path", "load_js_graph"]),
            ("langgraph_api.js", ["load_graph", "load_graph_from_path", "load_js_graph"]),
        ]

        import_errors: list[str] = []

        for module_name, function_names in loader_candidates:
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                import_errors.append(f"{module_name}: {exc}")
                continue

            for function_name in function_names:
                loader = getattr(module, function_name, None)
                if not callable(loader):
                    continue

                sig = inspect.signature(loader)

                call_patterns: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
                    ((str(file_path), export_name), {}),
                    ((str(file_path),), {}),
                    ((str(file_path),), {"export_name": export_name}),
                ]

                for args, kwargs in call_patterns:
                    try:
                        sig.bind(*args, **kwargs)
                    except TypeError:
                        continue

                    result = loader(*args, **kwargs)

                    if asyncio.iscoroutine(result):
                        result = await result

                    return result

        import_summary = "; ".join(import_errors) if import_errors else "No compatible JS loader entrypoint found"
        raise ValueError(
            "Failed to load LangGraphJS graph. Ensure Node.js 18+ is installed "
            "and libs/aegra-js-bridge/ dependencies are set up, or install a "
            f"LangGraph JS bridge package. Details: {import_summary}"
        )

    async def _load_graph_from_file(self, graph_id: str, graph_info: dict[str, str]) -> Any:
        """Load graph from filesystem.

        Paths are resolved relative to the config file's directory.

        For callable exports, the function inspects the signature to detect
        factory patterns. 0-arg factories are called once at load time.
        Config/runtime factories are registered in ``_graph_factories`` for
        per-request invocation and ``None`` is returned — no default-args
        call is made here, so factory graphs are never compiled with
        ``user=None`` at startup.

        Args:
            graph_id: The graph identifier from aegra.json.
            graph_info: Dict with ``file_path`` and ``export_name`` keys.

        Returns:
            A compiled ``Pregel`` or uncompiled ``StateGraph`` for static
            graphs, or ``None`` for factory graphs (the factory is stored
            in ``_graph_factories`` instead).

        Raises:
            ValueError: If the file or export is not found.
        """
        runtime = graph_info.get("runtime", "python")
        if runtime == "langgraphjs":
            return await self._load_langgraphjs_graph(graph_id, graph_info)

        raw_path = graph_info["file_path"]
        file_path = Path(raw_path)

        # Resolve relative paths from config file directory
        if not file_path.is_absolute():
            file_path = (self.config_path.parent / file_path).resolve()

        if not file_path.exists():
            raise ValueError(f"Graph file not found: {file_path}")

        # Dynamic import of graph module
        spec = importlib.util.spec_from_file_location(_module_name_for(graph_id), str(file_path.resolve()))
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load graph module: {file_path}")

        module = importlib.util.module_from_spec(spec)
        module_name = spec.name
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        # Get the exported graph
        export_name = graph_info["export_name"]
        if not hasattr(module, export_name):
            raise ValueError(f"Graph export not found: {export_name} in {file_path}")

        graph = getattr(module, export_name)

        if callable(graph):
            # Classify the factory signature (populates the dispatch hook registry)
            classify_factory(graph, graph_id)

            if not is_factory(graph_id):
                # 0-arg factory — call once at load time
                if asyncio.iscoroutinefunction(graph):
                    graph = await graph()
                else:
                    graph = graph()
                # Handle async context managers / coroutines returned by 0-arg factories
                async with generate_graph(graph, graph_id) as resolved:
                    graph = resolved
            else:
                # Config/runtime factory — store the callable for per-request
                # invocation.  Do NOT call with defaults here; the factory will
                # be invoked with the real user/config on each request.  If a
                # base graph is needed later for schema extraction, _get_base_graph
                # will call _call_factory_with_defaults lazily.
                self._graph_factories[graph_id] = graph
                return

        return graph

    async def _call_factory_with_defaults(self, fn: Callable, graph_id: str) -> Pregel | StateGraph:
        """Call a factory with minimal args to get a base graph for schema extraction.

        Intended **only** for introspection (schema extraction, graph
        structure discovery) — never for the hot execution path. Uses
        ``assistants.read`` access context and ``user=None``.

        .. note::

            ``build_server_runtime(user=None)`` may fall back to
            ``get_auth_ctx()`` if called within an HTTP request context,
            so the runtime may carry a real user. This is incidental,
            not guaranteed — callers must not rely on it.

        Args:
            fn: The factory callable.
            graph_id: The graph identifier.

        Returns:
            A compiled ``Pregel`` or uncompiled ``StateGraph``.
        """
        empty_config: dict[str, Any] = {"configurable": {}}
        runtime = build_server_runtime(
            access_context="assistants.read",
            store=None,
            user=None,
        )
        result = invoke_factory(fn, graph_id, empty_config, runtime)

        # Resolve async context managers, coroutines, etc.
        async with generate_graph(result, graph_id) as resolved:
            graph = resolved
        return graph

    def list_graphs(self) -> dict[str, str]:
        """List all available graphs."""
        return {graph_id: info["file_path"] for graph_id, info in self._graph_registry.items()}

    def invalidate_cache(self, graph_id: str | None = None) -> None:
        """Invalidate graph cache for hot-reload.

        Clears both the base graph cache and the factory callable registry,
        as well as the factory dispatch hooks in the ``graph_factory`` module.
        The module is also removed from ``sys.modules`` so the next import
        re-executes the file.

        After invalidation, ``_graph_factories`` is empty. The next call to
        ``_get_base_graph`` will fall through to ``_load_graph_from_file``,
        which re-discovers and re-classifies the factory, then retries via
        the factory path — so callers do not need to re-run
        ``_load_all_graph_modules`` after invalidation.

        Args:
            graph_id: Specific graph to invalidate, or ``None`` to clear all.
        """
        if graph_id:
            self._base_graph_cache.pop(graph_id, None)
            self._graph_factories.pop(graph_id, None)
            sys.modules.pop(_module_name_for(graph_id), None)
        else:
            self._base_graph_cache.clear()
            self._graph_factories.clear()
            for key in list(sys.modules.keys()):
                if key.startswith("aegra_graphs."):
                    sys.modules.pop(key, None)
        clear_factory_registry(graph_id)

    def get_config(self) -> dict[str, Any] | None:
        """Get loaded configuration"""
        return self.config

    def get_dependencies(self) -> list:
        """Get dependencies from config"""
        if self.config is None:
            return []
        return self.config.get("dependencies", [])

    def get_http_config(self) -> dict[str, Any] | None:
        """Get HTTP configuration from loaded config file.

        Returns:
            HTTP configuration dict or None if not configured
        """
        if self.config is None:
            return None
        return self.config.get("http")


# Global service instance
_langgraph_service = None


def get_langgraph_service() -> LangGraphService:
    """Get global LangGraph service instance"""
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphService()
    return _langgraph_service


def inject_user_context(user: Any | None, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inject user context into LangGraph configuration for user isolation.

    Passes ALL user fields (including custom auth handler fields like
    subscription_tier, team_id, etc.) to the graph config under
    'langgraph_auth_user'.

    Args:
        user: User object with identity and optional extra fields
        base_config: Base configuration to extend

    Returns:
        Configuration dict with user context injected
    """
    config: dict[str, Any] = (base_config or {}).copy()
    config["configurable"] = config.get("configurable", {})

    # All user-related data injection (only if user exists)
    if user:
        # Basic user identity for multi-tenant scoping
        config["configurable"].setdefault("user_id", user.identity)
        config["configurable"].setdefault("user_display_name", getattr(user, "display_name", None) or user.identity)

        # Full auth payload for graph nodes - includes ALL fields from auth handler
        if "langgraph_auth_user" not in config["configurable"]:
            try:
                # user.to_dict() returns all fields including extras from auth handlers
                config["configurable"]["langgraph_auth_user"] = user.to_dict()
            except Exception:
                # Fallback: minimal dict if to_dict unavailable or fails
                config["configurable"]["langgraph_auth_user"] = {"identity": user.identity}

    return config


def create_thread_config(
    thread_id: str, user: User | BaseUser | None, *, additional_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create LangGraph configuration for a specific thread with user context"""
    base_config = {"configurable": {"thread_id": thread_id}}

    if additional_config:
        base_config.update(additional_config)

    return inject_user_context(user, base_config)


def create_run_config(
    run_id: str,
    thread_id: str,
    user: User | BaseUser | None,
    *,
    additional_config: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create LangGraph configuration for a specific run with full context.

    The function is *additive*: it never removes or renames anything the client
    supplied.  We simply ensure a `configurable` dict exists and then merge a
    few server-side keys so graph nodes can rely on them.
    """
    from copy import deepcopy

    cfg: dict = deepcopy(additional_config) if additional_config else {}

    # Ensure a configurable section exists
    cfg.setdefault("configurable", {})

    # Merge server-provided fields (do NOT overwrite if client already set)
    cfg["configurable"].setdefault("thread_id", thread_id)
    cfg["configurable"].setdefault("run_id", run_id)

    # Ensure the root run ID is set to match so that astream_events recognizes it
    cfg.setdefault("run_id", run_id)

    # Add observability callbacks from various potential sources
    tracing_callbacks = get_tracing_callbacks()
    if tracing_callbacks:
        existing_callbacks = cfg.get("callbacks", [])
        if not isinstance(existing_callbacks, list):
            # If we want to be more robust, we can log a warning here
            existing_callbacks = []

        # Combine existing callbacks with new tracing callbacks to be non-destructive
        cfg["callbacks"] = existing_callbacks + tracing_callbacks

    # Add metadata from all observability providers (independent of callbacks)
    cfg.setdefault("metadata", {})
    user_identity = user.identity if user else None
    observability_metadata = get_tracing_metadata(run_id, thread_id, user_identity)
    cfg["metadata"].update(observability_metadata)

    # Apply checkpoint parameters if provided
    if checkpoint and isinstance(checkpoint, dict):
        cfg["configurable"].update({k: v for k, v in checkpoint.items() if v is not None})

    # Finally inject user context via existing helper
    return inject_user_context(user, cfg)
