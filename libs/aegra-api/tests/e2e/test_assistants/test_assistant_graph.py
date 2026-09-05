import pytest
from langgraph_sdk.errors import NotFoundError

from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_assistant_graph():
    """
    Test that we can retrieve the graph structure for an assistant.
    """
    client = get_e2e_client()

    # 1. Create an assistant
    assistant = await client.assistants.create(
        name="Test Graph Assistant",
        description="Assistant for testing graph endpoint",
        graph_id="agent",
        if_exists="do_nothing",
    )

    try:
        # 2. Get the graph structure without xray
        graph = await client.assistants.get_graph(assistant_id=assistant["assistant_id"])

        # 3. Verify graph structure
        assert "nodes" in graph, "Graph should have nodes"
        assert "edges" in graph, "Graph should have edges"
        assert isinstance(graph["nodes"], list), "Nodes should be a list"
        assert isinstance(graph["edges"], list), "Edges should be a list"
        assert len(graph["nodes"]) > 0, "Graph should have at least one node"

        elog(
            "Graph structure retrieved successfully",
            {
                "assistant_id": assistant["assistant_id"],
                "node_count": len(graph["nodes"]),
                "edge_count": len(graph["edges"]),
            },
        )

    finally:
        # Cleanup
        await client.assistants.delete(assistant_id=assistant["assistant_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_assistant_graph_with_xray_boolean():
    """
    Test that we can retrieve the graph structure with xray=True.
    """
    client = get_e2e_client()

    # 1. Create an assistant
    assistant = await client.assistants.create(
        name="Test Graph XRay Assistant",
        description="Assistant for testing graph endpoint with xray",
        graph_id="agent",
        if_exists="do_nothing",
    )

    try:
        # 2. Get the graph structure with xray=True
        graph = await client.assistants.get_graph(assistant_id=assistant["assistant_id"], xray=True)

        # 3. Verify graph structure
        assert "nodes" in graph, "Graph should have nodes"
        assert "edges" in graph, "Graph should have edges"
        assert isinstance(graph["nodes"], list), "Nodes should be a list"
        assert isinstance(graph["edges"], list), "Edges should be a list"

        elog(
            "Graph structure with xray retrieved successfully",
            {
                "assistant_id": assistant["assistant_id"],
                "node_count": len(graph["nodes"]),
                "edge_count": len(graph["edges"]),
            },
        )

    finally:
        # Cleanup
        await client.assistants.delete(assistant_id=assistant["assistant_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_assistant_graph_with_xray_integer():
    """
    Test that we can retrieve the graph structure with xray as an integer depth.
    """
    client = get_e2e_client()

    # 1. Create an assistant
    assistant = await client.assistants.create(
        name="Test Graph XRay Depth Assistant",
        description="Assistant for testing graph endpoint with xray depth",
        graph_id="agent",
        if_exists="do_nothing",
    )

    try:
        # 2. Get the graph structure with xray=1 (depth of 1)
        graph = await client.assistants.get_graph(assistant_id=assistant["assistant_id"], xray=1)

        # 3. Verify graph structure
        assert "nodes" in graph, "Graph should have nodes"
        assert "edges" in graph, "Graph should have edges"
        assert isinstance(graph["nodes"], list), "Nodes should be a list"
        assert isinstance(graph["edges"], list), "Edges should be a list"

        elog(
            "Graph structure with xray depth retrieved successfully",
            {
                "assistant_id": assistant["assistant_id"],
                "xray_depth": 1,
                "node_count": len(graph["nodes"]),
                "edge_count": len(graph["edges"]),
            },
        )

    finally:
        # Cleanup
        await client.assistants.delete(assistant_id=assistant["assistant_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_assistant_graph_not_found():
    """
    Test that getting a graph for a non-existent assistant returns 404.
    """
    client = get_e2e_client()

    # Try to get graph for non-existent assistant
    with pytest.raises(Exception) as exc_info:
        await client.assistants.get_graph(assistant_id="00000000-0000-0000-0000-000000000000")

    # Verify it's a 404 error
    assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()

    elog("Graph endpoint correctly returns 404 for non-existent assistant", {})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_assistant_subgraphs():
    """
    Test that we can retrieve subgraphs for an assistant.
    Uses subgraph_agent which has actual subgraphs.
    """
    client = get_e2e_client()

    # 1. Create an assistant with a graph that has subgraphs
    assistant = await client.assistants.create(
        name="Test Subgraphs Assistant",
        description="Assistant for testing subgraphs endpoint",
        graph_id="subgraph_agent",
        if_exists="do_nothing",
    )

    try:
        # 2. Get the subgraphs
        subgraphs = await client.assistants.get_subgraphs(assistant_id=assistant["assistant_id"])

        # 3. Verify subgraphs structure
        assert isinstance(subgraphs, dict), "Subgraphs should be a dictionary"
        assert len(subgraphs) > 0, "subgraph_agent should have subgraphs"

        elog(
            "Subgraphs retrieved successfully",
            {
                "assistant_id": assistant["assistant_id"],
                "subgraph_count": len(subgraphs),
                "subgraph_names": list(subgraphs.keys()),
            },
        )

    finally:
        # Cleanup
        await client.assistants.delete(assistant_id=assistant["assistant_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_assistant_subgraphs_with_recurse():
    """
    Test that we can retrieve subgraphs recursively.
    Uses subgraph_agent which has actual subgraphs.
    """
    client = get_e2e_client()

    # 1. Create an assistant with a graph that has subgraphs
    assistant = await client.assistants.create(
        name="Test Subgraphs Recurse Assistant",
        description="Assistant for testing subgraphs endpoint with recurse",
        graph_id="subgraph_agent",
        if_exists="do_nothing",
    )

    try:
        # 2. Get the subgraphs recursively
        subgraphs = await client.assistants.get_subgraphs(assistant_id=assistant["assistant_id"], recurse=True)

        # 3. Verify subgraphs structure
        assert isinstance(subgraphs, dict), "Subgraphs should be a dictionary"
        assert len(subgraphs) > 0, "subgraph_agent should have subgraphs"

        elog(
            "Subgraphs with recurse retrieved successfully",
            {
                "assistant_id": assistant["assistant_id"],
                "subgraph_count": len(subgraphs),
                "subgraph_names": list(subgraphs.keys()),
                "recurse": True,
            },
        )

    finally:
        # Cleanup
        await client.assistants.delete(assistant_id=assistant["assistant_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_assistant_subgraphs_not_found():
    """
    Test that getting subgraphs for a non-existent assistant returns 404.
    """
    client = get_e2e_client()

    # Try to get subgraphs for non-existent assistant
    with pytest.raises(Exception) as exc_info:
        await client.assistants.get_subgraphs(assistant_id="00000000-0000-0000-0000-000000000000")

    # Verify it's a 404 error
    assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()

    elog("Subgraphs endpoint correctly returns 404 for non-existent assistant", {})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_assistant_subgraph_by_namespace() -> None:
    """Test that we can retrieve a specific subgraph by namespace.

    Uses subgraph_agent which has actual subgraphs.
    """
    client = get_e2e_client()

    # 1. Create an assistant with a graph that has subgraphs
    assistant = await client.assistants.create(
        name="Test Subgraph Namespace Assistant",
        description="Assistant for testing single subgraph namespace endpoint",
        graph_id="subgraph_agent",
        if_exists="do_nothing",
    )

    try:
        # 2. First get all subgraphs to know a valid namespace
        all_subgraphs = await client.assistants.get_subgraphs(assistant_id=assistant["assistant_id"])
        assert len(all_subgraphs) > 0, "subgraph_agent should have subgraphs"
        target_namespace = next(iter(all_subgraphs.keys()))

        # 3. Query the specific namespace
        subgraph = await client.assistants.get_subgraphs(
            assistant_id=assistant["assistant_id"],
            namespace=target_namespace,
        )

        assert isinstance(subgraph, dict)
        assert set(subgraph.keys()) == {target_namespace}
        assert subgraph[target_namespace] == all_subgraphs[target_namespace]

        elog(
            "Single subgraph retrieved successfully by namespace",
            {
                "assistant_id": assistant["assistant_id"],
                "namespace": target_namespace,
            },
        )
    finally:
        await client.assistants.delete(assistant_id=assistant["assistant_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_assistant_subgraph_by_namespace_not_found() -> None:
    """Test that querying a non-existent subgraph namespace returns 404."""
    client = get_e2e_client()

    assistant = await client.assistants.create(
        name="Test Subgraph 404 Assistant",
        description="Assistant for testing 404 on missing subgraph namespace",
        graph_id="subgraph_agent",
        if_exists="do_nothing",
    )

    try:
        with pytest.raises(NotFoundError) as exc_info:
            await client.assistants.get_subgraphs(
                assistant_id=assistant["assistant_id"],
                namespace="non_existent_namespace_xyz",
            )
        assert exc_info.value.status_code == 404
        elog("Subgraphs endpoint correctly returns 404 for non-existent namespace", {})
    finally:
        await client.assistants.delete(assistant_id=assistant["assistant_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_system_assistant_graph_access():
    """
    Test that we can access graph for system-created assistants.
    """
    client = get_e2e_client()

    # Get the system assistant (should be created on startup)
    assistants = await client.assistants.search()

    # Find a system assistant (created with graph_id)
    system_assistant = None
    for assistant in assistants:
        if assistant.get("user_id") == "system":
            system_assistant = assistant
            break

    if system_assistant:
        # Try to get the graph for system assistant
        graph = await client.assistants.get_graph(assistant_id=system_assistant["assistant_id"])

        # Verify graph structure
        assert "nodes" in graph, "Graph should have nodes"
        assert "edges" in graph, "Graph should have edges"

        elog(
            "System assistant graph accessed successfully",
            {
                "assistant_id": system_assistant["assistant_id"],
                "node_count": len(graph["nodes"]),
            },
        )
    else:
        elog("No system assistant found, skipping test", {})
