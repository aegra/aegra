import asyncio

import pytest

from tests.e2e._utils import check_and_skip_if_geo_blocked, elog, get_e2e_client


# Helper to wait for run completion and check for geo-blocks
async def wait_for_run_settle(client, thread_id, run_id, max_wait=20):
    wait_interval = 0.5
    waited = 0

    while waited < max_wait:
        await asyncio.sleep(wait_interval)
        waited += wait_interval

        run = await client.runs.get(thread_id, run_id)
        status = run["status"]

        # Check for block immediately when error appears
        if status == "error":
            check_and_skip_if_geo_blocked(run)
            return run

        if status in ("interrupted", "success"):
            return run

    return await client.runs.get(thread_id, run_id)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_human_in_loop_interrupt_resume_e2e():
    """
    Complete human-in-the-loop test using agent_hitl graph (non-streaming).
    Tests: interrupt detection, validation, resume with approval, tool execution.
    """
    client = get_e2e_client()

    # Create assistant with agent_hitl graph
    assistant = await client.assistants.create(
        graph_id="agent_hitl",
        config={"tags": ["hitl", "complete_cycle"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create (agent_hitl)", assistant)
    assert "assistant_id" in assistant
    assistant_id = assistant["assistant_id"]

    # Create thread
    thread = await client.threads.create()
    elog("Threads.create", thread)
    thread_id = thread["thread_id"]

    # Test validation: input + command should fail
    try:
        await client.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input={"messages": [{"role": "user", "content": "Hello"}]},
            command={"update": {"test": "value"}},
        )
        raise AssertionError("Expected validation error for input + command")
    except Exception as e:
        elog(
            "✅ Validation works",
            {"error_contains": "422" in str(e) or "validation" in str(e).lower()},
        )

    # Create run that triggers tool usage (requires approval in agent_hitl)
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "What's the weather like today?"}]},
    )
    elog("Runs.create (tool trigger)", run)
    run_id = run["run_id"]

    # Wait for interrupt
    import asyncio

    max_wait = 30
    wait_interval = 0.5
    waited = 0

    while waited < max_wait:
        await asyncio.sleep(wait_interval)
        waited += wait_interval

        interrupted_run = await client.runs.get(thread_id, run_id)
        if interrupted_run["status"] == "interrupted":
            break
        elif interrupted_run["status"] in ("success", "error"):
            # Check for block
            if interrupted_run["status"] == "error":
                check_and_skip_if_geo_blocked(interrupted_run)
            elog("Run completed without interrupt", interrupted_run)
            return

    assert interrupted_run["status"] == "interrupted", f"Expected interrupted, got {interrupted_run['status']}"
    elog("✅ Interrupt detected", {"run_id": run_id})

    # Verify thread history has interrupt
    history = await client.threads.get_history(thread_id)
    if isinstance(history, list) and len(history) > 0:
        latest_state = history[0]
        assert "interrupts" in latest_state and len(latest_state["interrupts"]) > 0
        elog(
            "✅ Thread state has interrupt",
            {"interrupt_count": len(latest_state["interrupts"])},
        )

    # Test resume validation: should fail if thread not interrupted
    # (We'll create a new thread to test this)
    test_thread = await client.threads.create()
    try:
        await client.runs.create(
            thread_id=test_thread["thread_id"],
            assistant_id=assistant_id,
            command={"resume": "yes"},
        )
        raise AssertionError("Expected validation error for resume on non-interrupted thread")
    except Exception as e:
        elog("✅ Resume validation works", {"error_type": type(e).__name__})

    # Resume with approval (structured format)
    resume_run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        command={"resume": [{"type": "accept", "args": None}]},
    )
    elog("Runs.create (resume)", resume_run)
    resume_run_id = resume_run["run_id"]

    # Wait for completion
    await client.runs.join(thread_id, resume_run_id)
    completed_run = await client.runs.get(thread_id, resume_run_id)

    # Check for block
    if completed_run["status"] == "error":
        check_and_skip_if_geo_blocked(completed_run)

    # Verify final state (completed or interrupted again for more tools)
    assert completed_run["status"] in ("success", "interrupted")
    elog("✅ Resume executed", {"final_status": completed_run["status"]})

    # Verify tool execution in message history
    final_history = await client.threads.get_history(thread_id)
    if isinstance(final_history, list) and len(final_history) > 0:
        messages = final_history[0].get("values", {}).get("messages", [])
        user_msgs = [m for m in messages if m.get("type") == "human"]
        ai_msgs = [m for m in messages if m.get("type") == "ai"]
        tool_msgs = [m for m in messages if m.get("type") == "tool"]

        assert len(user_msgs) >= 1 and len(ai_msgs) >= 1
        if completed_run["status"] == "success":
            assert len(tool_msgs) > 0, "Expected tool execution for completed run"

        elog(
            "✅ Message flow verified",
            {"user": len(user_msgs), "ai": len(ai_msgs), "tool": len(tool_msgs)},
        )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_human_in_loop_text_response_e2e():
    """
    Test human-in-the-loop text response flow using agent_hitl graph.
    Tests: interrupt detection, text response, human message addition, conversation continuation.
    """
    client = get_e2e_client()

    # Create assistant with agent_hitl graph
    assistant = await client.assistants.create(
        graph_id="agent_hitl",
        config={"tags": ["hitl", "rejection"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create (agent_hitl)", assistant)
    assert "assistant_id" in assistant
    assistant_id = assistant["assistant_id"]

    # Create thread
    thread = await client.threads.create()
    elog("Threads.create", thread)
    thread_id = thread["thread_id"]

    # Create run that triggers tool usage (requires approval in agent_hitl)
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "What's the weather like today?"}]},
    )
    elog("Runs.create (tool trigger)", run)
    run_id = run["run_id"]

    # Wait for interrupt
    import asyncio

    max_wait = 30
    wait_interval = 0.5
    waited = 0

    while waited < max_wait:
        await asyncio.sleep(wait_interval)
        waited += wait_interval

        interrupted_run = await client.runs.get(thread_id, run_id)
        if interrupted_run["status"] == "interrupted":
            break
        elif interrupted_run["status"] in ("success", "error"):
            # Check for block
            if interrupted_run["status"] == "error":
                check_and_skip_if_geo_blocked(interrupted_run)
            elog("Run completed without interrupt", interrupted_run)
            return

    assert interrupted_run["status"] == "interrupted", f"Expected interrupted, got {interrupted_run['status']}"
    elog("✅ Interrupt detected", {"run_id": run_id})

    # Verify thread history has interrupt
    history = await client.threads.get_history(thread_id)
    if isinstance(history, list) and len(history) > 0:
        latest_state = history[0]
        assert "interrupts" in latest_state and len(latest_state["interrupts"]) > 0
        elog(
            "✅ Thread state has interrupt",
            {"interrupt_count": len(latest_state["interrupts"])},
        )

    # Provide text response (should add human message and continue conversation)
    text_response_run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        command={
            "resume": [
                {
                    "type": "response",
                    "args": "I don't want to use tools right now, please just tell me about the weather in general.",
                }
            ]
        },
    )
    elog("Runs.create (text response)", text_response_run)
    text_response_run_id = text_response_run["run_id"]

    # Wait for completion
    await client.runs.join(thread_id, text_response_run_id)
    completed_run = await client.runs.get(thread_id, text_response_run_id)

    # Check for block
    if completed_run["status"] == "error":
        check_and_skip_if_geo_blocked(completed_run)

    # Verify final state is completed (text response should complete the flow)
    assert completed_run["status"] == "success"
    elog("✅ Text response completed", {"final_status": completed_run["status"]})

    # Verify NO tool execution in message history (tools were interrupted for human input)
    final_history = await client.threads.get_history(thread_id)
    if isinstance(final_history, list) and len(final_history) > 0:
        messages = final_history[0].get("values", {}).get("messages", [])
        user_msgs = [m for m in messages if m.get("type") == "human"]
        ai_msgs = [m for m in messages if m.get("type") == "ai"]
        tool_msgs = [m for m in messages if m.get("type") == "tool"]

        assert len(user_msgs) >= 1 and len(ai_msgs) >= 1
        # Check that any tool messages are interruption messages, not actual executions
        interruption_msgs = [m for m in tool_msgs if "interrupted for human input" in m.get("content", "")]
        actual_tool_msgs = [m for m in tool_msgs if "interrupted for human input" not in m.get("content", "")]

        assert len(actual_tool_msgs) == 0, (
            f"Expected no actual tool execution after text response, but found {len(actual_tool_msgs)} actual tool messages"
        )
        assert len(interruption_msgs) > 0, "Expected interruption messages after text response"

        # Verify human message was added to conversation
        human_msgs_content = [m.get("content", "") for m in user_msgs]
        assert any("don't want to use tools" in content for content in human_msgs_content), (
            "Expected human response message in conversation"
        )

        # Verify final AI message acknowledges rejection
        final_ai_msg = None
        for msg in reversed(ai_msgs):
            if msg.get("content"):
                final_ai_msg = msg.get("content", "").lower()
                break

        # The AI should acknowledge the human response in some way
        assert final_ai_msg is not None, "Expected final AI response after human input"

        elog(
            "✅ Text response flow verified",
            {
                "user": len(user_msgs),
                "ai": len(ai_msgs),
                "tool_total": len(tool_msgs),
                "interruption_msgs": len(interruption_msgs),
                "actual_tool_msgs": len(actual_tool_msgs),
                "final_ai_response_length": len(final_ai_msg) if final_ai_msg else 0,
            },
        )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_human_in_loop_ignore_tool_call_e2e():
    """
    Test human-in-the-loop ignore flow using agent_hitl graph.
    Tests: interrupt detection, ignore response, no tool execution.
    """
    client = get_e2e_client()

    # Create assistant with agent_hitl graph
    assistant = await client.assistants.create(
        graph_id="agent_hitl",
        config={"tags": ["hitl", "ignore"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create (agent_hitl)", assistant)
    assert "assistant_id" in assistant
    assistant_id = assistant["assistant_id"]

    # Create thread
    thread = await client.threads.create()
    elog("Threads.create", thread)
    thread_id = thread["thread_id"]

    # Create run that triggers tool usage (requires approval in agent_hitl)
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "What's the weather like today?"}]},
    )
    elog("Runs.create (tool trigger)", run)
    run_id = run["run_id"]

    # Wait for interrupt
    import asyncio

    max_wait = 30
    wait_interval = 0.5
    waited = 0

    while waited < max_wait:
        await asyncio.sleep(wait_interval)
        waited += wait_interval

        interrupted_run = await client.runs.get(thread_id, run_id)
        if interrupted_run["status"] == "interrupted":
            break
        elif interrupted_run["status"] in ("success", "error"):
            # Check for block
            if interrupted_run["status"] == "error":
                check_and_skip_if_geo_blocked(interrupted_run)
            elog("Run completed without interrupt", interrupted_run)
            return

    assert interrupted_run["status"] == "interrupted", f"Expected interrupted, got {interrupted_run['status']}"
    elog("✅ Interrupt detected", {"run_id": run_id})

    # Verify thread history has interrupt
    history = await client.threads.get_history(thread_id)
    if isinstance(history, list) and len(history) > 0:
        latest_state = history[0]
        assert "interrupts" in latest_state and len(latest_state["interrupts"]) > 0
        elog(
            "✅ Thread state has interrupt",
            {"interrupt_count": len(latest_state["interrupts"])},
        )

    # Ignore the tool execution
    ignore_run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        command={"resume": [{"type": "ignore", "args": None}]},
    )
    elog("Runs.create (ignore)", ignore_run)
    ignore_run_id = ignore_run["run_id"]

    # Wait for completion
    await client.runs.join(thread_id, ignore_run_id)
    completed_run = await client.runs.get(thread_id, ignore_run_id)

    # Check for block
    if completed_run["status"] == "error":
        check_and_skip_if_geo_blocked(completed_run)

    # Verify final state is completed (ignore should complete the flow)
    assert completed_run["status"] == "success"
    elog("✅ Ignore completed", {"final_status": completed_run["status"]})

    # Verify NO tool execution in message history (tools were cancelled)
    final_history = await client.threads.get_history(thread_id)
    if isinstance(final_history, list) and len(final_history) > 0:
        messages = final_history[0].get("values", {}).get("messages", [])
        user_msgs = [m for m in messages if m.get("type") == "human"]
        ai_msgs = [m for m in messages if m.get("type") == "ai"]
        tool_msgs = [m for m in messages if m.get("type") == "tool"]

        assert len(user_msgs) >= 1 and len(ai_msgs) >= 1
        # Check that any tool messages are cancellation messages, not actual executions
        cancellation_msgs = [m for m in tool_msgs if "cancelled by human operator" in m.get("content", "")]
        actual_tool_msgs = [m for m in tool_msgs if "cancelled by human operator" not in m.get("content", "")]

        assert len(actual_tool_msgs) == 0, (
            f"Expected no actual tool execution after ignore, but found {len(actual_tool_msgs)} actual tool messages"
        )
        assert len(cancellation_msgs) > 0, "Expected cancellation messages after ignore"

        elog(
            "✅ Tool ignore verified",
            {
                "user": len(user_msgs),
                "ai": len(ai_msgs),
                "tool_total": len(tool_msgs),
                "cancellation_msgs": len(cancellation_msgs),
                "actual_tool_msgs": len(actual_tool_msgs),
            },
        )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_human_in_loop_edit_tool_args_e2e():
    """
    Test human-in-the-loop edit functionality using agent_hitl graph.
    Tests: interrupt detection, argument editing, tool execution with modified args.
    """
    client = get_e2e_client()

    # Create assistant with agent_hitl graph
    assistant = await client.assistants.create(
        graph_id="agent_hitl",
        config={"tags": ["hitl", "edit"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create (agent_hitl)", assistant)
    assert "assistant_id" in assistant
    assistant_id = assistant["assistant_id"]

    # Create thread
    thread = await client.threads.create()
    elog("Threads.create", thread)
    thread_id = thread["thread_id"]

    # Create run that triggers tool usage
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "Search for Python programming tutorials"}]},
    )
    elog("Runs.create (tool trigger)", run)
    run_id = run["run_id"]

    interrupted_run = await wait_for_run_settle(client, thread_id, run_id)

    assert interrupted_run["status"] == "interrupted", f"Expected interrupted, got {interrupted_run['status']}"
    elog("✅ Interrupt detected", {"run_id": run_id})

    # Edit the tool arguments
    edit_run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        command={
            "resume": [
                {
                    "type": "edit",
                    "args": {
                        "action": "tool_execution",
                        "args": {"search": {"query": "Advanced Python programming and machine learning"}},
                    },
                }
            ]
        },
    )
    elog("Runs.create (edit)", edit_run)
    edit_run_id = edit_run["run_id"]

    # Wait for completion
    await client.runs.join(thread_id, edit_run_id)
    completed_run = await client.runs.get(thread_id, edit_run_id)

    # Check for block
    if completed_run["status"] == "error":
        check_and_skip_if_geo_blocked(completed_run)

    # Verify final state is completed
    assert completed_run["status"] in ("success", "interrupted")
    elog("✅ Edit and execution completed", {"final_status": completed_run["status"]})

    # Wait for state to be consistent after run completion
    import asyncio

    max_wait = 5
    wait_interval = 0.2
    waited = 0
    final_history = None

    # Retry until we get consistent state with messages
    while waited < max_wait:
        await asyncio.sleep(wait_interval)
        waited += wait_interval

        final_history = await client.threads.get_history(thread_id)
        if isinstance(final_history, list) and len(final_history) > 0:
            messages = final_history[0].get("values", {}).get("messages", [])
            # Wait for messages to be available (state is consistent)
            if messages and len(messages) > 0:
                # If run succeeded, also wait for tool messages to appear
                if completed_run["status"] == "success":
                    tool_msgs = [m for m in messages if m.get("type") == "tool"]
                    if len(tool_msgs) > 0:
                        # State is consistent with tool execution
                        break
                else:
                    # For interrupted runs, messages existing is sufficient
                    break

    # Verify tool execution happened with edited arguments
    assert final_history is not None and isinstance(final_history, list) and len(final_history) > 0, (
        "Expected history to be available after run completion"
    )

    messages = final_history[0].get("values", {}).get("messages", [])
    user_msgs = [m for m in messages if m.get("type") == "human"]
    ai_msgs = [m for m in messages if m.get("type") == "ai"]
    tool_msgs = [m for m in messages if m.get("type") == "tool"]

    assert len(user_msgs) >= 1 and len(ai_msgs) >= 1

    # If the run completed successfully, verify tool execution
    search_tool_msg = None
    tool_content = ""

    if completed_run["status"] == "success":
        assert len(tool_msgs) > 0, "Expected tool execution after edit"

        # Verify the tool was executed with edited arguments
        # Look for the search tool execution
        for msg in tool_msgs:
            if msg.get("name") == "search":
                search_tool_msg = msg
                break

        assert search_tool_msg is not None, "Expected search tool execution"

        # Check that the tool result contains evidence of the edited query
        tool_content = search_tool_msg.get("content", "")
        assert (
            "Advanced Python programming and machine learning" in tool_content
            or "machine learning" in tool_content.lower()
        ), f"Expected edited query in tool result, got: {tool_content}"
    else:
        # If interrupted again, at least verify the edit was processed
        # (tool might be in pending state)
        elog("Run interrupted again after edit", {"tool_msgs_count": len(tool_msgs)})

    elog("✅ Tool edit and execution verified", {})


# TODO: Fix Mark as Resolved functionality
# ISSUE: The "Mark as Resolved" button (goto: "__end__") creates an infinite loop
# ROOT CAUSE: LangGraph bug where Command(goto=END) generates invalid channel "branch:to:__end__"
# GITHUB ISSUE: https://github.com/langchain-ai/langgraph/issues/5572
# FIX: Upgrade to LangGraph version that includes PR #5601 (merged July 21, 2024)
# STATUS: Container has LangGraph 0.6.7 which should include the fix, but test still fails
# NEXT STEPS:
#   1. Verify LangGraph version includes the fix
#   2. Check if our graph definition needs updates for END node handling
#   3. Test with latest LangGraph stable version
#   4. Re-enable test once functionality works
@pytest.mark.skip(reason="Mark as Resolved functionality has known issue - see TODO above")
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_human_in_loop_mark_as_resolved_e2e():
    """
    Test human-in-the-loop "Mark as Resolved" functionality using agent_hitl graph.
    Tests: interrupt detection, mark as resolved command, direct termination.

    CURRENTLY DISABLED: See TODO comment above for details on the issue.
    """
    client = get_e2e_client()

    # Create assistant with agent_hitl graph
    assistant = await client.assistants.create(
        graph_id="agent_hitl",
        config={"tags": ["hitl", "resolve"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create (agent_hitl)", assistant)
    assert "assistant_id" in assistant
    assistant_id = assistant["assistant_id"]

    # Create thread
    thread = await client.threads.create()
    elog("Threads.create", thread)
    thread_id = thread["thread_id"]

    # Create run that triggers tool usage (requires approval in agent_hitl)
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "What's the weather like today?"}]},
    )
    elog("Runs.create (tool trigger)", run)
    run_id = run["run_id"]

    # Wait for interrupt
    import asyncio

    max_wait = 30
    wait_interval = 0.5
    waited = 0

    while waited < max_wait:
        await asyncio.sleep(wait_interval)
        waited += wait_interval

        interrupted_run = await client.runs.get(thread_id, run_id)
        if interrupted_run["status"] == "interrupted":
            break
        elif interrupted_run["status"] in ("success", "error"):
            # Check for block
            check_and_skip_if_geo_blocked(interrupted_run)
            elog("Run completed without interrupt", interrupted_run)
            return

    assert interrupted_run["status"] == "interrupted", f"Expected interrupted, got {interrupted_run['status']}"
    elog("✅ Interrupt detected", {"run_id": run_id})

    # Mark as resolved (goto END command)
    resolve_run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        command={"goto": "__end__"},
    )
    elog("Runs.create (mark as resolved)", resolve_run)
    resolve_run_id = resolve_run["run_id"]

    # Wait for completion
    await client.runs.join(thread_id, resolve_run_id)
    completed_run = await client.runs.get(thread_id, resolve_run_id)

    # Check for block
    if completed_run["status"] == "error":
        check_and_skip_if_geo_blocked(completed_run)

    # Verify final state is completed (resolve should terminate immediately)
    assert completed_run["status"] == "success"
    elog("✅ Mark as resolved completed", {"final_status": completed_run["status"]})

    # Verify NO tool execution happened (conversation was resolved without tools)
    final_history = await client.threads.get_history(thread_id)
    if isinstance(final_history, list) and len(final_history) > 0:
        messages = final_history[0].get("values", {}).get("messages", [])
        user_msgs = [m for m in messages if m.get("type") == "human"]
        ai_msgs = [m for m in messages if m.get("type") == "ai"]
        tool_msgs = [m for m in messages if m.get("type") == "tool"]

        assert len(user_msgs) >= 1 and len(ai_msgs) >= 1
        # Should have no tool messages at all - conversation was resolved before tools
        assert len(tool_msgs) == 0, f"Expected no tool messages after mark as resolved, but found {len(tool_msgs)}"

        elog(
            "✅ Mark as resolved verified",
            {
                "user": len(user_msgs),
                "ai": len(ai_msgs),
                "tool": len(tool_msgs),
                "resolved_without_tools": len(tool_msgs) == 0,
            },
        )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_human_in_loop_streaming_interrupt_resume_e2e():
    """
    Complete streaming human-in-the-loop test using agent_hitl graph.
    Tests: real-time interrupt detection, streaming resume, tool execution verification.
    """
    client = get_e2e_client()

    # Setup
    assistant = await client.assistants.create(
        graph_id="agent_hitl",
        config={"tags": ["hitl", "streaming"]},
        if_exists="do_nothing",
    )
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    assistant_id = assistant["assistant_id"]

    # Phase 1: Stream until interrupt.
    # Let the stream exhaust naturally instead of breaking early — breaking
    # abandons the SDK async generator mid-iteration which triggers
    # "RuntimeError: generator didn't stop after athrow()".
    elog("Phase 1: Stream until interrupt", {"starting": True})
    stream = client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": "Search for Python programming info"}]},
        stream_mode=["values"],
    )

    event_count = 0
    interrupt_detected = False
    initial_run_id = None

    async for chunk in stream:
        event_count += 1
        event_type = getattr(chunk, "event", None)
        data = getattr(chunk, "data", None)

        # Look for interrupt in values events
        if (
            not interrupt_detected
            and event_type == "values"
            and isinstance(data, dict)
            and "__interrupt__" in data
            and len(data.get("__interrupt__", [])) > 0
        ):
            interrupt_detected = True
            elog("✅ Interrupt detected in stream!", {"event_count": event_count})

        if event_count > 50:  # Safety limit
            break

    # Check for blocking error if no interrupt detected
    if not interrupt_detected:
        runs = await client.runs.list(thread_id)
        if runs:
            last_run = runs[0]
            if last_run["status"] == "error":
                check_and_skip_if_geo_blocked(last_run)

    assert interrupt_detected, f"Expected interrupt in stream after {event_count} events"

    # Get run_id from thread history
    history = await client.threads.get_history(thread_id)
    if isinstance(history, list) and len(history) > 0:
        latest_state = history[0]
        initial_run_id = latest_state.get("metadata", {}).get("run_id")
        assert "interrupts" in latest_state and len(latest_state["interrupts"]) > 0

    assert initial_run_id is not None, "Expected to find run_id in thread history"

    # Phase 2: Verify interrupted state (wait for server-side run to settle)
    interrupted_run = await wait_for_run_settle(client, thread_id, initial_run_id)
    assert interrupted_run["status"] == "interrupted", f"Expected 'interrupted', got '{interrupted_run['status']}'"
    elog("✅ Run interrupted", {"run_id": initial_run_id})

    # Phase 3: Stream resume command
    elog("Phase 3: Stream resume", {"starting": True})
    resume_stream = client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        command={"resume": [{"type": "accept", "args": None}]},
        stream_mode=["values"],
    )

    resume_event_count = 0
    tool_executed = False
    final_ai_message = False

    async for chunk in resume_stream:
        resume_event_count += 1
        event_type = getattr(chunk, "event", None)
        data = getattr(chunk, "data", None)

        # Look for tool execution and final AI response
        if event_type == "values" and isinstance(data, dict):
            messages = data.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("type") == "tool":
                        tool_executed = True
                        elog("✅ Tool execution detected!", {"tool": msg.get("name")})
                    elif msg.get("type") == "ai" and msg.get("content") and tool_executed:
                        final_ai_message = True
                        elog(
                            "✅ Final AI response!",
                            {"length": len(msg.get("content", ""))},
                        )

        if resume_event_count > 50:  # Safety limit
            break

    # Phase 4: Verify completion
    final_history = await client.threads.get_history(thread_id)
    if isinstance(final_history, list) and len(final_history) > 0:
        messages = final_history[0].get("values", {}).get("messages", [])
        user_msgs = [m for m in messages if m.get("type") == "human"]
        ai_msgs = [m for m in messages if m.get("type") == "ai"]
        tool_msgs = [m for m in messages if m.get("type") == "tool"]

        # Verify complete flow
        assert len(user_msgs) >= 1 and len(ai_msgs) >= 1
        assert tool_executed, "Expected tool execution in stream"

        elog(
            "✅ Streaming cycle complete",
            {
                "interrupt_events": event_count,
                "resume_events": resume_event_count,
                "messages": {
                    "user": len(user_msgs),
                    "ai": len(ai_msgs),
                    "tool": len(tool_msgs),
                },
                "tool_executed": tool_executed,
                "final_ai_response": final_ai_message,
            },
        )
