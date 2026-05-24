"""Setup script for the cron_example graph.

Run this once to create an assistant and a cron that fires every minute.
The cron keeps a persistent thread, so the message history accumulates
across runs — each tick appends a new message.

Usage:
  python examples/cron_example/setup.py

Requires the server to be running:
  CRON_POLL_INTERVAL_SECONDS=60 uv run aegra dev --no-db-check
"""

import asyncio

import httpx
from langgraph_sdk import get_client


async def main() -> None:
    """Create the example assistant, thread, and cron against a local server."""
    client = get_client(url="http://localhost:2026")

    # Create a dedicated thread so history persists across runs
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print(f"Thread created: {thread_id}")

    # Create an assistant for the cron_example graph
    assistant = await client.assistants.create(graph_id="cron_example")
    assistant_id = assistant["assistant_id"]
    print(f"Assistant created: {assistant_id}")

    # Schedule the agent to run every minute, bound to the thread.
    # Use POST /threads/{thread_id}/runs/crons so every tick reuses the same thread.
    # The response is a Run (the first immediate run), not a Cron record.
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            f"http://localhost:2026/threads/{thread_id}/runs/crons",
            json={
                "assistant_id": assistant_id,
                "schedule": "* * * * *",
                "input": {"messages": []},
            },
        )
        resp.raise_for_status()
        first_run = resp.json()

    print("Cron created  schedule=* * * * *")
    print(f"First run:    {first_run['run_id']}")
    print()
    print("The agent will tick every minute and append a message to the thread.")
    print(f"Check history: GET /threads/{thread_id}/history")


if __name__ == "__main__":
    asyncio.run(main())
