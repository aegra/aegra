# LangGraph.js Chatbot Example

A simple chatbot built with [LangGraph.js](https://github.com/langchain-ai/langgraphjs) running on [Aegra](https://github.com/ibbybuilds/aegra).

## Overview

This example demonstrates using a LangGraph.js graph with Aegra. The graph is a simple chatbot that:
- Takes messages as input
- Calls OpenAI's GPT-4o-mini model
- Returns the model's response

Checkpointing is handled natively by the JS bridge using `@langchain/langgraph-checkpoint-postgres`, which enables full interrupt, resume, and time-travel support.

## Setup

### Prerequisites
- Node.js 18+ installed
- OpenAI API key

### Configuration

Add to your `aegra.json`:

```json
{
  "graphs": {
    "js_chatbot": {
      "runtime": "langgraphjs",
      "path": "./examples/langgraphjs_chatbot/graph.ts:graph"
    }
  }
}
```

### Install Dependencies

```bash
cd examples/langgraphjs_chatbot
npm install
```

### Environment

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

## Usage

Once Aegra is running (`aegra dev`), use the LangGraph SDK:

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2026")

# Create a thread
thread = await client.threads.create()

# Stream a conversation
async for chunk in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id="js_chatbot",
    input={"messages": [{"type": "human", "content": "Hello!"}]},
):
    print(chunk)
```

## Graph Structure

```text
__start__ → chatbot → __end__
```

Single-node graph — the `chatbot` node invokes the LLM with the full message history and returns the response.

## Architecture Notes

- The graph is exported as an **uncompiled** `StateGraph` — the Aegra JS bridge compiles it with a native `PostgresSaver` checkpointer at load time.
- Do **not** call `builder.compile()` in your graph file — let the bridge handle compilation so checkpointing works correctly.
- The `DATABASE_URL` environment variable is passed to the JS bridge process, normalised to `postgresql://` scheme.
