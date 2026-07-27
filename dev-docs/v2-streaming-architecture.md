# v2 Event Streaming — Maintainer / Debug Guide

Internal engineering notes for Agent Protocol v2 streaming. NOT user-facing.
Lives outside `docs/` so Mintlify never publishes it. Read this before debugging
a v2 streaming issue.

---

## 1. One-paragraph model

v2 is a **thread-scoped** SSE protocol (v1 is run-scoped). A client opens ONE
stream on a thread, fires commands, and events for **any run on that thread**
flow down the single open connection. Events travel
graph → `native_stream` → broker → `session` → socket. `session.py` is the only
file with real logic (seq stamping, channel filter, interrupt split, lifecycle
derivation); everything else forwards. v1 is untouched; a per-run flag
(`event_streaming_v2`) routes each run to the v2 or v1 stream consumer.

---

## 2. Endpoints (entire public surface)

```
POST /threads/{id}/stream/events   → SSE stream (long-lived)
POST /threads/{id}/commands        → one command in, one JSON envelope out
```

Client choreography — every bug traces back to this sequence:

```
1. open   stream/events          (SSE stays open for the whole session)
2. post   commands {run.start}   (starts run A)
3. run A's events flow down the step-1 stream
4. [HITL] interrupt surfaces on the stream as input.requested
   → post commands {input.respond} → starts run B on the SAME thread
   → run B's events flow down the SAME step-1 stream (never reopened)
5. lifecycle: completed closes it
```

Resume is **a new run on the same open stream**, not a new connection. Hold this.

---

## 3. Pipeline (physical path of an event)

```
LangGraph graph
  │  astream_events(version="v3")            engine emits protocol-shaped events
  ▼
services/event_streaming/native_stream.py    thin: forward, unwrap [event,meta] msg tuples
  │  (method, event) pairs
  ▼
services/broker.py  (in-memory)  OR  redis_broker.py  (prod)
  │  per-run buffer + live fan-out to EVERY subscriber
  ▼
services/event_streaming/session.py  (ThreadEventSession)   THE BRAIN
  │  re-stamp seq, channel filter, split interrupts, derive lifecycle
  ▼
api/event_streaming.py  (_frame_events)      frame as SSE, write to socket
  ▼
client SDK
```

Which file when X breaks:

| Symptom | File |
| --- | --- |
| wrong event *content* (tool call malformed, message shape) | `normalizers.py` |
| event on wrong channel / missing lifecycle / seq off / interrupt not split | `session.py` |
| one consumer gets events, another doesn't / HITL watcher blind | `broker.py` / `redis_broker.py` |
| run won't start / resume rejected / auth | `commands.py` + `run_preparation.py` |
| everything v2 dead / 503 | `capabilities.py` + `FF_V2_EVENT_STREAMING` |
| old thread hangs on open | `_drain_run` status backstop (`session.py`) |

---

## 4. `session.py` — per-event work

`_project → _channel_events → _emit` for each raw broker event:

1. **seq** — thread-monotonic counter, the reconnect cursor. Client resends last
   `seq` as `since`; session skips `<= since`. seq increments BEFORE channel
   filtering (absolute), so reconnecting with a different channel set still
   resumes correctly.
2. **channel filter** — drop events for channels the client didn't subscribe to.
3. **interrupt split** — an interrupt rides inside `values`/`updates`; pull it
   onto the `input` channel as `input.requested`. This is how HITL surfaces.
4. **lifecycle derivation** — root `running` seed at run start; per-subgraph
   `started`/`completed`/`failed`; terminal `completed`/`failed`/`interrupted`
   at end; terminal cascades any still-open subgraph namespaces (deepest first).

---

## 5. Broker — the subtle one (broke HITL twice)

Per-run object: buffer (reconnect replay) + live fan-out.

**The gotcha:** the v2 SDK opens **TWO** SSE per run — the main event stream AND a
separate lifecycle-watcher (channels `lifecycle`+`input`) that populates
`thread.interrupts`. Both must receive **every** event. The original in-memory
broker used one shared queue → the two consumers competed → watcher missed the
interrupt → resume broke. Redis pub/sub already fanned out, so **prod worked
while dev was broken**. Fixed: each `aiter()` gets its own queue; `put()` writes
to all; replay buffer on subscribe so nothing is dropped.

**Rule: always `make e2e-both` (dev in-memory + prod Redis).** Prod-only
verification hides dev broker-fanout bugs.

**Expired-broker wedge (fixed):** opening a stream on an OLD thread whose run
events expired (Redis TTL ~10 min / in-memory cleanup ~1 h) used to recreate an
empty never-finished broker → `aiter` waited forever → whole stream hung with
zero events. Fix: the run lister returns `(run_id, status, graph_name)` and
`_drain_run` uses the persisted status as a backstop — terminal status + empty
replay = silent skip; terminal + partial replay = synthesized terminal.

---

## 6. HITL flow (the hardest path)

```
run A → interrupt
  → session splits it onto input channel → client sees input.requested
  → SDK watcher SSE registers it in thread.interrupts
  → client posts commands {input.respond, interrupt_id, response}
  → commands.py builds Command(resume={interrupt_id: response})
  → _prepare_run starts run B on the SAME thread
  → session (still open) picks up run B, streams it → lifecycle: completed
```

Four fixed regression points (likely places a future change re-breaks HITL):

- **Stream must stay open** across the A→B gap — idle-grace applied after every
  drain, not exit-on-first-drain.
- **Assistant recovery** — SDK sends no `assistant_id` on respond; recovered from
  the thread's most recent run (`_thread_assistant_id`, user-scoped).
- **Resume race** — the interrupt reaches the client before the run executor
  commits `thread_status="interrupted"`; a fast respond gets a spurious 400.
  Fix: after the first read isn't interrupted, poll a few FRESH short-lived
  sessions before rejecting. Do NOT rollback the request session (broke a v1
  mocked-session test).
- **Resume by id** — `Command(resume={interrupt_id: value})` map; langgraph
  detects a map by all-keys-being-xxh3-128-hexdigests. A bare value RAISES when
  more than one interrupt is pending. Batch form: `{responses:[{interrupt_id,
  response}]}` merged into one map. Unknown/malformed id → `no_such_interrupt`.

---

## 7. Security (the config-injection class)

Threat: a client sends another user's `thread_id` (in the path, or smuggled via
`config.configurable.thread_id`, or via a checkpoint) and runs on their state.
The checkpointer keys state **solely on thread_id**, so this must be airtight.

Three layers, all present + tested:

1. **Route ownership check** — `_verify_thread_owned_or_new` at the top of BOTH
   v2 routes → 404 if the path thread_id is owned by another user. Runs before
   any run work. Tests: `test_cross_tenant_thread_is_404` (both routes).
2. **Server forces thread_id** — `create_run_config` (`langgraph_service.py`)
   merges client config additively but then OVERWRITES
   `configurable.thread_id`/`run_id` with the route-verified values. A body- or
   checkpoint-supplied thread_id is discarded. The v2 path uses this via
   `run_executor.py` (shared executor). Tests:
   `test_create_run_config_ignores_client_thread_id_override`,
   `test_create_run_config_checkpoint_cannot_override_thread_id`,
   `test_thread_checkpoint_scoping.py`.
3. **SQL tenant filter** — every v2 DB query carries `user_id == user.identity`
   (`_thread_assistant_id`, `_thread_is_interrupted`, the run lister); assistant
   resolution is `user.identity OR "system"`. Required because `@auth.on` is
   default-allow when no handler matches (GHSA-m98r-6667-4wq7).

Known latent debt (pre-existing, NOT v2, not exploitable): `update_thread_metadata`
in `run_preparation.py` reads/updates the thread row with no user_id filter. Only
reached after the route ownership check passes, so no live hole. Belongs in the
auth-dispatch hoist workstream (the ThreadService/RunService follow-ups to #401).

---

## 8. Flag / kill switch (operational lever)

`FF_V2_EVENT_STREAMING` — **default ON**. Set `false` → both v2 routes return 503
(they stay mounted, capability-gated), v2 is dead, v1 fully unaffected. This is
the **instant rollback** if v2 misbehaves in prod: flip the env var, no code
redeploy. Also 503 if the runtime is too old (capability probe in
`capabilities.py` checks for the required langgraph symbols).

---

## 9. Heartbeats

The v2 stream route uses `make_sse_response`, which sets
`ping=SSE_PING_INTERVAL_SECS` with a cached `: heartbeat` comment frame. Keeps the
connection warm while the graph is silent (LLM call, sitting on an interrupt) so
proxies/LBs don't kill an idle SSE. The heartbeat is a comment frame — no seq,
clients ignore it; it exists purely to keep the socket alive.

---

## 10. Deliberately NOT implemented (don't chase these as bugs)

- **`tools` channel** — silent. The streaming engine (`astream_events v3`) has no
  tools transformer on the transformer path we use; `StreamToolCallHandler` (added
  in langgraph 1.2.7) rides the `stream_mode=["tools"]` path, which v3 rejects
  (v3 builds stream modes only from transformer `required_stream_modes`, and no
  tools transformer exists). Tool calls + results are still fully visible via the
  `messages` (content-block `tool_call` chunks) and `values`/`updates` (tool-role
  messages, matched by `tool_call_id`) channels — verified live. Revisit when
  langgraph ships a ToolsTransformer: then it's forward the events + promote
  `__node`→`params.node` + optional tool-output-message mirroring + a capability
  probe.
- **WebSocket transport + WS-only commands** (`subscription.*`, `agent.getTree`) —
  N/A, we're SSE.
- **Multi-run live interleaving** — runs drain oldest-first; concurrent runs on
  one thread are not interleaved (newer run's events wait for the older to end).
  Rare; a bigger redesign if ever needed.

---

## 11. Debug cheat-sheet

| Symptom | First thing to check |
| --- | --- |
| client hangs, zero events | broker empty/wedged, or `_drain_run` status backstop |
| HITL resume returns 400 | resume race (`_validate_resume_command`) or interrupt_id shape |
| works in prod, not dev (or reverse) | broker fan-out / Redis-vs-in-memory divergence — run `make e2e-both` |
| wrong event content | `normalizers.py` |
| wrong channel / no lifecycle / seq off | `session.py` |
| cross-tenant leak fear | route `_verify_thread_owned_or_new` + `create_run_config` thread_id force |
| all v2 dead / 503 | `FF_V2_EVENT_STREAMING` off, or capability probe failing |
| old thread hangs on open | expired-broker wedge → `_drain_run` status backstop |

---

## 12. Test map

- Unit — `tests/unit/test_services/test_event_streaming/` (session, normalizers,
  commands, protocol, capabilities, broker).
- Integration — `tests/integration/test_api/test_event_streaming.py` (routes,
  validation, ownership 404, SSE frames; DB + broker faked).
- E2E — `tests/e2e/test_event_streaming/test_sdk_v2_e2e.py` (stock langgraph SDK
  against a real server; the only layer that proves wire compat + the two-SSE
  fan-out + real HITL). MUST pass in both dev and prod: `make e2e-both`.

Integration canNOT replace e2e: it mocks the broker + DB + real SDK, so the two
biggest historical bugs (broker fan-out, resume race, expired-broker wedge) are
invisible to it by construction. Keep both levels.
