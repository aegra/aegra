"""Agent Protocol v2 event streaming.

Thread-scoped SSE streaming and command transport, layered over the
existing run broker. Isolated from the legacy ``runs/stream`` path so the
v2 logic can be reviewed and feature-flagged on its own.
"""
