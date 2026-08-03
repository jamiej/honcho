"""Per-trace span context registry backing the `PhoenixExporter`.

The exporter sees one `CapturedLLMCall` at a time, but a single agentic run fans
out into many calls that must nest under one run span with per-iteration step
spans. OpenTelemetry links observations by SpanContext, and each ID is minted
fresh and unpredictable — so this module remembers the run/step contexts created
for a trace and hands them back as the parent of later calls.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from opentelemetry.trace import SpanContext

logger = logging.getLogger(__name__)

_MAX_TRACES = 4096

@dataclass
class _TraceState:
    root_span_context: SpanContext | None = None
    run_span_contexts: dict[str, SpanContext] = field(default_factory=dict)
    step_span_contexts: dict[tuple[str, int], SpanContext] = field(default_factory=dict)

_traces: OrderedDict[str, _TraceState] = OrderedDict()
_lock = threading.Lock()

def _get_or_create_state(trace_key: str) -> _TraceState:
    state = _traces.get(trace_key)
    if state is None:
        if len(_traces) >= _MAX_TRACES:
            _traces.popitem(last=False)
        state = _traces[trace_key] = _TraceState()
    else:
        _traces.move_to_end(trace_key)
    return state

def ensure_trace_root(trace_key: str, create: Callable[[], SpanContext | None]) -> SpanContext | None:
    with _lock:
        state = _get_or_create_state(trace_key)
        if state.root_span_context is None:
            state.root_span_context = create()
        return state.root_span_context

def ensure_run_span(trace_key: str, branch: str, create: Callable[[], SpanContext | None]) -> SpanContext | None:
    with _lock:
        state = _get_or_create_state(trace_key)
        existing = state.run_span_contexts.get(branch)
        if existing is None:
            existing = create()
            if existing is not None:
                state.run_span_contexts[branch] = existing
        return existing

def ensure_step_span(trace_key: str, branch: str, iteration: int, create: Callable[[], SpanContext | None]) -> SpanContext | None:
    with _lock:
        state = _get_or_create_state(trace_key)
        key = (branch, iteration)
        existing = state.step_span_contexts.get(key)
        if existing is None:
            existing = create()
            if existing is not None:
                state.step_span_contexts[key] = existing
        return existing

def reset() -> None:
    """Drop all tracked traces — used on shutdown and in tests."""
    with _lock:
        _traces.clear()
