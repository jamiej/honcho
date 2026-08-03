"""OpenInference projection over the captured LLM trace stream.

`PhoenixExporter` receives `CapturedLLMCall` and projects them into OpenTelemetry
spans using OpenInference semantic conventions, mimicking the structure of
the Langfuse trace tree.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import SpanContext, NonRecordingSpan
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from src.config import settings
from src.llm.capture import CapturedLLMCall
from src.telemetry import phoenix_session

logger = logging.getLogger(__name__)

# finish_reason values that mark the generation as failed.
_ERROR_FINISHES = frozenset({"error", "cancelled"})

class PhoenixExporter:
    """LLMCallExporter that projects captured calls onto OpenInference traces."""

    def __init__(self) -> None:
        self._tracer = trace.get_tracer("honcho.telemetry.phoenix")

    def export(self, call: CapturedLLMCall) -> None:
        if not settings.phoenix_exporter_enabled:
            return
        try:
            self._export(call)
        except Exception:  # pragma: no cover
            logger.debug("Phoenix exporter failed", exc_info=True)

    def _export(self, call: CapturedLLMCall) -> None:
        trace_key = call.trace_id or call.run_id or call.span_id
        if not trace_key:
            return

        parent_ctx: SpanContext | None = None
        if call.run_id is not None:
            branch = call.agent_type or "_"
            
            root_ctx = self._ensure_trace_root(trace_key, call)
            
            run_ctx = phoenix_session.ensure_run_span(
                trace_key,
                branch,
                lambda: self._create_span(
                    name=call.track_name or "LLM run",
                    parent_ctx=root_ctx,
                    span_kind=OpenInferenceSpanKindValues.CHAIN,
                    call=call
                ),
            )
            parent_ctx = run_ctx
            if call.iteration is not None and run_ctx is not None:
                parent_ctx = phoenix_session.ensure_step_span(
                    trace_key,
                    branch,
                    call.iteration,
                    lambda: self._create_span(
                        name=f"{call.track_name} step" if call.track_name else "Agent step",
                        parent_ctx=run_ctx,
                        span_kind=OpenInferenceSpanKindValues.CHAIN,
                        call=call
                    ),
                )
        
        self._create_generation(
            parent_ctx=parent_ctx,
            call=call,
        )

        if parent_ctx is not None and call.output_tool_calls:
            for seq, tool_call in enumerate(call.output_tool_calls):
                self._create_tool_span(
                    parent_ctx=parent_ctx,
                    tool_call=tool_call,
                    call=call,
                )

    def _ensure_trace_root(self, trace_key: str, call: CapturedLLMCall) -> SpanContext | None:
        if call.parent_category != "dream":
            return None
        return phoenix_session.ensure_trace_root(
            trace_key,
            lambda: self._create_span(
                name=call.track_name or "Dream",
                parent_ctx=None,
                span_kind=OpenInferenceSpanKindValues.CHAIN,
                call=call
            ),
        )

    def _create_span(
        self,
        name: str,
        parent_ctx: SpanContext | None,
        span_kind: OpenInferenceSpanKindValues,
        call: CapturedLLMCall
    ) -> SpanContext | None:
        context = trace.set_span_in_context(NonRecordingSpan(parent_ctx)) if parent_ctx else None
        span = self._tracer.start_span(name=name, context=context)
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind.value)
        self._stamp_metadata(span, call)
        span.end()
        return span.get_span_context()

    def _create_generation(self, parent_ctx: SpanContext | None, call: CapturedLLMCall) -> None:
        context = trace.set_span_in_context(NonRecordingSpan(parent_ctx)) if parent_ctx else None
        name = f"{call.track_name} generation" if call.track_name else "generation"
        
        span = self._tracer.start_span(name=name, context=context)
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        
        if call.model:
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, call.model)
            
        span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps([
            {"role": m.role, "content": m.content, **({"tool_call_id": m.tool_call_id} if getattr(m, "tool_call_id", None) else {})} 
            for m in call.input_messages
        ]))
        
        if isinstance(call.output_content, str) and call.output_content.strip():
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, call.output_content)
        elif call.output_tool_calls:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps([tc.get("name") for tc in call.output_tool_calls]))

        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, call.input_tokens)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, call.output_tokens)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, call.input_tokens + call.output_tokens)
        
        self._stamp_metadata(span, call)
        span.end()

    def _create_tool_span(
        self,
        parent_ctx: SpanContext,
        tool_call: dict[str, Any],
        call: CapturedLLMCall,
    ) -> None:
        context = trace.set_span_in_context(NonRecordingSpan(parent_ctx))
        name = str(tool_call.get("name") or "tool")
        
        span = self._tracer.start_span(name=name, context=context)
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
        
        tool_input = tool_call.get("input")
        if tool_input:
            span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input))
            
        self._stamp_metadata(span, call)
        span.end()

    def _stamp_metadata(self, span: trace.Span, call: CapturedLLMCall) -> None:
        span.set_attribute("honcho.trace_id", call.trace_id or "")
        span.set_attribute("honcho.session_id", call.session_id or "")
        span.set_attribute("honcho.namespace", str(settings.NAMESPACE))
        if call.workspace_name:
            span.set_attribute("honcho.workspace_name", call.workspace_name)
        if call.agent_type:
            span.set_attribute("honcho.agent_type", call.agent_type)
        if call.call_purpose:
            span.set_attribute("honcho.call_purpose", call.call_purpose)

__all__ = ["PhoenixExporter"]
