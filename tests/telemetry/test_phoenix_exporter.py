import json
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from src.config import settings
from src.llm.capture import CapturedLLMCall, CapturedMessage
from src.telemetry.phoenix_exporter import PhoenixExporter
from src.telemetry import phoenix_session

@pytest.fixture
def phoenix_telemetry(monkeypatch):
    monkeypatch.setattr(settings, "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setattr(settings, "NAMESPACE", "test-namespace")
    phoenix_session.reset()
    
    provider = TracerProvider()
    memory_exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(memory_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    yield memory_exporter
    
    phoenix_session.reset()

def test_export_call_with_generation(phoenix_telemetry):
    exporter = PhoenixExporter()
    
    call = CapturedLLMCall(
        trace_id="test-trace",
        span_id="test-span",
        parent_span_id=None,
        iteration=0,
        step_seq=0,
        attempt=1,
        was_fallback=False,
        run_id="test-run",
        workspace_name="test-workspace",
        call_purpose="test-purpose",
        parent_category="dialectic",
        agent_type="assistant",
        session_id="test-session",
        observer=None,
        observed=None,
        peer_name=None,
        track_name="chat",
        transport="openai",
        provider_label=None,
        model="gpt-4o",
        input_messages=[
            CapturedMessage(role="user", content="Hello", tool_call_id=None, content_hash="hash")
        ],
        tool_schemas=[],
        tool_choice=None,
        output_content="World",
        output_tool_calls=[],
        thinking_content=None,
        thinking_blocks=[],
        reasoning_details=[],
        finish_reason="stop",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        was_stream=False,
        input_truncated=False
    )
    
    exporter.export(call)
    
    spans = phoenix_telemetry.get_finished_spans()
    
    assert len(spans) == 3
    run_span = next(s for s in spans if s.name == "chat")
    step_span = next(s for s in spans if s.name == "chat step")
    gen_span = next(s for s in spans if s.name == "chat generation")
    
    # Check tree structure
    assert gen_span.parent.span_id == step_span.context.span_id
    assert step_span.parent.span_id == run_span.context.span_id
    assert run_span.parent is None
    
    # Check attributes
    assert gen_span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert gen_span.attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-4o"
    assert gen_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
    assert gen_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
    assert gen_span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30
    assert gen_span.attributes[SpanAttributes.OUTPUT_VALUE] == "World"
    assert gen_span.attributes["honcho.trace_id"] == "test-trace"
    assert gen_span.attributes["honcho.workspace_name"] == "test-workspace"

def test_export_tool_call(phoenix_telemetry):
    exporter = PhoenixExporter()
    
    call = CapturedLLMCall(
        trace_id="test-trace",
        span_id="test-span",
        parent_span_id=None,
        iteration=1,
        step_seq=1,
        attempt=1,
        was_fallback=False,
        run_id="test-run",
        workspace_name="test-workspace",
        call_purpose="test-purpose",
        parent_category="dialectic",
        agent_type="assistant",
        session_id="test-session",
        observer=None,
        observed=None,
        peer_name=None,
        track_name="chat",
        transport="openai",
        provider_label=None,
        model="gpt-4o",
        input_messages=[],
        tool_schemas=[],
        tool_choice=None,
        output_content=None,
        output_tool_calls=[{"name": "test_tool", "input": {"arg": "val"}}],
        thinking_content=None,
        thinking_blocks=[],
        reasoning_details=[],
        finish_reason="tool_calls",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        was_stream=False,
        input_truncated=False
    )
    
    exporter.export(call)
    
    spans = phoenix_telemetry.get_finished_spans()
    
    # run span, step span, generation span, tool span (4 spans)
    assert len(spans) == 4
    
    tool_span = next(s for s in spans if s.name == "test_tool")
    assert tool_span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.TOOL.value
    assert tool_span.attributes[SpanAttributes.INPUT_VALUE] == json.dumps({"arg": "val"})

def test_export_disabled(monkeypatch):
    monkeypatch.setattr(settings, "PHOENIX_COLLECTOR_ENDPOINT", None)
    exporter = PhoenixExporter()
    # If the exporter executes the call it will crash since we pass None
    exporter.export(None) # type: ignore
