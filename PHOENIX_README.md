# Arize Phoenix Telemetry in Honcho

Honcho supports native OpenTelemetry integration with [Arize Phoenix](https://phoenix.arize.com/) via the [OpenInference](https://openinference.com/) semantic conventions. This allows you to trace LLM calls, track reasoning steps, and monitor agentic workflows seamlessly.

## Configuration

To enable telemetry tracking and route it to your Phoenix instance, you must configure a few settings.

### Using `.env`
Add the following to your `.env` file:

```env
# 1. Enable Honcho's global telemetry master switch
TELEMETRY_ENABLED=true

# 2. Set the Phoenix OTLP endpoint (must include the /v1/traces path)
PHOENIX_COLLECTOR_ENDPOINT=http://<your-phoenix-host>:4318/v1/traces

# 3. Specify the workspace/project name where the traces will be routed
PHOENIX_PROJECT_NAME=honcho-dev
```

### Using `config.toml`
If you are managing configuration via a `config.toml` file, set it like so:

```toml
[app]
PHOENIX_COLLECTOR_ENDPOINT = "http://<your-phoenix-host>:4318/v1/traces"
PHOENIX_PROJECT_NAME = "honcho-dev"

[telemetry]
ENABLED = true
```

*Note: If the telemetry master switch (`TELEMETRY_ENABLED` or `[telemetry] ENABLED`) is set to `false`, Phoenix configuration will be entirely ignored, and no traces will be emitted.*

### Project Routing & Authentication Headers

By default, the Python exporter automatically sends `PHOENIX_PROJECT_NAME` via the **`x-project-name`** HTTP header. Phoenix's HTTP OTLP collector specifically requires this header (over OpenTelemetry resource attributes) to route traces to the correct project.

If you need to pass additional headers (for example, authentication tokens or API keys to a managed Phoenix instance), you should use the standard OpenTelemetry environment variable `OTEL_EXPORTER_OTLP_HEADERS`.

In your `.env` or `docker-compose.yml`:
```env
# Pass auth tokens or other configuration headers
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer my-secret-token"
```

*Note: Per the OpenTelemetry specification, programmatic headers set in the application code (like our `x-project-name` injection) take strict precedence over `OTEL_EXPORTER_OTLP_HEADERS` during the header merge. This allows DevOps to inject auth freely via the environment variable without accidentally breaking the application's project routing logic.*

---

## Architectural Changes & Breakdown

To support Arize Phoenix natively, we expanded Honcho's `LLMCallExporter` mechanism to project its internal `CapturedLLMCall` event streams onto standard OpenTelemetry spans. The additions respect Honcho's strict requirement that telemetry must be non-blocking and fail safely.

### 1. New Dependencies (`pyproject.toml`)
- **`opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`**: The core SDK and HTTP exporter to ship OTLP spans.
- **`openinference-semantic-conventions`**: Provides the standardized vocabulary (e.g. `LLM_MODEL_NAME`, `INPUT_VALUE`, `OPENINFERENCE_SPAN_KIND`) required to ensure spans render perfectly within the Phoenix UI.

### 2. Configuration Definitions (`src/config.py`, `.env.template`, `config.toml.example`)
- **Added properties**: `PHOENIX_COLLECTOR_ENDPOINT` and `PHOENIX_PROJECT_NAME` inside the core `AppSettings` class.
- **Added logic**: An intelligent `phoenix_exporter_enabled` property determines if the OpenTelemetry provider should be activated at boot.

### 3. Span Context Registry (`src/telemetry/phoenix_session.py`)
- **Purpose**: Honcho handles its LLM streaming decoupled from standard Python context-vars (`contextvars`). Since a single agent run fans out into many distinct calls, we must stitch them together.
- **Implementation**: Much like `langfuse_session.py`, this module acts as a thread-safe registry caching OpenTelemetry `SpanContext` objects. It ensures that the spans for iterations and reasoning steps strictly nest as children under the correct trace roots.

### 4. The Phoenix Exporter (`src/telemetry/phoenix_exporter.py`)
- **Purpose**: The engine connecting Honcho's `CapturedLLMCall` stream to OpenTelemetry.
- **Implementation**: 
  - Subscribes as an `LLMCallExporter` (a sink for captured LLM events).
  - Uses the `NonRecordingSpan` technique to dynamically stitch tree structures back together by explicitly assigning `parent_ctx` without relying on active context scopes.
  - Converts JSON-stringified messages, token counts, model names, and tool-call details into OpenInference attributes.

### 5. Telemetry Lifecycle Wiring (`src/telemetry/events/__init__.py`)
- **Initialization**: If telemetry is enabled and the Phoenix endpoint is provided, the `TracerProvider` is instantiated.
- **Resource Linking**: Attaches `phoenix.project.name` inside an OpenTelemetry `Resource` blob so the Phoenix Collector can correctly categorize the traces into the target project workspace.
- **Teardown**: Binds to Honcho's shutdown event loop to invoke a `force_flush()` on the TracerProvider, guaranteeing no spans are dropped when the server exits.

### 6. Unit Testing (`tests/telemetry/test_phoenix_exporter.py`)
- Validates the span tree architecture via an in-memory exporter, ensuring the semantic conventions correctly attach step/generation spans under root run spans without network calls.
