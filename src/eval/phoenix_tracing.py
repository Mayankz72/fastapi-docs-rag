"""Wire Gemini calls up to a local Phoenix server (run via `docker compose up -d phoenix`).

Phoenix runs as its own container (see docker-compose.yml) — this module only needs
the lightweight OpenTelemetry SDK + OpenInference instrumentation to export traces
to it, avoiding pulling the full `arize-phoenix` package (and its heavy, version-fragile
dependency tree) into the app's own environment.

UI: http://localhost:6006
"""
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

PHOENIX_OTLP_ENDPOINT = "http://localhost:6006/v1/traces"
PROJECT_NAME = "fastapi-rag"

_instrumented = False


def start_tracing() -> None:
    global _instrumented
    if _instrumented:
        return
    resource = Resource.create({ResourceAttributes.PROJECT_NAME: PROJECT_NAME})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_OTLP_ENDPOINT)))
    trace.set_tracer_provider(provider)
    GoogleGenAIInstrumentor().instrument(tracer_provider=provider)
    _instrumented = True
    print(f"Tracing to Phoenix at {PHOENIX_OTLP_ENDPOINT} (UI: http://localhost:6006)")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    start_tracing()
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    result = pipeline.run("How do I define a path parameter with a type?")
    print(result.answer)
    print("\nOpen http://localhost:6006 to see the trace.")
