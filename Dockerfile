FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

# Render injects $PORT at runtime and expects the app to bind to it; default
# to 7860 for local/other-host runs where $PORT isn't set. Shell form (not
# exec form) so $PORT actually gets expanded.
EXPOSE 7860
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}
