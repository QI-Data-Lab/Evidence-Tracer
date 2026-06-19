# Evidence Tracer Backend

FastAPI service for Evidence Tracer. It discovers local PDFs, converts them with Docling, stores page-aware structure in SQLite, serves preview images for the desktop client, and runs evidence retrieval over processed documents.

The backend is intentionally local-first. Source PDFs stay on disk, generated state lives under `backend/data/`, and retrieval responses keep source locators whenever the pipeline can resolve them. Agentic retrieval also classifies each accepted evidence item as `for`, `against`, or `neutral` for the query.

## Setup

```bash
cd backend
uv sync
```

Requires Python 3.12 or newer.

## Run

```bash
cd backend
uv run uvicorn app.main:app --reload
```

The API then serves on `http://127.0.0.1:8000` by default.

Check the running service with:

```bash
curl http://127.0.0.1:8000/health
```

## Main Endpoints

- `GET /health` - backend health, version, root path, and database path.
- `GET /config` - read the configured root PDF folder.
- `PUT /config/root` - set the active root PDF folder.
- `POST /scan` - recursively scan for PDFs and update the catalog.
- `GET /tree` - return the folder/document tree for the UI.
- `GET /documents` - list cataloged documents.
- `GET /documents/query` - query documents, chunks, and items.
- `POST /documents/process` - process all stale or selected documents synchronously.
- `POST /documents/process/stream` - stream batch progress and final results as newline-delimited JSON.
- `POST /documents/{id}/process` - process one document.
- `GET /documents/{id}/pages` - list page metadata and chunk counts.
- `GET /documents/{id}/pages/{page}` - fetch page-level chunk data and bounding-box overlays.
- `GET /documents/{id}/pages/{page}/image` - render or serve a cached PNG page preview.
- `GET /documents/{id}/file` - serve the source PDF file.
- `GET /evidence/documents/{id}/toc` - return table-of-contents entries derived from section headers.
- `GET /evidence/documents/{id}/sections` - find the best matching section for a query.
- `GET /evidence/search` - run keyword retrieval over processed chunks.
- `POST /evidence/run` - run agentic retrieval and return the final response, including evidence stance and aggregate For/Against/Neutral counts.
- `POST /evidence/run/stream` - stream agentic retrieval events as newline-delimited JSON, including `evidence_added` events with per-evidence stance.

## Storage

The backend stores its local state in `backend/data/`:

- `catalog.sqlite3` - source-of-truth database for documents, pages, items, chunks, and page mappings.
- `page_previews/` - cached rendered page images for the viewer overlay.
- `retrieval_traces/` - readable traces, JSON traces, model inputs/outputs, tool logs, and evaluator stance decisions.
- `retrieval_query_results.jsonl` - append-only query result log with found sources and stance count totals.

Generated data may include excerpts from private documents. Keep `backend/data/` out of commits and treat trace artifacts as sensitive.

## Evidence Agent

Agentic retrieval uses LangGraph and LangChain with a local Ollama model. Defaults live in `app/core/config.py`:

- Ollama base URL: `http://127.0.0.1:8880`
- Ollama model: `qwen3:latest`
- Request default: `max_tasks = 32`
- Safety caps: `agent_max_tasks = 250`, `agent_max_tool_calls = 250`, `agent_max_evidence = 250`, `agent_max_graph_steps = 800`

Prepare the default model with:

```bash
ollama pull qwen3:latest
```

Each run starts from exactly one selected document. The planner and actor inspect sections, chunks, structured items, pages, and resolved citations; the curator accepts traceable evidence; then the evaluator classifies each accepted item as supporting, refuting, or neutral. The API response keeps the raw evidence list and adds an `analysis` object with `for_items`, `against_items`, `neutral_items`, and corresponding chunk counts.

## Observability

Phoenix tracing is enabled by default. To inspect local LangChain/LangGraph/Ollama traces, start Phoenix in a separate terminal:

```bash
uvx --from arize-phoenix phoenix serve
```

Then open the Phoenix UI at `http://127.0.0.1:6006`.

The backend sends traces to `http://127.0.0.1:6006/v1/traces` and uses the telemetry project name `evidence-tracer` by default. Disable tracing while running the backend with:

```bash
PHOENIX_ENABLED=false uv run uvicorn app.main:app --reload
```

## Test

```bash
cd backend
uv run python -m unittest discover app/tests -v
```
