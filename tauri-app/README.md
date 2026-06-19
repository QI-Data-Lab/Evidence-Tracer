# Evidence Tracer Desktop Client

React + Tauri desktop client for Evidence Tracer. The app connects to the local FastAPI backend, lets you choose and scan a PDF root folder, processes stale documents with streamed progress, and provides a page-aware evidence review surface.

## Setup

```bash
cd tauri-app
npm install
```

Requires Node.js, npm, and the Rust/Tauri toolchain.

## Run

For the full desktop experience:

```bash
cd tauri-app
npm run tauri dev
```

For browser-only UI work:

```bash
cd tauri-app
npm run dev
```

The browser preview is useful for layout work, but native dialogs and file-opening behavior require the Tauri runtime.

## Runtime Expectations

- The backend should be running at `http://127.0.0.1:8000`, unless you set a different backend URL in the app.
- Source PDFs remain in the selected root folder on local disk.
- Backend-generated catalog, previews, and retrieval traces live under `backend/data/`.
- Evidence-agent runs require Ollama to serve the configured model, currently `qwen3:latest` at `http://127.0.0.1:8880`.

## Main UI Surfaces

- Backend URL and root-folder configuration.
- Folder/document tree for scanned PDFs.
- Streamed processing progress for stale or selected documents.
- Page navigation, page preview images, and chunk bounding-box overlays.
- Chunk inspection and source PDF opening.
- Evidence query panel with a configurable run task cap.
- Streaming retrieval activity, accepted evidence, source locators, and final response.

## Commands

```bash
npm run dev
npm run build
npm run preview
npm run tauri dev
```

Use `npm run build` before sharing UI changes; it runs TypeScript checking and produces the Vite bundle.
