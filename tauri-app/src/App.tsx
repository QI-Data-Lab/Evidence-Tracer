import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactElement,
} from "react";
import "./App.css";

type Status = "idle" | "loading" | "success" | "error";

type DocumentRecord = {
  id: number;
  root_path: string;
  relative_path: string;
  source_path: string;
  file_name: string;
  status: string;
  page_count: number | null;
  chunk_count: number;
  item_count: number;
  error_message: string | null;
  last_processed_at: string | null;
  updated_at: string;
  summary: Record<string, number>;
  needs_processing: boolean;
};

type TreeNode = {
  type: "folder" | "document";
  name: string;
  path: string;
  children?: TreeNode[];
  document_id?: number;
  status?: string;
  page_count?: number | null;
  chunk_count?: number;
};

type PageSummary = {
  page_number: number;
  width: number;
  height: number;
  rotation: number;
  chunk_count: number;
};

type NormalizedBbox = {
  left_pct: number;
  top_pct: number;
  width_pct: number;
  height_pct: number;
};

type ChunkRecord = {
  id: number;
  chunker: string;
  chunk_index: number;
  chunk_type: string;
  text: string;
  contextualized_text: string;
  section_path: string[];
  pages: number[];
  item_refs: string[];
  labels: string[];
  bbox: Record<string, unknown> | null;
  normalized_bbox: NormalizedBbox | null;
};

type PageBundle = {
  document: DocumentRecord;
  page: {
    page_number: number;
    width: number;
    height: number;
    rotation: number;
  };
  chunks: ChunkRecord[];
};

type ScanResult = {
  root_path: string;
  scanned: number;
  new: number;
  stale: number;
  unchanged: number;
  missing: number;
};

type ProcessResult = {
  requested: number;
  processed: DocumentRecord[];
  failed: Array<{
    document_id: number;
    relative_path: string;
    error: string;
  }>;
};

type ProcessDocumentsPayload = {
  document_ids?: number[];
  only_stale?: boolean;
};

type ProcessStreamEvent = {
  type: string;
  total_documents?: number;
  total_pages?: number;
  processed_documents?: number;
  processed_pages?: number;
  page_count?: number;
  document?: DocumentRecord;
  document_id?: number;
  relative_path?: string;
  error?: string;
  result?: ProcessResult;
};

type ProcessingProgress = {
  totalDocuments: number;
  processedDocuments: number;
  totalPages: number;
  processedPages: number;
  currentPath: string | null;
};

type QueryDocument = Pick<DocumentRecord, "id" | "file_name" | "relative_path" | "status" | "page_count">;

type SourceLocator = {
  document_id: number;
  relative_path: string;
  file_name: string;
  page_number: number | null;
  chunk_id: number | null;
  chunk_index: number | null;
  item_id: number | null;
  item_ref?: string | null;
  bbox: Record<string, unknown> | null;
  normalized_bbox: NormalizedBbox | null;
};

type EvidenceRecord = {
  id: string;
  source_key: string;
  source_locator: SourceLocator;
  document: QueryDocument;
  section_path: string[];
  text: string;
  relevance: string;
  stance?: "for" | "against" | "neutral";
  stance_rationale?: string;
  found_by_tasks: string[];
};

type EvidenceAnalysisItem = {
  summary: string;
  source_keys: string[];
  chunk_count: number;
};

type EvidenceAnalysis = {
  for_items: EvidenceAnalysisItem[];
  against_items: EvidenceAnalysisItem[];
  neutral_items: EvidenceAnalysisItem[];
  supporting_chunk_count: number;
  refuting_chunk_count: number;
  neutral_chunk_count: number;
  warning?: string | null;
};

type EvidenceTask = {
  id: string;
  type: string;
  status: string;
  reason: string;
  params: Record<string, unknown>;
  observation_count?: number;
};

type TraceGraph = {
  nodes: Array<Record<string, unknown>>;
  edges: Array<Record<string, unknown>>;
};

type EvidenceRunResponse = {
  run_id: string;
  query: string;
  stop_reason: string;
  tasks: EvidenceTask[];
  evidence: EvidenceRecord[];
  analysis: EvidenceAnalysis;
  trace_graph: TraceGraph;
  artifacts: {
    trace_text_path: string;
    trace_json_path: string;
    tool_results_text_path: string;
    query_log_path: string;
  };
};

type EvidenceStreamEvent = {
  type: string;
  run_id?: string;
  query?: string;
  label?: string;
  node?: string;
  task?: EvidenceTask;
  tasks?: EvidenceTask[];
  tool?: {
    tool_name?: string;
    tool_args?: Record<string, unknown>;
  };
  candidate_count?: number;
  evidence?: EvidenceRecord;
  evidence_count?: number;
  result?: EvidenceRunResponse;
  detail?: string;
};

type QueryActivityKind = "status" | "task" | "evidence" | "complete" | "error";

type QueryActivity = {
  id: string;
  kind: QueryActivityKind;
  label: string;
  node?: string;
  detail?: string;
};

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";
const DEFAULT_SIDEBAR_WIDTH = 360;
const MIN_SIDEBAR_WIDTH = 280;
const MAX_SIDEBAR_WIDTH = 520;
const DEFAULT_QUERY_TASK_BUDGET = 32;
const MIN_QUERY_TASK_BUDGET = 1;
const MAX_QUERY_TASK_BUDGET = 250;
const MIN_CONTENT_WIDTH = 640;
const SHELL_HORIZONTAL_PADDING = 32;
const SIDEBAR_HANDLE_WIDTH = 18;
const SIDEBAR_STACK_BREAKPOINT = 1040;
const SIDEBAR_WIDTH_STORAGE_KEY = "sidebarWidth";
const QUERY_TASK_BUDGET_STORAGE_KEY = "queryTaskBudget";
const QUERY_TASK_BUDGET_HELP =
  "Run-wide total task cap for this query, including planning, citation hops, and follow-ups. Higher values can explore more, but tool-call, evidence, and graph-step safety caps can still end the run first.";
const BBOX_COLORS = [
  "220, 38, 38",
  "22, 163, 74",
  "37, 99, 235",
  "217, 119, 6",
  "8, 145, 178",
  "190, 24, 93",
];
const BBOX_PADDING_PCT = 0.12;

function createPendingEvidenceRunResponse(query: string): EvidenceRunResponse {
  return {
    run_id: "starting",
    query,
    stop_reason: "running",
    tasks: [],
    evidence: [],
    analysis: {
      for_items: [],
      against_items: [],
      neutral_items: [],
      supporting_chunk_count: 0,
      refuting_chunk_count: 0,
      neutral_chunk_count: 0,
      warning: null,
    },
    trace_graph: { nodes: [], edges: [] },
    artifacts: {
      trace_text_path: "",
      trace_json_path: "",
      tool_results_text_path: "",
      query_log_path: "",
    },
  };
}

function mergeTasks(existing: EvidenceTask[], incoming: EvidenceTask[]) {
  const byId = new Map(existing.map((task) => [task.id, task]));
  for (const task of incoming) {
    byId.set(task.id, { ...byId.get(task.id), ...task });
  }
  return Array.from(byId.values());
}

function mergeEvidence(existing: EvidenceRecord[], incoming: EvidenceRecord) {
  const index = existing.findIndex((item) => item.source_key === incoming.source_key);
  if (index === -1) {
    return [...existing, incoming];
  }
  return existing.map((item, itemIndex) => (itemIndex === index ? { ...item, ...incoming } : item));
}

function evidenceLocationLabel(evidence: EvidenceRecord) {
  const parts = [];
  if (evidence.source_locator.page_number !== null) {
    parts.push(`p. ${evidence.source_locator.page_number}`);
  }
  if (evidence.source_locator.chunk_index !== null) {
    parts.push(`#${evidence.source_locator.chunk_index}`);
  } else if (evidence.source_locator.item_id !== null) {
    parts.push(`item ${evidence.source_locator.item_id}`);
  }
  return parts.join(" ") || evidence.source_key;
}

function clampSidebarWidth(width: number, viewportWidth: number) {
  const maxWidthFromViewport = viewportWidth - SHELL_HORIZONTAL_PADDING - SIDEBAR_HANDLE_WIDTH - MIN_CONTENT_WIDTH;
  const boundedMaxWidth = Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, maxWidthFromViewport));
  return Math.min(Math.max(width, MIN_SIDEBAR_WIDTH), boundedMaxWidth);
}

function clampQueryTaskBudget(value: number) {
  return Math.min(Math.max(Math.round(value), MIN_QUERY_TASK_BUDGET), MAX_QUERY_TASK_BUDGET);
}

function App() {
  const [backendUrl, setBackendUrl] = useState<string>(() => localStorage.getItem("backendUrl") ?? DEFAULT_BACKEND_URL);
  const [backendInput, setBackendInput] = useState<string>(() => localStorage.getItem("backendUrl") ?? DEFAULT_BACKEND_URL);
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    const storedWidth = Number(localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY));
    return Number.isFinite(storedWidth) && storedWidth > 0 ? storedWidth : DEFAULT_SIDEBAR_WIDTH;
  });
  const [viewportWidth, setViewportWidth] = useState<number>(() => window.innerWidth);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [rootPathInput, setRootPathInput] = useState("");
  const [backendStatus, setBackendStatus] = useState<Status>("idle");
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [processingProgress, setProcessingProgress] = useState<ProcessingProgress | null>(null);
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [tree, setTree] = useState<TreeNode | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<DocumentRecord | null>(null);
  const [pages, setPages] = useState<PageSummary[]>([]);
  const [selectedPageNumber, setSelectedPageNumber] = useState<number | null>(null);
  const [pageBundle, setPageBundle] = useState<PageBundle | null>(null);
  const [pageImageVersion, setPageImageVersion] = useState(0);
  const [selectedChunkId, setSelectedChunkId] = useState<number | null>(null);
  const [expandedFolders, setExpandedFolders] = useState<Record<string, boolean>>({ "": true });
  const [imageFailed, setImageFailed] = useState(false);
  const [queryText, setQueryText] = useState("");
  const [queryStatus, setQueryStatus] = useState<Status>("idle");
  const [queryResponse, setQueryResponse] = useState<EvidenceRunResponse | null>(null);
  const [queryActivity, setQueryActivity] = useState<QueryActivity | null>(null);
  const [queryTaskBudget, setQueryTaskBudget] = useState<number>(() => {
    const storedBudget = Number(localStorage.getItem(QUERY_TASK_BUDGET_STORAGE_KEY));
    return Number.isFinite(storedBudget) ? clampQueryTaskBudget(storedBudget) : DEFAULT_QUERY_TASK_BUDGET;
  });
  const [pageJumpInput, setPageJumpInput] = useState<string>("");
  const sidebarResizeRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const queryActivityCounterRef = useRef(0);

  const selectedChunk = useMemo(
    () => pageBundle?.chunks.find((chunk) => chunk.id === selectedChunkId) ?? pageBundle?.chunks[0] ?? null,
    [pageBundle, selectedChunkId],
  );
  const canResizeSidebar = viewportWidth > SIDEBAR_STACK_BREAKPOINT;
  const effectiveSidebarWidth = clampSidebarWidth(sidebarWidth, viewportWidth);
  const shellStyle = canResizeSidebar
    ? { gridTemplateColumns: `${effectiveSidebarWidth}px ${SIDEBAR_HANDLE_WIDTH}px minmax(0, 1fr)` }
    : undefined;
  const processingProgressPercent = processingProgress
    ? processingProgress.totalPages > 0
      ? Math.min(100, Math.round((processingProgress.processedPages / processingProgress.totalPages) * 100))
      : processingProgress.totalDocuments > 0
        ? Math.min(100, Math.round((processingProgress.processedDocuments / processingProgress.totalDocuments) * 100))
        : 0
    : 0;
  const processingProgressLabel = processingProgress
    ? processingProgress.totalPages > 0
      ? `Processing ${processingProgress.processedPages} / ${processingProgress.totalPages} pages`
      : processingProgress.totalDocuments > 0
        ? `Processing ${processingProgress.processedDocuments} / ${processingProgress.totalDocuments} documents`
        : "Checking documents"
    : "";

  useEffect(() => {
    setPageJumpInput(selectedPageNumber ? String(selectedPageNumber) : "");
  }, [selectedPageNumber]);

  useEffect(() => {
    localStorage.setItem("backendUrl", backendUrl);
  }, [backendUrl]);

  useEffect(() => {
    localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(effectiveSidebarWidth));
  }, [effectiveSidebarWidth]);

  useEffect(() => {
    localStorage.setItem(QUERY_TASK_BUDGET_STORAGE_KEY, String(queryTaskBudget));
  }, [queryTaskBudget]);

  useEffect(() => {
    const handleResize = () => {
      setViewportWidth(window.innerWidth);
    };

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  useEffect(() => {
    setSidebarWidth((currentWidth) => clampSidebarWidth(currentWidth, viewportWidth));
    if (!canResizeSidebar) {
      sidebarResizeRef.current = null;
      setIsResizingSidebar(false);
    }
  }, [canResizeSidebar, viewportWidth]);

  useEffect(() => {
    if (!isResizingSidebar) {
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      const resizeState = sidebarResizeRef.current;
      if (!resizeState) {
        return;
      }

      const nextWidth = resizeState.startWidth + (event.clientX - resizeState.startX);
      setSidebarWidth(clampSidebarWidth(nextWidth, window.innerWidth));
    };

    const stopResizing = () => {
      sidebarResizeRef.current = null;
      setIsResizingSidebar(false);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResizing);
    window.addEventListener("pointercancel", stopResizing);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResizing);
      window.removeEventListener("pointercancel", stopResizing);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizingSidebar]);

  useEffect(() => {
    void refreshWorkspace();
  }, [backendUrl]);

  function startSidebarResize(event: ReactPointerEvent<HTMLDivElement>) {
    if (!canResizeSidebar) {
      return;
    }

    sidebarResizeRef.current = {
      startX: event.clientX,
      startWidth: effectiveSidebarWidth,
    };
    setIsResizingSidebar(true);
  }

  function handleSidebarResizeKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (!canResizeSidebar) {
      return;
    }

    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setSidebarWidth((currentWidth) => clampSidebarWidth(currentWidth - 24, viewportWidth));
      return;
    }

    if (event.key === "ArrowRight") {
      event.preventDefault();
      setSidebarWidth((currentWidth) => clampSidebarWidth(currentWidth + 24, viewportWidth));
      return;
    }

    if (event.key === "Home") {
      event.preventDefault();
      setSidebarWidth(clampSidebarWidth(MIN_SIDEBAR_WIDTH, viewportWidth));
      return;
    }

    if (event.key === "End") {
      event.preventDefault();
      setSidebarWidth(clampSidebarWidth(MAX_SIDEBAR_WIDTH, viewportWidth));
    }
  }

  async function api<T>(path: string, init?: RequestInit): Promise<T> {
    const method = (init?.method ?? "GET").toUpperCase();
    const requestPath =
      method === "GET"
        ? `${path}${path.includes("?") ? "&" : "?"}_=${Date.now()}`
        : path;

    const response = await fetch(`${backendUrl}${requestPath}`, {
      cache: "no-store",
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
      ...init,
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const payload = (await response.json()) as { detail?: string };
        detail = payload.detail ?? detail;
      } catch {
        detail = await response.text();
      }
      throw new Error(detail || "Request failed");
    }

    return (await response.json()) as T;
  }

  function showQueryActivity(entry: Omit<QueryActivity, "id">) {
    queryActivityCounterRef.current += 1;
    const id = `activity-${queryActivityCounterRef.current}`;
    setQueryActivity({ id, ...entry });
  }

  function applyProcessingStreamEvent(event: ProcessStreamEvent) {
    const currentPath = event.document?.relative_path ?? event.relative_path ?? null;
    setProcessingProgress((current) => ({
      totalDocuments: event.total_documents ?? current?.totalDocuments ?? 0,
      processedDocuments: event.processed_documents ?? current?.processedDocuments ?? 0,
      totalPages: event.total_pages ?? current?.totalPages ?? 0,
      processedPages: event.processed_pages ?? current?.processedPages ?? 0,
      currentPath: event.type === "process_completed" ? null : currentPath ?? current?.currentPath ?? null,
    }));
  }

  async function processDocumentsWithProgress(payload: ProcessDocumentsPayload): Promise<ProcessResult> {
    setProcessingProgress({
      totalDocuments: 0,
      processedDocuments: 0,
      totalPages: 0,
      processedPages: 0,
      currentPath: null,
    });

    const response = await fetch(`${backendUrl}/documents/process/stream`, {
      cache: "no-store",
      headers: {
        "Content-Type": "application/json",
      },
      method: "POST",
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const errorPayload = (await response.json()) as { detail?: string };
        detail = errorPayload.detail ?? detail;
      } catch {
        detail = await response.text();
      }
      throw new Error(detail || "Request failed");
    }

    if (!response.body) {
      throw new Error("Processing stream is not available.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result: ProcessResult | null = null;

    const handleLine = (line: string) => {
      const trimmedLine = line.trim();
      if (!trimmedLine) {
        return;
      }
      const event = JSON.parse(trimmedLine) as ProcessStreamEvent;
      applyProcessingStreamEvent(event);
      if (event.type === "process_completed" && event.result) {
        result = event.result;
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      let newlineIndex = buffer.indexOf("\n");
      while (newlineIndex !== -1) {
        handleLine(buffer.slice(0, newlineIndex));
        buffer = buffer.slice(newlineIndex + 1);
        newlineIndex = buffer.indexOf("\n");
      }
      if (done) {
        break;
      }
    }

    handleLine(buffer);
    if (!result) {
      throw new Error("Processing stream ended before completion.");
    }
    return result;
  }

  async function refreshWorkspace() {
    setBackendStatus("loading");
    setError("");
    setMessage("");

    try {
      const config = await api<{ root_path: string | null }>("/config");
      setRootPathInput(config.root_path ?? "");

      const treeResponse = await api<TreeNode>("/tree");
      setTree(treeResponse);

      const documentsResponse = await api<{ items: DocumentRecord[] }>("/documents");
      setDocuments(documentsResponse.items);
      setBackendStatus("success");

      if (selectedDocumentId !== null) {
        const currentDocument = documentsResponse.items.find((item) => item.id === selectedDocumentId) ?? null;
        if (currentDocument !== null) {
          await loadDocument(currentDocument.id, false);
        } else {
          setSelectedDocumentId(null);
          setSelectedDocument(null);
          setPages([]);
          setSelectedPageNumber(null);
          setPageBundle(null);
          setSelectedChunkId(null);
          setImageFailed(false);
        }
      }
    } catch (err) {
      setBackendStatus("error");
      setError(err instanceof Error ? err.message : "Unable to reach the backend.");
    }
  }

  async function saveBackendUrl() {
    const trimmed = backendInput.trim();
    if (!trimmed) {
      setError("Backend URL cannot be empty.");
      return;
    }
    setBackendUrl(trimmed.replace(/\/$/, ""));
  }

  async function rescanLibrary() {
    const trimmed = rootPathInput.trim();
    if (!trimmed) {
      setError("Root path cannot be empty.");
      return;
    }

    setBusyAction("scan");
    setError("");
    setMessage("");

    try {
      await api<{ root_path: string }>("/config/root", {
        method: "PUT",
        body: JSON.stringify({ root_path: trimmed }),
      });
      const result = await api<ScanResult>("/scan", {
        method: "POST",
        body: JSON.stringify({ root_path: trimmed }),
      });
      const processResult = await processDocumentsWithProgress({ only_stale: true });
      await refreshWorkspace();
      setMessage(
        `Scanned ${result.scanned} PDFs (${result.new} new, ${result.stale} stale). Processed ${processResult.processed.length}, failed ${processResult.failed.length}.`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Rescan failed.");
    } finally {
      setBusyAction(null);
      setProcessingProgress(null);
    }
  }

  async function selectRootFolder() {
    setError("");
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        directory: true,
        multiple: false,
        title: "Select Root PDF Folder",
      });

      if (typeof selected === "string" && selected.length > 0) {
        setRootPathInput(selected);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to open folder picker.");
    }
  }

  async function processDocument(documentId: number) {
    setBusyAction(`process-${documentId}`);
    setError("");
    setMessage("");

    try {
      const refreshed = await api<DocumentRecord>(`/documents/${documentId}/process`, {
        method: "POST",
      });
      setMessage(`Processed ${refreshed.file_name}.`);
      await refreshWorkspace();
      await loadDocument(documentId, true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Document processing failed.");
    } finally {
      setBusyAction(null);
    }
  }

  async function loadDocument(
    documentId: number,
    loadFirstPage: boolean,
    targetPageNumber?: number | null,
    targetChunkId?: number | null,
  ) {
    setSelectedDocumentId(documentId);
    setSelectedChunkId(null);
    setPageBundle(null);
    setImageFailed(false);

    try {
      const document = await api<DocumentRecord>(`/documents/${documentId}`);
      setSelectedDocument(document);

      const pagesResponse = await api<{ document: DocumentRecord; pages: PageSummary[] }>(`/documents/${documentId}/pages`);
      setPages(pagesResponse.pages);

      if (pagesResponse.pages.length > 0) {
        const targetPageExists =
          targetPageNumber !== undefined &&
          targetPageNumber !== null &&
          pagesResponse.pages.some((page) => page.page_number === targetPageNumber);
        const nextPageNumber = targetPageExists
          ? targetPageNumber
          : loadFirstPage
          ? pagesResponse.pages[0].page_number
          : selectedPageNumber && pagesResponse.pages.some((page) => page.page_number === selectedPageNumber)
            ? selectedPageNumber
            : pagesResponse.pages[0].page_number;
        await loadPage(documentId, nextPageNumber, targetChunkId);
      } else {
        setSelectedPageNumber(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load the selected document.");
    }
  }

  async function loadPage(documentId: number, pageNumber: number, targetChunkId?: number | null) {
    setImageFailed(false);
    try {
      const bundle = await api<PageBundle>(`/documents/${documentId}/pages/${pageNumber}`);
      setSelectedPageNumber(pageNumber);
      setPageBundle(bundle);
      setPageImageVersion((current) => current + 1);
      setSelectedChunkId(targetChunkId ?? bundle.chunks[0]?.id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load the page.");
    }
  }

  function applyEvidenceStreamEvent(event: EvidenceStreamEvent, fallbackQuery: string) {
    if (event.type === "run_started") {
      setQueryResponse((current) => ({
        ...(current ?? createPendingEvidenceRunResponse(event.query ?? fallbackQuery)),
        run_id: event.run_id ?? current?.run_id ?? "starting",
        query: event.query ?? current?.query ?? fallbackQuery,
      }));
      return;
    }

    if (event.type === "status") {
      setQueryResponse((current) => {
        const base = current ?? createPendingEvidenceRunResponse(fallbackQuery);
        return {
          ...base,
          run_id: event.run_id ?? base.run_id,
          tasks: event.task ? mergeTasks(base.tasks, [event.task]) : base.tasks,
        };
      });
      showQueryActivity({
        kind: event.node === "actor" ? "task" : "status",
        label: event.label ?? `${event.node ?? "Agent"} working...`,
        node: event.node,
        detail: event.tool?.tool_name,
      });
      return;
    }

    if (event.type === "tasks_planned") {
      setQueryResponse((current) => {
        const base = current ?? createPendingEvidenceRunResponse(fallbackQuery);
        return {
          ...base,
          run_id: event.run_id ?? base.run_id,
          tasks: event.tasks ? mergeTasks(base.tasks, event.tasks) : base.tasks,
        };
      });
      showQueryActivity({ kind: "status", label: event.label ?? "Planner queued tasks.", node: "planner" });
      return;
    }

    if (event.type === "evidence_added" && event.evidence) {
      const evidence = event.evidence;
      setQueryResponse((current) => {
        const base = current ?? createPendingEvidenceRunResponse(fallbackQuery);
        return {
          ...base,
          run_id: event.run_id ?? base.run_id,
          evidence: mergeEvidence(base.evidence, evidence),
        };
      });
      showQueryActivity({
        kind: "evidence",
        label: `Evidence found: ${evidenceLocationLabel(evidence)}`,
        detail: evidence.document.relative_path,
      });
      return;
    }

    if (event.type === "run_completed" && event.result) {
      setQueryResponse(event.result);
      showQueryActivity({
        kind: "complete",
        label: `Retrieval complete: ${event.result.evidence.length} evidence item(s).`,
      });
    }
  }

  async function runQuery() {
    const trimmedQuery = queryText.trim();
    if (!trimmedQuery) {
      setError("Query cannot be empty.");
      return;
    }
    if (queryStatus === "loading") {
      return;
    }
    if (selectedDocumentId === null) {
      setError("Select a document before running retrieval.");
      return;
    }

    setQueryStatus("loading");
    setError("");
    setMessage("");
    setQueryResponse(createPendingEvidenceRunResponse(trimmedQuery));
    setQueryActivity(null);
    showQueryActivity({ kind: "status", label: "Starting retrieval...", node: "planner" });

    try {
      const payload: { query: string; max_tasks: number; document_ids?: number[] } = {
        query: trimmedQuery,
        max_tasks: queryTaskBudget,
        document_ids: [selectedDocumentId],
      };
      const response = await fetch(`${backendUrl}/evidence/run/stream`, {
        cache: "no-store",
        headers: {
          "Content-Type": "application/json",
        },
        method: "POST",
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let detail = response.statusText;
        try {
          const payload = (await response.json()) as { detail?: string };
          detail = payload.detail ?? detail;
        } catch {
          detail = await response.text();
        }
        throw new Error(detail || "Request failed");
      }

      if (!response.body) {
        throw new Error("Streaming response is not available.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;

      const handleLine = (line: string) => {
        const trimmedLine = line.trim();
        if (!trimmedLine) {
          return;
        }
        const event = JSON.parse(trimmedLine) as EvidenceStreamEvent;
        if (event.type === "run_error") {
          showQueryActivity({ kind: "error", label: event.detail ?? "Retrieval failed." });
          throw new Error(event.detail ?? "Query failed.");
        }
        if (event.type === "run_completed") {
          completed = true;
        }
        applyEvidenceStreamEvent(event, trimmedQuery);
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        let newlineIndex = buffer.indexOf("\n");
        while (newlineIndex !== -1) {
          handleLine(buffer.slice(0, newlineIndex));
          buffer = buffer.slice(newlineIndex + 1);
          newlineIndex = buffer.indexOf("\n");
        }
        if (done) {
          break;
        }
      }

      handleLine(buffer);
      if (!completed) {
        throw new Error("Query stream ended before completion.");
      }
      setQueryStatus("success");
    } catch (err) {
      setQueryStatus("error");
      setError(err instanceof Error ? err.message : "Query failed.");
    }
  }

  async function jumpToEvidence(evidence: EvidenceRecord) {
    const locator = evidence.source_locator;
    await loadDocument(locator.document_id, locator.page_number === null, locator.page_number, locator.chunk_id);
  }

  function toggleFolder(path: string) {
    setExpandedFolders((current) => ({
      ...current,
      [path]: !current[path],
    }));
  }

  function renderTreeNode(node: TreeNode, depth = 0): ReactElement {
    if (node.type === "folder") {
      const folderPath = node.path;
      const isExpanded = expandedFolders[folderPath] ?? depth < 2;
      return (
        <div key={`folder-${folderPath}`} className="tree-node">
          <button
            className="tree-row folder-row"
            onClick={() => toggleFolder(folderPath)}
            type="button"
            style={{ paddingLeft: `${depth * 14 + 10}px` }}
          >
            <span className="tree-caret">{isExpanded ? "▾" : "▸"}</span>
            <span className="tree-name">{node.name}</span>
          </button>
          {isExpanded && node.children?.map((child) => renderTreeNode(child, depth + 1))}
        </div>
      );
    }

    const isSelected = node.document_id === selectedDocumentId;
    return (
      <button
        key={`document-${node.document_id ?? node.path}`}
        className={`tree-row document-row ${isSelected ? "selected" : ""}`}
        onClick={() => {
          if (node.document_id !== undefined) {
            void loadDocument(node.document_id, true);
          }
        }}
        type="button"
        style={{ paddingLeft: `${depth * 14 + 28}px` }}
      >
        <span className="tree-name">{node.name}</span>
        <span className={`status-pill status-${node.status ?? "idle"}`}>{node.status ?? "unknown"}</span>
      </button>
    );
  }

  const imageUrl =
    selectedDocumentId !== null && selectedPageNumber !== null
      ? `${backendUrl}/documents/${selectedDocumentId}/pages/${selectedPageNumber}/image?scale=1.6&t=${selectedDocument?.updated_at ?? "0"}&v=${pageImageVersion}`
      : null;
  const supportingEvidence = queryResponse?.evidence.filter((evidence) => evidence.stance === "for") ?? [];
  const refutingEvidence = queryResponse?.evidence.filter((evidence) => evidence.stance === "against") ?? [];
  const neutralEvidence = queryResponse?.evidence.filter((evidence) => evidence.stance !== "for" && evidence.stance !== "against") ?? [];

  function renderEvidenceCard(evidence: EvidenceRecord) {
    return (
      <button
        key={evidence.source_key}
        className="query-result analysis-evidence-card"
        onClick={() => void jumpToEvidence(evidence)}
        type="button"
      >
        <div className="query-result-header">
          <span className="status-pill status-chunk">evidence</span>
          <span>{evidence.document.relative_path}</span>
          <span>{evidence.source_locator.page_number ? `p. ${evidence.source_locator.page_number}` : "no page"}</span>
          <span>{evidence.source_locator.chunk_index !== null ? `#${evidence.source_locator.chunk_index}` : evidence.id}</span>
        </div>
        <strong>{evidence.section_path.join(" / ") || evidence.source_key}</strong>
        {evidence.relevance ? <span className="evidence-relevance">{evidence.relevance}</span> : null}
        {evidence.stance !== "neutral" && evidence.stance_rationale ? (
          <span className="evidence-stance-rationale">{evidence.stance_rationale}</span>
        ) : null}
        <p>{evidence.text.slice(0, 220) || "(empty text)"}</p>
        <div className="linked-documents">
          <span>{evidence.source_key}</span>
          <span>tasks {evidence.found_by_tasks.join(", ")}</span>
        </div>
      </button>
    );
  }

  return (
    <main className={`app-shell ${canResizeSidebar ? "app-shell-resizable" : "app-shell-stacked"} ${isResizingSidebar ? "is-dragging" : ""}`} style={shellStyle}>
      <aside className="sidebar">
        <section className="panel">
          <div className="panel-header">
            <h1>Evidence Tracer</h1>
            <span className={`connection-pill ${backendStatus}`}>{backendStatus}</span>
          </div>

          <label className="field-label" htmlFor="backend-url">
            Backend URL
          </label>
          <div className="inline-form">
            <input
              id="backend-url"
              value={backendInput}
              onChange={(event) => setBackendInput(event.currentTarget.value)}
              placeholder={DEFAULT_BACKEND_URL}
            />
            <button onClick={() => void saveBackendUrl()} type="button">
              Connect
            </button>
          </div>

          <label className="field-label" htmlFor="root-path">
            Root PDF folder
          </label>
          <div className="inline-form">
            <input
              id="root-path"
              value={rootPathInput}
              onChange={(event) => setRootPathInput(event.currentTarget.value)}
              placeholder="/path/to/pdf/root"
            />
            <button aria-label="Select folder" className="icon-button" onClick={() => void selectRootFolder()} type="button">
              <svg aria-hidden="true" className="folder-icon" viewBox="0 0 24 24">
                <path d="M3.75 6.75h5.1l1.55 1.9h9.85v8.6a2 2 0 0 1-2 2H5.75a2 2 0 0 1-2-2z" />
                <path d="M3.75 8.65h16.5" />
              </svg>
            </button>
          </div>

          <div className="action-grid">
            <button disabled={busyAction === "scan"} onClick={() => void rescanLibrary()} type="button">
              Rescan
            </button>
            <button onClick={() => void refreshWorkspace()} type="button">
              Refresh
            </button>
          </div>

          {processingProgress ? (
            <div aria-live="polite" className="processing-progress" role="status">
              <div className="processing-progress-meta">
                <span>{processingProgressLabel}</span>
                <span>{processingProgressPercent}%</span>
              </div>
              <div aria-hidden="true" className="processing-progress-bar">
                <span className="processing-progress-fill" style={{ width: `${processingProgressPercent}%` }} />
              </div>
              {processingProgress.currentPath ? <p className="processing-progress-path">{processingProgress.currentPath}</p> : null}
            </div>
          ) : null}

          {message ? <p className="notice success">{message}</p> : null}
          {error ? <p className="notice error">{error}</p> : null}
        </section>

        <section className="panel tree-panel">
          <div className="panel-header compact">
            <h2>Documents</h2>
            <span>{documents.length}</span>
          </div>

          <div className="tree-scroll">{tree ? renderTreeNode(tree) : <p className="empty">No scanned tree yet.</p>}</div>
        </section>
      </aside>

      <div
        aria-label="Resize document sidebar"
        aria-orientation="vertical"
        aria-valuemax={clampSidebarWidth(MAX_SIDEBAR_WIDTH, viewportWidth)}
        aria-valuemin={MIN_SIDEBAR_WIDTH}
        aria-valuenow={effectiveSidebarWidth}
        className="sidebar-resize-handle"
        onKeyDown={handleSidebarResizeKeyDown}
        onPointerDown={startSidebarResize}
        role="separator"
        tabIndex={canResizeSidebar ? 0 : -1}
      />

      <section className="content">
        <section className="panel document-panel">
          <div className="panel-header">
            <div>
              <h2>{selectedDocument?.file_name ?? "Select a document"}</h2>
              <p className="subtle">{selectedDocument?.relative_path ?? "Choose a scanned PDF from the left."}</p>
            </div>

            {selectedDocument ? (
              <div className="document-actions">
                <span className={`status-pill status-${selectedDocument.status}`}>{selectedDocument.status}</span>
                <button
                  disabled={busyAction === `process-${selectedDocument.id}`}
                  onClick={() => void processDocument(selectedDocument.id)}
                  type="button"
                >
                  Process
                </button>
                <a
                  className="link-button"
                  href={`${backendUrl}/documents/${selectedDocument.id}/file`}
                  rel="noreferrer"
                  target="_blank"
                >
                  Open PDF
                </a>
              </div>
            ) : null}
          </div>

          {selectedDocument ? (
            <div className="document-meta">
              <span>Pages: {selectedDocument.page_count ?? "—"}</span>
              <span>Chunks: {selectedDocument.chunk_count}</span>
              <span>Items: {selectedDocument.item_count}</span>
              <span>Processed: {selectedDocument.last_processed_at ?? "not yet"}</span>
            </div>
          ) : null}

          {selectedDocument?.error_message ? <p className="notice error">{selectedDocument.error_message}</p> : null}

          <section className="query-panel">
            <div className="query-header">
              <div>
                <h3>Evidence Query</h3>
                <p className="subtle">
                  {queryResponse
                    ? `${queryResponse.evidence.length} evidence chunks from ${queryResponse.tasks.length} agent tasks`
                    : "Run the retrieval agent over chunks, sections, and document references."}
                </p>
              </div>
              <span className={`connection-pill ${queryStatus}`}>{queryStatus}</span>
            </div>

            <div className="query-controls">
              <input
                aria-label="Evidence query"
                value={queryText}
                onChange={(event) => setQueryText(event.currentTarget.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    void runQuery();
                  }
                }}
                placeholder="Search sections, figure/table labels, references, or words..."
              />
              <label className="query-budget" title={QUERY_TASK_BUDGET_HELP}>
                <span>Run task cap</span>
                <input
                  aria-label="Run task cap"
                  max={MAX_QUERY_TASK_BUDGET}
                  min={MIN_QUERY_TASK_BUDGET}
                  onChange={(event) => {
                    if (Number.isFinite(event.currentTarget.valueAsNumber)) {
                      setQueryTaskBudget(clampQueryTaskBudget(event.currentTarget.valueAsNumber));
                    }
                  }}
                  step={1}
                  title={QUERY_TASK_BUDGET_HELP}
                  type="number"
                  value={queryTaskBudget}
                />
              </label>
              <label className="query-checkbox">
                <input
                  checked={selectedDocumentId !== null}
                  disabled
                  type="checkbox"
                />
                Current doc
              </label>
              <button disabled={queryStatus === "loading"} onClick={() => void runQuery()} type="button">
                Query
              </button>
            </div>

            {queryActivity ? (
              <div aria-live="polite" className="query-activity">
                <div
                  key={queryActivity.id}
                  className={`activity-row activity-${queryActivity.kind} ${queryStatus === "loading" ? "active" : ""}`}
                >
                  <span aria-hidden="true" className="activity-indicator" />
                  <span className="activity-copy">
                    {queryActivity.node ? <strong>{queryActivity.node}</strong> : null}
                    <span>{queryActivity.label}</span>
                    {queryActivity.detail ? <em>{queryActivity.detail}</em> : null}
                  </span>
                </div>
              </div>
            ) : null}

            {queryResponse ? (
              <div className="coverage-strip">
                <span>Run {queryResponse.run_id}</span>
                <span>Stop {queryResponse.stop_reason}</span>
                <span>Tasks {queryResponse.tasks.length}</span>
                <span>Evidence {queryResponse.evidence.length}</span>
                <span>Edges {queryResponse.trace_graph.edges.length}</span>
              </div>
            ) : null}

            {queryResponse?.artifacts.trace_text_path ? (
              <div className="trace-artifacts">
                <span>Trace: {queryResponse.artifacts.trace_text_path}</span>
                <span>Tool results: {queryResponse.artifacts.tool_results_text_path}</span>
                <span>Query log: {queryResponse.artifacts.query_log_path}</span>
              </div>
            ) : null}

            {queryResponse ? (
              <div className="evidence-analysis">
                <div className="analysis-column analysis-for">
                  <div className="analysis-column-header">
                    <strong>For</strong>
                    <span>{supportingEvidence.length} chunks</span>
                  </div>
                  {supportingEvidence.length === 0 ? (
                    <p>No supporting evidence yet.</p>
                  ) : (
                    <div className="analysis-evidence-list">{supportingEvidence.map(renderEvidenceCard)}</div>
                  )}
                </div>
                <div className="analysis-column analysis-against">
                  <div className="analysis-column-header">
                    <strong>Against</strong>
                    <span>{refutingEvidence.length} chunks</span>
                  </div>
                  {refutingEvidence.length === 0 ? (
                    <p>No refuting evidence yet.</p>
                  ) : (
                    <div className="analysis-evidence-list">{refutingEvidence.map(renderEvidenceCard)}</div>
                  )}
                </div>
                {queryResponse.analysis.warning ? <p className="analysis-warning">{queryResponse.analysis.warning}</p> : null}
              </div>
            ) : null}

            {queryResponse ? (
              <div className="query-results neutral-results">
                {queryResponse.evidence.length === 0 ? (
                  <p className="empty">No evidence chunks were collected for this query.</p>
                ) : neutralEvidence.length === 0 ? null : (
                  <>
                    <div className="neutral-results-header">
                      <strong>Neutral</strong>
                      <span>{neutralEvidence.length} chunks</span>
                    </div>
                    {neutralEvidence.map(renderEvidenceCard)}
                  </>
                )}
              </div>
            ) : null}
          </section>

          {pages.length > 0 ? (
            <div className="page-toolbar">
              <button
                disabled={selectedPageNumber === null || selectedPageNumber <= 1}
                onClick={() => selectedDocumentId && void loadPage(selectedDocumentId, 1)}
                type="button"
              >
                First
              </button>
              <button
                disabled={selectedPageNumber === null || selectedPageNumber <= 1}
                onClick={() => selectedDocumentId && selectedPageNumber && void loadPage(selectedDocumentId, selectedPageNumber - 1)}
                type="button"
              >
                Previous
              </button>

              <label className="page-jump">
                <span>Page</span>
                <input
                  inputMode="numeric"
                  max={pages.length}
                  min={1}
                  onChange={(event) => setPageJumpInput(event.currentTarget.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && selectedDocumentId) {
                      const nextPage = Number(pageJumpInput);
                      if (Number.isInteger(nextPage) && nextPage >= 1 && nextPage <= pages.length) {
                        void loadPage(selectedDocumentId, nextPage);
                      }
                    }
                  }}
                  type="number"
                  value={pageJumpInput}
                />
                <span className="page-total">/ {pages.length}</span>
              </label>

              <button
                disabled={selectedPageNumber === null || selectedPageNumber >= pages.length}
                onClick={() => selectedDocumentId && selectedPageNumber && void loadPage(selectedDocumentId, selectedPageNumber + 1)}
                type="button"
              >
                Next
              </button>
              <button
                disabled={selectedPageNumber === null || selectedPageNumber >= pages.length}
                onClick={() => selectedDocumentId && void loadPage(selectedDocumentId, pages.length)}
                type="button"
              >
                Last
              </button>
            </div>
          ) : selectedDocument ? (
            <p className="empty">Process this document to generate pages and chunks.</p>
          ) : null}

          {pageBundle ? (
            <div className="viewer-layout">
              <div className="viewer-panel">
                <div className="viewer-surface">
                  {imageUrl && !imageFailed ? (
                    <div className="page-stage">
                      <img
                        alt={`Page ${pageBundle.page.page_number}`}
                        className="page-image"
                        onError={() => setImageFailed(true)}
                        src={imageUrl}
                      />
                      <svg className="page-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" role="presentation">
                        {pageBundle.chunks.map((chunk, offset) => {
                          const bbox = chunk.normalized_bbox;
                          if (!bbox) {
                            return null;
                          }

                          const isSelected = chunk.id === selectedChunk?.id;
                          const color = BBOX_COLORS[offset % BBOX_COLORS.length];
                          const x = Math.max(0, bbox.left_pct - BBOX_PADDING_PCT);
                          const y = Math.max(0, bbox.top_pct - BBOX_PADDING_PCT);
                          const right = Math.min(100, bbox.left_pct + bbox.width_pct + BBOX_PADDING_PCT);
                          const bottom = Math.min(100, bbox.top_pct + bbox.height_pct + BBOX_PADDING_PCT);
                          return (
                            <rect
                              key={chunk.id}
                              className="chunk-rect"
                              x={x}
                              y={y}
                              width={Math.max(0, right - x)}
                              height={Math.max(0, bottom - y)}
                              fill={`rgba(${color}, 0.141)`}
                              stroke={`rgb(${color})`}
                              strokeWidth={isSelected ? 5 : 3}
                              vectorEffect="non-scaling-stroke"
                              onClick={() => setSelectedChunkId(chunk.id)}
                            />
                          );
                        })}
                      </svg>
                    </div>
                  ) : (
                    <div className="empty viewer-fallback">
                      <p>Page preview unavailable.</p>
                      <p>Chunk metadata is still loaded in the right panel.</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="detail-panel">
                <div className="chunk-list">
                  <h3>Page Chunks</h3>
                  {pageBundle.chunks.length === 0 ? (
                    <p className="empty">No chunk bboxes were found for this page.</p>
                  ) : (
                    pageBundle.chunks.map((chunk) => (
                      <button
                        key={chunk.id}
                        className={`chunk-list-item ${chunk.id === selectedChunk?.id ? "selected" : ""}`}
                        onClick={() => setSelectedChunkId(chunk.id)}
                        type="button"
                      >
                        <div className="chunk-list-header">
                          <span>#{chunk.chunk_index}</span>
                          <span className={`status-pill status-${chunk.chunk_type}`}>{chunk.chunk_type}</span>
                        </div>
                        <p>{chunk.text.slice(0, 180) || "(empty chunk text)"}</p>
                      </button>
                    ))
                  )}
                </div>

                <div className="chunk-detail">
                  <h3>Chunk Detail</h3>
                  {selectedChunk ? (
                    <>
                      <div className="detail-grid">
                        <span>Chunker</span>
                        <strong>{selectedChunk.chunker}</strong>
                        <span>Section</span>
                        <strong>{selectedChunk.section_path.join(" / ") || "—"}</strong>
                        <span>Items</span>
                        <strong>{selectedChunk.item_refs.length}</strong>
                        <span>Labels</span>
                        <strong>{selectedChunk.labels.join(", ") || "—"}</strong>
                      </div>
                      <pre className="chunk-text">{selectedChunk.text || "(empty chunk text)"}</pre>
                    </>
                  ) : (
                    <p className="empty">Select a chunk to inspect its text.</p>
                  )}
                </div>
              </div>
            </div>
          ) : null}
        </section>
      </section>
    </main>
  );
}

export default App;
