import type {
  ProjectRecord,
  TaskGraphRecord,
  WorkflowReviewBundle,
  WorkflowRunRecord,
  WorkflowTaskLogRecord,
} from "@/lib/types";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown) {
    super(errorMessage(detail, status));
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function errorMessage(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object" && "detail" in detail) {
    const inner = (detail as { detail?: unknown }).detail;
    if (typeof inner === "string") return inner;
    if (inner && typeof inner === "object" && "message" in inner) {
      return String((inner as { message: unknown }).message);
    }
    if (Array.isArray(inner)) {
      return inner.map((item) => item?.msg ?? JSON.stringify(item)).join("; ");
    }
  }
  return `Request failed (${status})`;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      cache: "no-store",
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError(0, "Could not reach the ProjectForge API proxy.");
  }
  if (!response.ok) {
    const raw = await response.text();
    let detail: unknown = raw;
    try {
      detail = raw ? JSON.parse(raw) : "";
    } catch {
      // Preserve a plain-text error body when the backend does not return JSON.
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

const jsonBody = (value: unknown): Pick<RequestInit, "method" | "body"> => ({
  method: "POST",
  body: JSON.stringify(value),
});

export const api = {
  listProjects: () => requestJson<ProjectRecord[]>("/api/projects?limit=50"),

  createProject: (input: {
    name: string;
    raw_brief_text: string;
    rubric_text?: string;
    token_cap: number;
    spend_cap: string;
  }) => requestJson<ProjectRecord>("/api/projects", jsonBody(input)),

  startWorkflow: (projectId: string) =>
    requestJson<WorkflowRunRecord>(`/api/projects/${encodeURIComponent(projectId)}/workflows`, {
      ...jsonBody({ max_retries: 3, timeout_seconds: 3600 }),
    }),

  getWorkflow: (workflowId: string) =>
    requestJson<WorkflowRunRecord>(`/api/workflows/${encodeURIComponent(workflowId)}`),

  getReview: (workflowId: string) =>
    requestJson<WorkflowReviewBundle>(`/api/workflows/${encodeURIComponent(workflowId)}/review`),

  getTaskGraph: (projectId: string, graphId: string) =>
    requestJson<TaskGraphRecord>(
      `/api/projects/${encodeURIComponent(projectId)}/task-graphs/${encodeURIComponent(graphId)}`,
    ),

  getTaskLogs: (workflowId: string, taskId: string) =>
    requestJson<WorkflowTaskLogRecord[]>(
      `/api/workflows/${encodeURIComponent(workflowId)}/tasks/${encodeURIComponent(taskId)}/logs`,
    ),

  approve: (workflowId: string, input: {
    stage: "plan" | "final";
    decision: "approve" | "reject" | "request_changes";
    comment?: string;
    task_feedback?: { task_id: string; feedback: string }[];
  }) =>
    requestJson<WorkflowRunRecord>(
      `/api/workflows/${encodeURIComponent(workflowId)}/approvals`,
      jsonBody(input),
    ),

  submitClarifications: (workflowId: string, answers: Record<string, string>) =>
    requestJson<WorkflowRunRecord>(
      `/api/workflows/${encodeURIComponent(workflowId)}/clarifications`,
      jsonBody({ answers }),
    ),
};
