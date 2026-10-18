"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type {
  ClarifyingQuestion,
  ProjectRecord,
  TaskGraph,
  WorkflowPackageExport,
  WorkflowReviewBundle,
  WorkflowRunRecord,
} from "@/lib/types";
import { ClarificationChat, type ClarificationExchange } from "@/components/clarification-chat";
import { FinalReviewPanel } from "@/components/final-review-panel";
import { PlanPanel } from "@/components/plan-panel";
import { ProjectCreateForm, type ProjectDraft } from "@/components/project-create-form";
import { StatusBadge } from "@/components/status-badge";
import { TaskProgress } from "@/components/task-progress";

const LAST_WORKFLOW_KEY = "projectforge:last-workflow";

const stages = ["Project & Q&A", "Plan", "Build", "Final review", "Package"];

function stepForState(state: WorkflowRunRecord["state"]): number {
  if (["AWAITING_CLARIFICATION", "INTAKE", "SPEC"].includes(state)) return 0;
  if (["PLAN", "AWAITING_PLAN_APPROVAL"].includes(state)) return 1;
  if (["BUILD_LOOP", "INTEGRATION", "REVIEW"].includes(state)) return 2;
  if (state === "AWAITING_FINAL_APPROVAL") return 3;
  if (["PACKAGE", "DONE"].includes(state)) return 4;
  return 0;
}

function pendingQuestions(workflow: WorkflowRunRecord | null): ClarifyingQuestion[] {
  const raw = workflow?.result_payload.pending_questions;
  if (!Array.isArray(raw)) return [];
  return raw.filter((item): item is ClarifyingQuestion => Boolean(
    item && typeof item === "object"
    && typeof (item as ClarifyingQuestion).question_id === "string"
    && typeof (item as ClarifyingQuestion).question === "string"
    && typeof (item as ClarifyingQuestion).why_it_matters === "string",
  ));
}

function packageMetadata(workflow: WorkflowRunRecord | null): WorkflowPackageExport | null {
  const raw = workflow?.result_payload.package_export;
  if (!raw || typeof raw !== "object" || !("filename" in raw) || !("download_url" in raw)) return null;
  return raw as WorkflowPackageExport;
}

function stateMessage(state: WorkflowRunRecord["state"]): string {
  const messages: Partial<Record<WorkflowRunRecord["state"], string>> = {
    INTAKE: "Saving the brief and preparing the Requirement Agent.",
    SPEC: "The Requirement Agent is checking the assignment for missing details.",
    PLAN: "The Planner is preparing a dependency-aware task list.",
    BUILD_LOOP: "Workers are implementing and testing ready tasks in isolated sandboxes.",
    INTEGRATION: "The orchestrator is checking the completed task evidence.",
    REVIEW: "The Reviewer is comparing the result with the rubric.",
    PACKAGE: "The approved project is being documented and packaged.",
    DONE: "The reviewed project package is ready to download.",
    FAILED: "The workflow stopped. Review the failure details below.",
    ESCALATED: "The workflow was paused for human attention.",
  };
  return messages[state] ?? "Waiting for a workflow update.";
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
}

function ErrorNotice({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  if (!message) return null;
  return (
    <div role="alert" className="mb-5 flex items-start justify-between gap-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">
      <span>{message}</span>
      <button type="button" onClick={onDismiss} aria-label="Dismiss error" className="font-semibold text-rose-800 hover:text-rose-950">×</button>
    </div>
  );
}

export function Workbench() {
  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [selectedProject, setSelectedProject] = useState<ProjectRecord | null>(null);
  const [workflow, setWorkflow] = useState<WorkflowRunRecord | null>(null);
  const [taskGraph, setTaskGraph] = useState<TaskGraph | null>(null);
  const [reviewBundle, setReviewBundle] = useState<WorkflowReviewBundle | null>(null);
  const [clarificationHistory, setClarificationHistory] = useState<ClarificationExchange[]>([]);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [loadingProjects, setLoadingProjects] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const questions = useMemo(() => pendingQuestions(workflow), [workflow]);
  const workflowId = workflow?.id ?? null;

  const refreshWorkflow = useCallback(async (id: string, quiet = false) => {
    if (!quiet) setError("");
    try {
      const [latest, projectList] = await Promise.all([
        api.getWorkflow(id),
        Promise.resolve(projects),
      ]);
      setWorkflow(latest);
      const project = projectList.find((item) => item.id === latest.project_id);
      if (project) setSelectedProject(project);
      if (latest.state === "AWAITING_PLAN_APPROVAL" || latest.state === "AWAITING_FINAL_APPROVAL") {
        try {
          setReviewBundle(await api.getReview(id));
        } catch (reviewError) {
          if (!(reviewError instanceof ApiError) || reviewError.status !== 409) throw reviewError;
        }
      } else {
        setReviewBundle(null);
      }
      if (latest.task_graph_id) {
        try {
          const record = await api.getTaskGraph(latest.project_id, latest.task_graph_id);
          setTaskGraph(record.graph);
        } catch (graphError) {
          if (!(graphError instanceof ApiError) || graphError.status !== 404) throw graphError;
        }
      } else {
        setTaskGraph(null);
      }
      setApiConnected(true);
      if (!quiet) setError("");
    } catch (refreshError) {
      setApiConnected(false);
      if (!quiet) setError(refreshError instanceof ApiError ? refreshError.message : "Could not load this workflow.");
    }
  }, [projects]);

  useEffect(() => {
    let active = true;
    async function initialize() {
      try {
        const list = await api.listProjects();
        if (!active) return;
        setProjects(list);
        setApiConnected(true);
        const queryWorkflow = new URLSearchParams(window.location.search).get("workflow");
        const savedWorkflow = queryWorkflow || window.localStorage.getItem(LAST_WORKFLOW_KEY);
        if (savedWorkflow) {
          try {
            const run = await api.getWorkflow(savedWorkflow);
            if (!active) return;
            const project = list.find((item) => item.id === run.project_id);
            if (project) {
              setSelectedProject(project);
              setWorkflow(run);
              window.localStorage.setItem(LAST_WORKFLOW_KEY, run.id);
            }
          } catch {
            window.localStorage.removeItem(LAST_WORKFLOW_KEY);
          }
        }
      } catch (loadError) {
        if (!active) return;
        setApiConnected(false);
        setError(loadError instanceof ApiError
          ? loadError.message
          : "Could not connect to the ProjectForge API. Check the server configuration.");
      } finally {
        if (active) setLoadingProjects(false);
      }
    }
    void initialize();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!workflowId || (workflow && ["DONE", "FAILED", "ESCALATED"].includes(workflow.state))) return;
    let active = true;
    let loading = false;
    const poll = async () => {
      if (!active || loading) return;
      loading = true;
      try {
        await refreshWorkflow(workflowId, true);
      } finally {
        loading = false;
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [workflowId, workflow?.state, refreshWorkflow]);

  async function createProject(draft: ProjectDraft) {
    setBusy(true);
    setError("");
    try {
      const created = await api.createProject({
        name: draft.name,
        raw_brief_text: draft.raw_brief_text,
        ...(draft.rubric_text ? { rubric_text: draft.rubric_text } : {}),
        token_cap: 100_000,
        spend_cap: "10.00",
      });
      setProjects((current) => [created, ...current.filter((item) => item.id !== created.id)]);
      setSelectedProject(created);
      setWorkflow(null);
      setTaskGraph(null);
      setReviewBundle(null);
      setClarificationHistory([]);
    } catch (createError) {
      setError(createError instanceof ApiError ? createError.message : "Could not create the project.");
    } finally {
      setBusy(false);
    }
  }

  async function startWorkflow() {
    if (!selectedProject) return;
    setBusy(true);
    setError("");
    try {
      const run = await api.startWorkflow(selectedProject.id);
      setWorkflow(run);
      window.localStorage.setItem(LAST_WORKFLOW_KEY, run.id);
      window.history.replaceState(null, "", `?workflow=${encodeURIComponent(run.id)}`);
      await refreshWorkflow(run.id, true);
    } catch (startError) {
      setError(startError instanceof ApiError ? startError.message : "Could not start the workflow.");
    } finally {
      setBusy(false);
    }
  }

  function chooseProject(project: ProjectRecord) {
    setSelectedProject(project);
    setWorkflow(null);
    setTaskGraph(null);
    setReviewBundle(null);
    setClarificationHistory([]);
    window.localStorage.removeItem(LAST_WORKFLOW_KEY);
    window.history.replaceState(null, "", window.location.pathname);
    setError("");
  }

  async function submitClarifications(answers: Record<string, string>) {
    if (!workflow) return;
    setBusy(true);
    setError("");
    try {
      const exchanges = questions
        .filter((question) => answers[question.question_id])
        .map((question) => ({ question: question.question, answer: answers[question.question_id] }));
      const next = await api.submitClarifications(workflow.id, answers);
      setClarificationHistory((current) => [...current, ...exchanges]);
      setWorkflow(next);
      await refreshWorkflow(next.id, true);
    } catch (answerError) {
      setError(answerError instanceof ApiError ? answerError.message : "Could not submit those answers.");
    } finally {
      setBusy(false);
    }
  }

  async function decide(stage: "plan" | "final", decision: "approve" | "reject" | "request_changes", comment: string, taskFeedback?: { task_id: string; feedback: string }[]) {
    if (!workflow) return;
    setBusy(true);
    setError("");
    try {
      const next = await api.approve(workflow.id, {
        stage,
        decision,
        comment,
        ...(taskFeedback ? { task_feedback: taskFeedback } : {}),
      });
      setWorkflow(next);
      await refreshWorkflow(next.id, true);
    } catch (decisionError) {
      setError(decisionError instanceof ApiError ? decisionError.message : "Could not submit the decision.");
    } finally {
      setBusy(false);
    }
  }

  const currentStep = workflow ? stepForState(workflow.state) : 0;
  const metadata = packageMetadata(workflow);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 grid-cols-2 gap-1 rounded-lg bg-indigo-700 p-1.5" aria-hidden="true">
              <span className="rounded-sm bg-white" /><span className="rounded-sm bg-indigo-200" />
              <span className="rounded-sm bg-indigo-200" /><span className="rounded-sm bg-white" />
            </div>
            <div>
              <p className="text-sm font-bold tracking-tight text-slate-900">ProjectForge</p>
              <p className="text-xs text-slate-500">Project workspace</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden text-xs text-slate-500 sm:inline">Human approvals · audited builds</span>
            <span className={`inline-flex items-center gap-2 rounded-full px-2.5 py-1 text-xs font-medium ${apiConnected === true ? "bg-emerald-50 text-emerald-800" : apiConnected === false ? "bg-rose-50 text-rose-800" : "bg-slate-100 text-slate-600"}`}>
              <span className={`h-1.5 w-1.5 rounded-full ${apiConnected === true ? "bg-emerald-500" : apiConnected === false ? "bg-rose-500" : "bg-slate-400"}`} />
              {apiConnected === true ? "API connected" : apiConnected === false ? "API unavailable" : "Checking API"}
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1600px] gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[260px_minmax(0,1fr)] lg:px-8">
        <aside className="space-y-4">
          <button
            type="button"
            onClick={() => {
              setSelectedProject(null);
              setWorkflow(null);
              setTaskGraph(null);
              setReviewBundle(null);
              setClarificationHistory([]);
              window.localStorage.removeItem(LAST_WORKFLOW_KEY);
              window.history.replaceState(null, "", window.location.pathname);
              setError("");
            }}
            className="w-full rounded-lg bg-indigo-700 px-3 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-800"
          >
            + New project
          </button>
          <section className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
            <div className="flex items-center justify-between px-1 pb-2">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Recent projects</h2>
              {loadingProjects && <span className="text-xs text-slate-400">Loading…</span>}
            </div>
            <div className="space-y-1">
              {projects.map((project) => (
                <button
                  key={project.id}
                  type="button"
                  onClick={() => chooseProject(project)}
                  className={`block w-full rounded-lg px-3 py-2.5 text-left transition ${selectedProject?.id === project.id ? "bg-indigo-50 text-indigo-950 ring-1 ring-inset ring-indigo-200" : "text-slate-700 hover:bg-slate-50"}`}
                >
                  <span className="block truncate text-sm font-medium">{project.name}</span>
                  <span className="mt-0.5 block text-xs text-slate-500">Updated {formatDate(project.updated_at)}</span>
                </button>
              ))}
              {!loadingProjects && projects.length === 0 && (
                <p className="px-3 py-3 text-xs leading-5 text-slate-500">Projects you create will appear here.</p>
              )}
            </div>
          </section>
          <div className="rounded-xl border border-slate-200 bg-white p-3 text-xs leading-5 text-slate-500">
            ProjectForge pauses at the plan and final-review gates. Code runs only in the configured sandbox.
          </div>
        </aside>

        <main className="min-w-0">
          <ErrorNotice message={error} onDismiss={() => setError("")} />

          {workflow && (
            <div className="mb-5 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold text-slate-900">{selectedProject?.name ?? "Project workflow"}</p>
                  <p className="mt-0.5 text-xs text-slate-500">Workflow {workflow.id.slice(0, 8)} · {stateMessage(workflow.state)}</p>
                </div>
                <StatusBadge status={workflow.state} />
              </div>
              <ol className="grid grid-cols-2 gap-2 sm:grid-cols-5">
                {stages.map((stage, index) => (
                  <li key={stage} className={`rounded-md px-2 py-2 text-center text-xs font-medium ${index === currentStep ? "bg-indigo-700 text-white" : index < currentStep ? "bg-indigo-50 text-indigo-800" : "bg-slate-100 text-slate-500"}`}>
                    <span className="mr-1 opacity-70">{index < currentStep ? "✓" : index + 1}</span>{stage}
                  </li>
                ))}
              </ol>
            </div>
          )}

          {!selectedProject ? (
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
              <ProjectCreateForm busy={busy} onCreate={createProject} />
            </div>
          ) : !workflow ? (
            <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.12em] text-indigo-700">Project</p>
                  <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">{selectedProject.name}</h1>
                  <p className="mt-2 text-sm leading-6 text-slate-600">Start a workflow to normalize the brief, propose a task plan, and begin the gated build process.</p>
                </div>
                <button
                  type="button"
                  disabled={busy || apiConnected === false}
                  onClick={() => void startWorkflow()}
                  className="rounded-lg bg-indigo-700 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  {busy ? "Starting…" : "Start workflow"}
                </button>
              </div>
              <div className="mt-6 grid gap-4 md:grid-cols-2">
                <div className="rounded-lg bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Budget limits</p>
                  <p className="mt-2 text-sm text-slate-800">{selectedProject.budget.token_cap.toLocaleString()} tokens · ${selectedProject.budget.spend_cap} spend cap</p>
                  <p className="mt-1 text-xs text-slate-500">Hard retry limit: 3</p>
                </div>
                <div className="rounded-lg bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Rubric</p>
                  <p className="mt-2 text-sm text-slate-800">{selectedProject.rubric_text ? "Rubric attached" : "No separate rubric file"}</p>
                  <p className="mt-1 text-xs text-slate-500">The Requirement Agent will normalize rubric criteria before planning.</p>
                </div>
              </div>
              {selectedProject.raw_brief_text && (
                <details className="mt-5 rounded-lg border border-slate-200 p-4">
                  <summary className="cursor-pointer text-sm font-medium text-slate-700">View submitted brief</summary>
                  <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap text-sm leading-6 text-slate-600">{selectedProject.raw_brief_text}</pre>
                </details>
              )}
            </section>
          ) : workflow.state === "AWAITING_CLARIFICATION" ? (
            <ClarificationChat
              questions={questions}
              history={clarificationHistory}
              busy={busy}
              onSubmit={submitClarifications}
            />
          ) : workflow.state === "AWAITING_PLAN_APPROVAL" ? (
            reviewBundle ? (
              <PlanPanel
                bundle={reviewBundle}
                busy={busy}
                onDecision={(decision, comment) => decide("plan", decision, comment)}
              />
            ) : <LoadingPanel text="Loading the proposed plan…" />
          ) : workflow.state === "AWAITING_FINAL_APPROVAL" ? (
            reviewBundle ? (
              <FinalReviewPanel
                bundle={reviewBundle}
                busy={busy}
                onApprove={(comment) => decide("final", "approve", comment)}
                onRequestChanges={(comment, taskId, feedback) => decide("final", "request_changes", comment, [{ task_id: taskId, feedback }])}
              />
            ) : <LoadingPanel text="Loading the code diff and rubric review…" />
          ) : workflow.state === "DONE" ? (
            <section className="rounded-xl border border-emerald-200 bg-white p-6 shadow-sm sm:p-8">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-100 text-xl font-bold text-emerald-800" aria-hidden="true">✓</div>
              <p className="mt-5 text-xs font-semibold uppercase tracking-[0.12em] text-emerald-800">Package ready</p>
              <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">Your reviewed project is ready</h1>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">The approved README, project report, AI-usage disclosure, source files, and sandbox test evidence are included in the ZIP bundle.</p>
              {metadata && <p className="mt-4 text-xs text-slate-500">{metadata.filename} · {(metadata.size_bytes / 1024).toFixed(1)} KB · SHA-256 {metadata.sha256.slice(0, 16)}…</p>}
              <a
                href={`/api/workflows/${encodeURIComponent(workflow.id)}/package`}
                download={metadata?.filename ?? undefined}
                className="mt-5 inline-flex items-center gap-2 rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-emerald-800"
              >
                Download project package <span aria-hidden="true">↓</span>
              </a>
            </section>
          ) : workflow.state === "FAILED" || workflow.state === "ESCALATED" ? (
            <section className="rounded-xl border border-rose-200 bg-white p-6 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-rose-800">Workflow attention required</p>
              <h1 className="mt-2 text-2xl font-semibold text-slate-900">{workflow.state === "FAILED" ? "Workflow failed" : "Workflow escalated"}</h1>
              <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-rose-900">{workflow.failure_reason ?? "The workflow requires an administrator to inspect its persisted audit record."}</p>
              <p className="mt-3 text-xs text-slate-500">The workflow is paused. Retry limits and budget caps are not bypassed by the frontend.</p>
            </section>
          ) : taskGraph && ["BUILD_LOOP", "INTEGRATION", "REVIEW", "PACKAGE"].includes(workflow.state) ? (
            <TaskProgress graph={taskGraph} workflowId={workflow.id} />
          ) : (
            <LoadingPanel text={stateMessage(workflow.state)} />
          )}

          {workflow && ["BUILD_LOOP", "INTEGRATION", "REVIEW", "PACKAGE"].includes(workflow.state) && (
            <p className="mt-4 text-center text-xs text-slate-500">Progress refreshes every 2.5 seconds from the orchestrator.</p>
          )}
        </main>
      </div>
    </div>
  );
}

function LoadingPanel({ text }: { text: string }) {
  return (
    <div className="flex min-h-72 items-center justify-center rounded-xl border border-slate-200 bg-white p-6 text-center shadow-sm">
      <div>
        <span className="mx-auto block h-7 w-7 animate-spin rounded-full border-2 border-slate-200 border-t-indigo-700" aria-hidden="true" />
        <p className="mt-4 text-sm font-medium text-slate-800">{text}</p>
        <p className="mt-1 text-xs text-slate-500">The workflow remains controlled by the deterministic orchestrator.</p>
      </div>
    </div>
  );
}
