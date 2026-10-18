"use client";

import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { TaskGraph, TaskNode, WorkflowTaskLogRecord } from "@/lib/types";
import { StatusBadge } from "@/components/status-badge";

interface TaskProgressProps {
  graph: TaskGraph;
  workflowId: string;
}

function taskDepth(node: TaskNode, byId: Map<string, TaskNode>, seen = new Set<string>()): number {
  if (seen.has(node.id)) return 0;
  seen.add(node.id);
  const parents = node.dependencies.map((id) => byId.get(id)).filter((item): item is TaskNode => Boolean(item));
  if (!parents.length) return 0;
  return Math.min(3, 1 + Math.max(...parents.map((parent) => taskDepth(parent, byId, new Set(seen)))));
}

function timestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function TaskProgress({ graph, workflowId }: TaskProgressProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [logs, setLogs] = useState<WorkflowTaskLogRecord[]>([]);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [logError, setLogError] = useState("");
  const byId = useMemo(() => new Map(graph.nodes.map((node) => [node.id, node])), [graph.nodes]);
  const selectedTask = selectedId ? byId.get(selectedId) : undefined;

  useEffect(() => {
    if (!selectedId) {
      setLogs([]);
      return;
    }
    let active = true;
    const loadLogs = async () => {
      try {
        const result = await api.getTaskLogs(workflowId, selectedId);
        if (active) {
          setLogs(result);
          setLogError("");
        }
      } catch (error) {
        if (active) setLogError(error instanceof ApiError ? error.message : "Could not load task logs.");
      } finally {
        if (active) setLoadingLogs(false);
      }
    };
    setLoadingLogs(true);
    void loadLogs();
    const timer = window.setInterval(() => void loadLogs(), 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [selectedId, workflowId]);

  const done = graph.nodes.filter((node) => node.status === "completed").length;
  const failed = graph.nodes.filter((node) => node.status === "failed").length;
  const running = graph.nodes.filter((node) => node.status === "running").length;

  return (
    <section className="space-y-5">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-indigo-700">Build progress</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">Tasks and execution logs</h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Task status refreshes automatically. Select a task to view its coder, sandbox-test, and debugger audit records.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 text-sm">
        <span className="rounded-lg border border-slate-200 bg-white px-3 py-2">{done} / {graph.nodes.length} complete</span>
        <span className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-blue-800">{running} running</span>
        {failed > 0 && <span className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-rose-800">{failed} failed</span>}
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(280px,0.8fr)_minmax(0,1.2fr)]">
        <div className="space-y-2">
          {graph.nodes.map((node) => {
            const depth = taskDepth(node, byId);
            const selected = node.id === selectedId;
            return (
              <button
                key={node.id}
                type="button"
                aria-pressed={selected}
                onClick={() => setSelectedId(selected ? null : node.id)}
                className={`block w-full rounded-lg border p-3 text-left transition ${selected ? "border-indigo-500 bg-indigo-50 ring-2 ring-indigo-100" : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"}`}
                style={{ marginLeft: `${depth * 12}px`, width: `calc(100% - ${depth * 12}px)` }}
              >
                <div className="flex items-start justify-between gap-3">
                  <span className="min-w-0">
                    <span className="block text-xs font-semibold uppercase tracking-wide text-slate-500">{node.id}</span>
                    <span className="mt-1 block text-sm font-medium leading-5 text-slate-900">{node.description}</span>
                  </span>
                  <StatusBadge status={node.status} />
                </div>
                <span className="mt-2 block text-xs text-slate-500">
                  {node.dependencies.length ? `Depends on ${node.dependencies.join(", ")}` : "No dependencies"}
                  {node.retries > 0 ? ` · ${node.retries} ${node.retries === 1 ? "retry" : "retries"}` : ""}
                </span>
              </button>
            );
          })}
        </div>

        <div className="min-h-64 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          {!selectedTask ? (
            <div className="flex h-full min-h-56 flex-col items-center justify-center text-center">
              <p className="text-sm font-medium text-slate-700">Select a task to inspect its logs</p>
              <p className="mt-1 max-w-sm text-xs leading-5 text-slate-500">The API returns bounded agent summaries and sandbox output without exposing private model prompts.</p>
            </div>
          ) : (
            <div>
              <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-200 pb-3">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-indigo-700">Task log</p>
                  <h2 className="mt-1 text-lg font-semibold text-slate-900">{selectedTask.id}</h2>
                  <p className="mt-1 text-sm text-slate-600">{selectedTask.description}</p>
                </div>
                <StatusBadge status={selectedTask.status} />
              </div>
              {logError && <p role="alert" className="mt-4 rounded-md bg-rose-50 p-3 text-sm text-rose-800">{logError}</p>}
              {loadingLogs && logs.length === 0 && <p className="mt-4 text-sm text-slate-500">Loading logs…</p>}
              {!loadingLogs && logs.length === 0 && !logError && (
                <p className="mt-4 text-sm text-slate-500">No agent runs recorded for this task yet.</p>
              )}
              <div className="mt-4 space-y-4">
                {logs.map((log) => (
                  <article key={log.audit_run_id} className="rounded-lg border border-slate-200 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold capitalize text-slate-900">{log.agent_name}</span>
                        <span className="text-xs text-slate-500">{log.phase}</span>
                        {log.status && <StatusBadge status={log.status} />}
                      </div>
                      <time className="text-xs text-slate-500">{timestamp(log.timestamp)}</time>
                    </div>
                    {log.summary && <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-700">{log.summary}</p>}
                    {(log.model || log.provider) && (
                      <p className="mt-2 text-xs text-slate-500">Model: {log.model ?? "not recorded"}{log.provider ? ` · ${log.provider}` : ""}</p>
                    )}
                    {log.evidence.length > 0 && (
                      <ul className="mt-3 list-disc space-y-1 pl-5 text-xs leading-5 text-slate-600">
                        {log.evidence.map((item, index) => <li key={`${index}-${item.slice(0, 24)}`}>{item}</li>)}
                      </ul>
                    )}
                    {(log.stdout || log.stderr) && (
                      <details className="mt-3">
                        <summary className="cursor-pointer text-xs font-medium text-indigo-700">Sandbox output</summary>
                        {log.stdout && <pre className="mt-2 max-h-56 overflow-auto rounded-md bg-slate-950 p-3 text-xs leading-5 text-slate-100">{log.stdout}</pre>}
                        {log.stderr && <pre className="mt-2 max-h-56 overflow-auto rounded-md bg-rose-950 p-3 text-xs leading-5 text-rose-50">{log.stderr}</pre>}
                      </details>
                    )}
                  </article>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
