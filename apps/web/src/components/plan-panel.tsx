"use client";

import { useState } from "react";
import type { WorkflowReviewBundle } from "@/lib/types";
import { StatusBadge } from "@/components/status-badge";

interface PlanPanelProps {
  bundle: WorkflowReviewBundle;
  busy: boolean;
  onDecision: (decision: "approve" | "request_changes", comment: string) => Promise<void>;
}

export function PlanPanel({ bundle, busy, onDecision }: PlanPanelProps) {
  const [comment, setComment] = useState("");
  const graph = bundle.task_graph;

  return (
    <section className="mx-auto max-w-4xl space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-indigo-700">Human checkpoint · Plan</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">Review the proposed task plan</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
            Review task order and acceptance criteria. Code changes stay gated until the plan is approved.
          </p>
        </div>
        <StatusBadge status={bundle.state} />
      </header>

      {bundle.plan_summary && (
        <div className="rounded-xl border border-indigo-100 bg-indigo-50/70 p-4 text-sm leading-6 text-slate-800">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-indigo-800">Plan summary</p>
          {bundle.plan_summary}
        </div>
      )}

      <div className="space-y-3">
        {(graph?.nodes ?? []).map((node, index) => (
          <article key={node.id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-semibold text-slate-600">{index + 1}</span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-sm font-semibold text-slate-900">{node.id}</h2>
                  <span className="text-xs text-slate-500">{node.assigned_agent}</span>
                </div>
                <p className="mt-1 text-sm leading-6 text-slate-700">{node.description}</p>
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Acceptance criteria</p>
                    <ul className="mt-1 list-disc space-y-1 pl-5 text-sm leading-5 text-slate-700">
                      {node.acceptance_criteria.map((criterion) => <li key={criterion}>{criterion}</li>)}
                    </ul>
                  </div>
                  <div className="space-y-2 text-xs text-slate-600">
                    <p><span className="font-semibold text-slate-700">Depends on:</span> {node.dependencies.length ? node.dependencies.join(", ") : "Nothing"}</p>
                    <p><span className="font-semibold text-slate-700">Writes to:</span> {node.write_set.length ? node.write_set.join(", ") : "Exclusive project workspace"}</p>
                    <p><span className="font-semibold text-slate-700">Runtime:</span> {node.runtime ?? "Inferred from project stack"}</p>
                  </div>
                </div>
              </div>
            </div>
          </article>
        ))}
        {!graph?.nodes.length && <p className="rounded-lg bg-amber-50 p-4 text-sm text-amber-900">The Planner has not returned a task graph.</p>}
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <label className="block space-y-2">
          <span className="text-sm font-medium text-slate-800">Optional note for the Planner</span>
          <textarea
            rows={2}
            maxLength={2000}
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder="If the plan needs work, describe what to change…"
            className="w-full resize-y rounded-lg border border-slate-300 px-3 py-2.5 text-sm leading-6"
          />
        </label>
        <div className="mt-4 flex flex-wrap gap-3">
          <button
            type="button"
            disabled={busy || !graph?.nodes.length}
            onClick={() => void onDecision("approve", comment.trim())}
            className="rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {busy ? "Saving…" : "Approve plan"}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => void onDecision("request_changes", comment.trim())}
            className="rounded-lg border border-amber-300 bg-white px-4 py-2.5 text-sm font-semibold text-amber-900 hover:bg-amber-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Reject plan / request changes
          </button>
        </div>
      </div>
    </section>
  );
}
