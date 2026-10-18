"use client";

import { useMemo, useState } from "react";
import type { WorkflowReviewBundle } from "@/lib/types";
import { StatusBadge } from "@/components/status-badge";

interface FinalReviewPanelProps {
  bundle: WorkflowReviewBundle;
  busy: boolean;
  onApprove: (comment: string) => Promise<void>;
  onRequestChanges: (comment: string, taskId: string, feedback: string) => Promise<void>;
}

const verdictCopy: Record<string, string> = {
  passed: "Pass",
  failed: "Fail",
  not_verified: "Unverified",
  human_review_required: "Needs human judgment",
};

export function FinalReviewPanel({ bundle, busy, onApprove, onRequestChanges }: FinalReviewPanelProps) {
  const completedTasks = useMemo(
    () => (bundle.task_graph?.nodes ?? []).filter((node) => node.status === "completed"),
    [bundle.task_graph],
  );
  const [comment, setComment] = useState("");
  const [taskId, setTaskId] = useState("");
  const [feedback, setFeedback] = useState("");
  const reviewer = bundle.review;

  return (
    <section className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-indigo-700">Human checkpoint · Final review</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">Review the implementation</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Inspect the code diff and each rubric result. Items marked for human judgment remain your decision; they are never counted as automated passes.
          </p>
        </div>
        <StatusBadge status={bundle.state} />
      </header>

      {bundle.report_groundedness && (
        <section className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-emerald-950">Report groundedness</h2>
              <p className="mt-1 text-xs leading-5 text-emerald-900">
                Checked against {bundle.report_groundedness.matched_terms.length} ProjectSpec terms, {bundle.report_groundedness.matched_code_anchors.length} implementation anchors, and {bundle.report_groundedness.matched_test_anchors.length} test-evidence anchors before this checkpoint.
                {bundle.report_groundedness.regeneration_attempts > 0 && ` Regenerated ${bundle.report_groundedness.regeneration_attempts} time${bundle.report_groundedness.regeneration_attempts === 1 ? "" : "s"}.`}
              </p>
              <p className="mt-2 text-xs leading-5 text-emerald-900">
                Matched terms: {bundle.report_groundedness.matched_terms.slice(0, 8).join(", ") || "none"}.
                {bundle.report_groundedness.matched_terms.length > 8 && ` +${bundle.report_groundedness.matched_terms.length - 8} more`}
              </p>
              <p className="mt-1 text-xs leading-5 text-emerald-900">
                Source: {bundle.report_groundedness.matched_code_anchors.slice(0, 3).join(", ") || "none"}; tests: {bundle.report_groundedness.matched_test_anchors.slice(0, 3).join(", ") || "none"}.
              </p>
            </div>
            <StatusBadge status={bundle.report_groundedness.passed ? "passed" : "failed"} />
          </div>
          {bundle.report_groundedness.artifact_paths.length > 0 && (
            <p className="mt-2 text-xs leading-5 text-emerald-900">
              Repository evidence: {bundle.report_groundedness.artifact_paths.slice(0, 5).join(", ")}
              {bundle.report_groundedness.artifact_paths.length > 5 && ` +${bundle.report_groundedness.artifact_paths.length - 5} more files`}
            </p>
          )}
          {bundle.report_groundedness.issues.length > 0 && (
            <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-red-800">
              {bundle.report_groundedness.issues.map((issue) => <li key={issue}>{issue}</li>)}
            </ul>
          )}
          <p className="mt-2 text-xs text-emerald-900">
            The README and report are included in the diff above so you can inspect them before approval.
          </p>
        </section>
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(260px,0.6fr)]">
        <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Code diff</h2>
              <p className="mt-0.5 text-xs text-slate-500">Git diff from the initial project commit to the reviewed build.</p>
            </div>
            <span className="text-xs text-slate-500">{bundle.changed_files.length} changed file{bundle.changed_files.length === 1 ? "" : "s"}</span>
          </div>
          {bundle.changed_files.length > 0 && (
            <div className="flex flex-wrap gap-2 border-b border-slate-100 px-4 py-3">
              {bundle.changed_files.map((file) => <span key={file} className="rounded-md bg-slate-100 px-2 py-1 font-mono text-xs text-slate-700">{file}</span>)}
            </div>
          )}
          {bundle.diff_truncated && <p className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-900">Diff is truncated to the configured review limit.</p>}
          {bundle.code_diff ? (
            <pre className="max-h-[34rem] overflow-auto bg-slate-950 p-4 text-xs leading-5 text-slate-100"><code>{bundle.code_diff}</code></pre>
          ) : (
            <p className="p-5 text-sm text-slate-500">No code diff was returned for this review.</p>
          )}
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-900">Review summary</h2>
          <p className="mt-2 text-sm leading-6 text-slate-700">{reviewer?.summary ?? bundle.reviewer_summary ?? "No structured reviewer summary is available."}</p>
          {reviewer && (
            <div className="mt-4 space-y-2 border-t border-slate-200 pt-4 text-sm">
              <p className="flex items-center justify-between gap-2">
                <span className="text-slate-600">Automatable checks</span>
                <StatusBadge status={reviewer.automatable_checks_passed ? "passed" : "failed"} />
              </p>
              <p className="flex items-center justify-between gap-2">
                <span className="text-slate-600">Human-judgment items</span>
                <span className="font-medium text-violet-800">{reviewer.criteria.filter((item) => item.status === "human_review_required").length} for your review</span>
              </p>
            </div>
          )}
        </section>
      </div>

      <section className="space-y-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Rubric checklist</h2>
          <p className="mt-1 text-sm text-slate-600">Evidence is shown below each criterion, including why an item needs human judgment.</p>
        </div>
        {(reviewer?.criteria ?? []).map((item) => (
          <article key={item.rubric_item.criterion} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-sm font-semibold text-slate-900">{item.rubric_item.criterion}</h3>
                  <span className="rounded-md bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">{item.rubric_item.type}</span>
                </div>
                <p className="mt-1 text-xs leading-5 text-slate-500">Check method: {item.rubric_item.check_method}</p>
              </div>
              <StatusBadge status={item.status} />
            </div>
            <p className="mt-3 text-sm leading-6 text-slate-700">{item.rationale || verdictCopy[item.status]}</p>
            {item.recommendation && <p className="mt-2 text-xs font-medium text-indigo-800">Reviewer recommendation: {item.recommendation.replaceAll("_", " ")}</p>}
            {item.evidence.length > 0 && (
              <ul className="mt-3 list-disc space-y-1 pl-5 text-xs leading-5 text-slate-600">
                {item.evidence.map((evidence, index) => <li key={`${index}-${evidence.slice(0, 30)}`}>{evidence}</li>)}
              </ul>
            )}
          </article>
        ))}
        {!reviewer?.criteria.length && (
          <p className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">No itemized rubric verdict was returned. Do not treat this as a pass.</p>
        )}
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <label className="block space-y-2">
          <span className="text-sm font-medium text-slate-800">Final review comment <span className="font-normal text-slate-500">(optional for approval)</span></span>
          <textarea
            rows={2}
            maxLength={2000}
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder="Record the decision context…"
            className="w-full resize-y rounded-lg border border-slate-300 px-3 py-2.5 text-sm leading-6"
          />
        </label>
        <div className="mt-4 grid gap-3 md:grid-cols-[minmax(180px,0.45fr)_minmax(0,1fr)]">
          <label className="space-y-2">
            <span className="text-sm font-medium text-slate-800">Task to rework</span>
            <select
              value={taskId}
              onChange={(event) => setTaskId(event.target.value)}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm"
            >
              <option value="">Choose a completed task</option>
              {completedTasks.map((task) => <option key={task.id} value={task.id}>{task.id} — {task.description.slice(0, 70)}</option>)}
            </select>
          </label>
          <label className="space-y-2">
            <span className="text-sm font-medium text-slate-800">Specific change request</span>
            <textarea
              rows={2}
              minLength={10}
              maxLength={2000}
              value={feedback}
              onChange={(event) => setFeedback(event.target.value)}
              placeholder="Explain what needs to change for the selected task…"
              className="w-full resize-y rounded-lg border border-slate-300 px-3 py-2.5 text-sm leading-6"
            />
          </label>
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
          <button
            type="button"
            disabled={busy}
            onClick={() => void onApprove(comment.trim())}
            className="rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {busy ? "Saving…" : "Approve and package"}
          </button>
          <button
            type="button"
            disabled={busy || !taskId || feedback.trim().length < 10}
            onClick={() => void onRequestChanges(comment.trim(), taskId, feedback.trim())}
            className="rounded-lg border border-amber-300 bg-white px-4 py-2.5 text-sm font-semibold text-amber-900 hover:bg-amber-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Request changes
          </button>
          <span className="self-center text-xs text-slate-500">Requests are attached to the selected task and its dependent work.</span>
        </div>
      </section>
    </section>
  );
}
