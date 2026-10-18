import type { TaskStatus, WorkflowState } from "@/lib/types";

type BadgeStatus = TaskStatus | WorkflowState | "human_review_required" | "not_verified" | "passed" | "failed";

const styles: Record<string, string> = {
  completed: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  DONE: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  passed: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  running: "bg-blue-50 text-blue-800 ring-blue-200",
  BUILD_LOOP: "bg-blue-50 text-blue-800 ring-blue-200",
  INTAKE: "bg-blue-50 text-blue-800 ring-blue-200",
  SPEC: "bg-blue-50 text-blue-800 ring-blue-200",
  PLAN: "bg-blue-50 text-blue-800 ring-blue-200",
  INTEGRATION: "bg-blue-50 text-blue-800 ring-blue-200",
  REVIEW: "bg-blue-50 text-blue-800 ring-blue-200",
  PACKAGE: "bg-blue-50 text-blue-800 ring-blue-200",
  AWAITING_PLAN_APPROVAL: "bg-amber-50 text-amber-800 ring-amber-200",
  AWAITING_FINAL_APPROVAL: "bg-amber-50 text-amber-800 ring-amber-200",
  AWAITING_CLARIFICATION: "bg-amber-50 text-amber-800 ring-amber-200",
  human_review_required: "bg-violet-50 text-violet-800 ring-violet-200",
  not_verified: "bg-slate-100 text-slate-700 ring-slate-200",
  pending: "bg-slate-100 text-slate-700 ring-slate-200",
  ready: "bg-sky-50 text-sky-800 ring-sky-200",
  failed: "bg-rose-50 text-rose-800 ring-rose-200",
  FAILED: "bg-rose-50 text-rose-800 ring-rose-200",
  ESCALATED: "bg-rose-50 text-rose-800 ring-rose-200",
  cancelled: "bg-slate-100 text-slate-600 ring-slate-200",
};

const labels: Record<string, string> = {
  human_review_required: "Needs human judgment",
  not_verified: "Unverified",
  AWAITING_PLAN_APPROVAL: "Plan approval",
  AWAITING_FINAL_APPROVAL: "Final approval",
  AWAITING_CLARIFICATION: "Needs clarification",
  BUILD_LOOP: "Building",
  DONE: "Complete",
  FAILED: "Failed",
  ESCALATED: "Escalated",
};

export function StatusBadge({ status }: { status: BadgeStatus | string }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold capitalize ring-1 ring-inset ${styles[status] ?? "bg-slate-100 text-slate-700 ring-slate-200"}`}>
      {labels[status] ?? status.replaceAll("_", " ").toLowerCase()}
    </span>
  );
}
