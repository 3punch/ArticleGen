export type WorkflowState =
  | "INTAKE"
  | "SPEC"
  | "PLAN"
  | "AWAITING_PLAN_APPROVAL"
  | "AWAITING_CLARIFICATION"
  | "BUILD_LOOP"
  | "INTEGRATION"
  | "REVIEW"
  | "AWAITING_FINAL_APPROVAL"
  | "PACKAGE"
  | "DONE"
  | "FAILED"
  | "ESCALATED";

export type TaskStatus = "pending" | "ready" | "running" | "completed" | "failed" | "cancelled";

export interface ProjectSpec {
  title: string;
  description: string;
  deliverables: string[];
  constraints: string[];
  tech_stack_hints: string[];
  rubric_items: RubricItem[];
}

export interface RubricItem {
  criterion: string;
  type: "automatable" | "human-judgment";
  check_method: string;
}

export interface BudgetLedger {
  project_id: string;
  token_cap: number;
  spend_cap: string;
  used_tokens: number;
  used_spend: string;
  reserved_tokens: number;
  reserved_spend: string;
}

export interface ProjectRecord {
  id: string;
  name: string;
  spec: ProjectSpec | null;
  raw_brief_text: string | null;
  rubric_text: string | null;
  budget: BudgetLedger;
  created_at: string;
  updated_at: string;
}

export interface TaskNode {
  id: string;
  description: string;
  dependencies: string[];
  status: TaskStatus;
  acceptance_criteria: string[];
  retries: number;
  assigned_agent: string;
  read_set: string[];
  write_set: string[];
  runtime: "python" | "node" | null;
  timeout_seconds: number;
  review_feedback: string[];
}

export interface TaskEdge {
  source: string;
  target: string;
}

export interface TaskGraph {
  nodes: TaskNode[];
  edges: TaskEdge[];
}

export interface TaskGraphRecord {
  id: string;
  project_id: string;
  version: number;
  graph: TaskGraph;
  created_at: string;
}

export interface ClarifyingQuestion {
  question_id: string;
  question: string;
  why_it_matters: string;
}

export interface ReviewCriterionAssessment {
  rubric_item: RubricItem;
  status: "passed" | "failed" | "not_verified" | "human_review_required";
  recommendation: "approve" | "request_changes" | "inconclusive" | null;
  rationale: string;
  evidence: string[];
}

export interface ReviewerAgentResponse {
  summary: string;
  criteria: ReviewCriterionAssessment[];
  automatable_checks_passed: boolean;
  human_review_required: boolean;
  evidence: string[];
}

export interface ReportGroundednessResult {
  passed: boolean;
  required_terms: string[];
  matched_terms: string[];
  code_anchors: string[];
  matched_code_anchors: string[];
  test_anchors: string[];
  matched_test_anchors: string[];
  artifact_paths: string[];
  issues: string[];
  regeneration_attempts: number;
}

export interface WorkflowReviewBundle {
  workflow_id: string;
  project_id: string;
  stage: "plan" | "final";
  state: "AWAITING_PLAN_APPROVAL" | "AWAITING_FINAL_APPROVAL";
  plan_summary: string | null;
  task_graph: TaskGraph | null;
  code_diff: string;
  changed_files: string[];
  diff_truncated: boolean;
  review: ReviewerAgentResponse | null;
  reviewer_summary: string | null;
  report_groundedness: ReportGroundednessResult | null;
}

export interface WorkflowRunRecord {
  id: string;
  project_id: string;
  task_graph_id: string | null;
  state: WorkflowState;
  max_retries: number;
  timeout_seconds: number;
  failure_reason: string | null;
  result_payload: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface WorkflowEventRecord {
  workflow_id: string;
  sequence: number;
  from_state: WorkflowState | null;
  to_state: WorkflowState;
  reason: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface WorkflowTaskLogRecord {
  audit_run_id: string;
  workflow_id: string;
  task_id: string;
  agent_name: "coder" | "tester" | "debugger";
  phase: string;
  action: string;
  status: string | null;
  summary: string | null;
  timestamp: string;
  provider: string | null;
  model: string | null;
  tokens_used: number;
  spend: string;
  evidence: string[];
  stdout: string;
  stderr: string;
}

export interface WorkflowPackageExport {
  workflow_id: string;
  project_id: string;
  filename: string;
  download_url: string;
  size_bytes: number;
  sha256: string;
  files: string[];
  created_at: string;
}
