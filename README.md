# ArticleGen

An automated, **iterative multi-agent** pipeline that turns a structured brief into a
reviewed article or code deliverable. Five specialised agents run in a fixed order
inside a bounded loop; a decision gate rejects bad input before any agent spends a
token; and every iteration appends one block to an append-only `change_log.md` instead
of replaying history into the context window.

Built for **free-tier efficiency**: the package has zero runtime dependencies, every
context packet is measured against a hard token cap before use, and prior state is
compressed rather than retained.

---

## Architecture

```
                     ┌─────────────────────────────┐
   brief.json ─────► │  1. INPUT VERIFICATION GATE │──invalid──► ROUTE_BACK
                     └──────────────┬──────────────┘        (0 tokens spent)
                                    │ valid
                     ┌──────────────▼──────────────┐
                     │   2. MASTER ORCHESTRATOR    │
                     │  task breakdown · ordering  │
                     │  change_log.md · stop rules │
                     └──────────────┬──────────────┘
              bounded context packet│(measured, evicted by priority)
        ┌───────────┬───────────┬───┴───────┬────────────┬─────────────┐
        ▼           ▼           ▼           ▼            ▼             │
   1 Writer     2 Tester    3 Reviewer  4 Architect  5 Synthesizer     │
   content      correctness readability structure    merge → new draft │
        └───────────┴───────────┴───────────┴────────────┘             │
                                    │                                  │
                     ┌──────────────▼──────────────┐                   │
                     │  score → converge? ──no──────┼───────────────────┘
                     └──────────────┬──────────────┘
                                    │ yes / stop
                          master_draft.md + change_log.md
```

| Agent | Role | Owns | Prompt |
| ----- | ---- | ---- | ------ |
| 1 | Writer / Developer | prose and code blocks | `prompts/01_writer.md` |
| 2 | Tester / Validator | facts, citations, code accuracy, consistency | `prompts/02_tester.md` |
| 3 | Reviewer / Critic | tone, readability, flow, style | `prompts/03_reviewer.md` |
| 4 | Architect / Structure | section organisation, outline alignment | `prompts/04_architect.md` |
| 5 | Synthesizer / Refiner | merges feedback into the new master draft | `prompts/05_synthesizer.md` |

Execution order is fixed and non-negotiable (rule 4.2 in `claude.md`):
**Writer → Tester → Reviewer → Architect → Synthesizer.**

---

## Quick start

```bash
# No installation needed - standard library only.
python3 -m articlegen verify examples/brief_example.json   # gate only, zero tokens
python3 -m articlegen demo                                 # full loop on the example brief
python3 -m articlegen run path/to/brief.json --iterations 4
python3 -m articlegen log                                  # summarise change_log.jsonl
```

Or with the console script after `pip install -e .`:

```bash
articlegen run path/to/brief.json --target 0.9 --provider auto
```

With `pytest` available:

```bash
python3 -m pytest tests/ -q
make test && make demo
```

---

## Web dashboard

A Streamlit GUI wraps the same `Orchestrator` the CLI uses — no separate code path,
so the dashboard and `articlegen run` always behave identically.

```bash
pip install -e '.[ui]'        # installs streamlit + pandas (core stays dependency-free)
articlegen ui                 # http://localhost:8501
# or directly:
streamlit run app.py     # thin shim; the script lives at articlegen/ui/dashboard.py
```

Useful flags:

```bash
articlegen ui --port 8600 --address 0.0.0.0
articlegen ui --no-cors       # when served behind a reverse proxy that rewrites Origin
```

### Tabs

| Tab | What it does |
| --- | ------------ |
| **📋 Brief** | Form for task id, title, objective/requirements, audience, tone, mode, target words, constraints, citations, and a `id \| Title \| purpose \| word_target` sections editor. Upload a brief JSON or load the bundled example. **Validate brief** runs the decision gate and shows every issue with its required fix; **Run** is disabled until validation passes. |
| **⚡ Execution** | Live tracker. Five cards in mandatory order (Writer → Tester → Reviewer → Architect → Synthesizer) with status icons ⚪ pending / 🔵 running / 🟢 done / 🟠 degraded, plus a per-agent table of findings, blockers, context tokens, in/out tokens and engine, and the full event timeline. |
| **📈 Scores** | Score progression line against the target, a per-component breakdown (correctness / readability / structure / completeness), and a per-iteration metrics table. |
| **📄 Draft** | Rendered Markdown of the master draft, the raw source, word/score/iteration metrics, **⬇️ Download Article (.md)** and a JSON run bundle. |
| **📜 Change Log** | `change_log.md` as written by the pipeline, plus the parsed JSONL mirror as a sortable table. |

Run settings (max iterations, target score, context hard cap, provider, workspace,
change-log path) live in the sidebar.

### How the live updates work

The orchestrator gained an optional `EventBus`. During a run it emits
`RUN_START`, `GATE_PASS`/`GATE_ROUTE_BACK`, `ITERATION_START`, `AGENT_START`,
`AGENT_DONE`, `DRAFT_UPDATED`, `ITERATION_DONE` and `RUN_END`. The dashboard
subscribes a `TrackerState` to that bus and redraws pre-allocated Streamlit
placeholders from the callback, so progress appears while the loop is still working
rather than all at once when it finishes.

The bus isolates subscribers: a callback that raises is recorded in `bus.errors` and
cannot abort a pipeline run.

Nothing in `articlegen/ui/state.py`, `builder.py` or `events.py` imports Streamlit, so
all of that logic is covered by ordinary unit tests; only `render.py` touches
Streamlit, and `articlegen/ui/dashboard.py` is exercised through Streamlit's own `AppTest` harness.

The script ships inside the package (not as a repo-root file) so `articlegen ui` works after a plain `pip install`; the root `app.py` is a `runpy` shim that keeps `streamlit run app.py` working from a checkout.

---

### Brief format

```json
{
  "task_id": "my-article",
  "title": "…",
  "objective": "at least 40 characters; state the outcome the reader gets",
  "audience": "who this is for",
  "mode": "article",
  "tone": "neutral, precise, professional",
  "target_words": 1000,
  "requires_citations": false,
  "constraints": ["include at least one worked example"],
  "sections": [
    {"id": "introduction", "title": "Introduction",
     "purpose": "what this section must accomplish", "word_target": 200}
  ]
}
```

`articlegen init brief.json` writes a starter JSON template with explicit `TODO`
markers. The input gate routes it back until you replace those placeholders with
article-specific content; run `articlegen verify brief.json` before generation.

---

## Token & memory management

This is the part that keeps a long loop from degrading. It is enforced in code, not by
convention, and tuned for minimum spend:

| Control | Default | Where |
| ------- | ------- | ----- |
| Context hard cap per agent call | 4800 tokens | `TokenBudget.context_hard_cap` |
| Soft warning threshold | 3800 tokens | `TokenBudget.context_soft_warn` |
| Per-agent input/output ceilings | 1800–2400 / 700–2200 | `TokenBudget.agents` |
| Whole-run token ceiling | 42000 tokens | `TokenBudget.run_token_ceiling` |
| Open findings retained | 40 max | `MemoryStore.absorb_findings` |
| Rolling history retained | 8 entries / 450 tokens | `RollingMemory` |

### Token-saving mechanisms

Three levers keep the run's spend at the floor without touching quality:

* **Idle agents never run.** When the Writer has no section under target and no open
  finding inside its remit, `BaseAgent.should_skip` drops the provider call entirely —
  the report records `engine=skipped`, zero tokens, and the dashboard shows the agent
  as ⚫ skipped. On a converged draft this zeroes one of five calls per iteration.
* **The Writer receives a delta packet.** After the first iteration it sees full text
  only for sections it is allowed to change (under target or named in an open
  finding); untouched sections are reduced to one `(unchanged, N words)` attestation.
  Its context falls from O(draft) to O(needed work).
* **Repairs converge.** Reviewer findings that land inside the Writer's remit are
  fixed mechanically (filler removal, long-sentence/paragraph splitting,
  contraction expansion, word dedup) instead of re-generating the same scaffold —
  so a finding resolves the iteration after it appears rather than pacing the loop.

On the bundled demo these levers turn a 27,050-token estimate into ~23,000, and the
converging iteration drops roughly a quarter — before the budget caps that
hard-limit real provider calls are accounted for.

**Priority eviction.** A context packet is assembled from prioritised blocks; when the
cap would be breached, low-priority blocks are dropped and the eviction is written to
the change log:

```
1 system_rules_digest  (required - truncated, never dropped)
2 task_brief           (required)
3 master_draft         (required for Writer/Synthesizer)
4 open_findings        (evictable)
5 rolling_history      (evictable)
```

**No history replay.** Agents never see prior transcripts. Each iteration contributes
one compressed line to a rolling digest, and the full detail lives in `change_log.md`.
Memory cost is therefore O(budget), not O(iterations).

**The rules digest, not the constitution.** Agents receive only the fenced `Agent
Digest` block at the bottom of `claude.md` (~150 tokens), not the whole document —
because it is re-sent on every one of the five agent calls, every iteration.

---

## Change log

`change_log.md` is append-only and locked during writes (`fcntl`), with a JSONL mirror
for tooling. Each iteration records the gate result, per-agent status and engine, score
movement, findings applied/deferred, token spend, and any context eviction:

```markdown
## [2026-10-11T16:43:51Z] task=demo-article · iteration 2 · decision=CONTINUE

- gate: PASS (5 minor) at draft_verification
- agents: writer=ok/0f/offline-heuristic · tester=ok/0f/offline-heuristic · …
- score: 0.7241 → **0.9169** (Δ+0.1928)
- findings: 6 raised · 1 applied · 5 deferred
- context: cap=6000 evicted=none
- tokens: 15341 this iteration
```

---

## Convergence and stopping

The loop stops on the first of these, and always records why:

1. `score >= score_target` (default **0.85**)
2. Improvement below `min_delta` for `stall_window` consecutive iterations
3. Token ceiling reached
4. `max_iterations` reached

A run finishing below target is reported as `NEEDS_HUMAN`, not `PASS`, and the CLI
exits non-zero. Failures are loud.

Scores are computed from counted quantities — finding weights, outline coverage,
word-count deviation, measured readability — so runs are reproducible and a score
movement can be traced to a specific finding.

---

## Repository layout

```
articlegen/
├── gate.py            # Stage 1: input verification / decision gate
├── orchestrator.py    # Stage 2: master controller and loop
├── agents/            # Stage 3: the five specialised sub-agents
├── tokens.py          # Stage 4: token accounting and hard caps
├── memory.py          # Stage 4: context packets, eviction, rolling summary
├── changelog.py       # Stage 2: append-only change log
├── scoring.py         # convergence decision
├── llm.py             # provider abstraction + JSON parsing
├── prompts.py         # template loading and rules-digest extraction
├── events.py          # progress events for the dashboard
├── ui/                # dashboard: state, builder, render, dashboard.py (script)
└── cli.py             # verify | run | log | ui | init | demo
app.py                 # runpy shim -> articlegen/ui/dashboard.py
prompts/               # the five agent prompt templates
claude.md              # system rules (constitution) + Agent Digest
config/default.json    # budgets, thresholds, iteration limits
examples/              # a valid brief and an invalid one
tests/                 # gate, agents, memory, loop, CLI and dashboard coverage
```

---

## Engine modes

| Mode | Trigger | Behaviour |
| ---- | ------- | --------- |
| `anthropic` | `ANTHROPIC_API_KEY` set (or `--provider anthropic`) | real model calls via `urllib` |
| `openai` | `OPENAI_API_KEY` set (or `--provider openai`) | real model calls via `urllib` |
| `offline-heuristic` | no key (default) | deterministic rule-based engines |

The offline engine is **not** a model. It runs genuine checks — Python blocks are
actually compiled, anchors are actually resolved, metric contradictions are actually
detected — and its Writer drafts role-composed professional prose from the brief:
paragraphs are built from framing → audience → method → integrity → risk → payoff
sentence pools with a no-repeat guarantee per section, distinct section openings, a
closing takeaway, and a brief-objective anchor quoted in the first drafted section.
It exists so the loop, the budgets, the scoring and the convergence logic are
testable and runnable without credentials. Every report and change-log line names
the engine that produced it.

**Verification status:** the offline path is covered by the test suite and a full
end-to-end run. The two network clients are implemented but were not exercised against
the live APIs.

---

## CI

`.github/workflows/pipeline.yml` runs the test suite, verifies that the shipped valid
brief passes the gate and the shipped invalid brief is routed back, confirms all prompt
templates and the rules digest load, then executes the orchestrated loop and uploads
`runs/` as an artefact. `workflow_dispatch` accepts a brief path, iteration count and
provider; add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` as a repository secret to run
against a real model.

---

## ProjectForge — Phases 1–5

Phase 1 established shared Pydantic contracts, PostgreSQL persistence, Alembic
migrations, and the FastAPI CRUD surface. Phase 2 added a durable Postgres-backed FSM,
RQ/Redis dispatch, event/audit records, persisted budgets, task retries/timeouts,
resource-aware DAG scheduling, and the two mandatory human approval gates. Phase 3
replaces only the Requirement and Planner stubs with LangGraph-wrapped model calls
routed through optional LiteLLM: each agent uses a single linear model-call subgraph,
with strict schema validation in deterministic Python and no LLM-controlled branching.
Requirement returns a validated `ProjectSpec` or pauses for explicit clarification,
and Planner emits a validated, acyclic `TaskGraph` that is persisted before the
plan-approval gate. Phase 4 established bounded Docker execution. Phase 5 replaces
Coder, Tester, and Debugger stubs in `BUILD_LOOP`: task-scoped code changes and generated
tests are committed to a per-project Git repository, tests run in the isolated Python
or Node sandbox, and bounded Debugger guidance is fed into each Coder retry. Normalized
ProjectSpec facts are chunked and retrieved from PostgreSQL pgvector, so Coder receives
only the top relevant excerpts rather than the entire specification. Orchestration and
control flow remain deterministic code; no LLM chooses state transitions. Integration,
review, and packaging remain stubs. The workflow engine is
exposed through a swappable `WorkflowEngine` interface. Production tables are not
created on API startup; apply the Alembic migrations explicitly. The API defaults to
loopback because authentication is not configured and must not be exposed publicly
without an authenticated deployment.

### Install and start PostgreSQL locally

```bash
python -m pip install -e '.[dev,ui,projectforge-api,projectforge-worker,projectforge-agents,projectforge-sandbox]'
export POSTGRES_PASSWORD='choose-a-local-password'
docker compose -f infra/docker-compose.yml up -d postgres redis
export PROJECTFORGE_DATABASE_URL="postgresql+psycopg://projectforge:${POSTGRES_PASSWORD}@127.0.0.1:5432/projectforge"
export PROJECTFORGE_REDIS_URL="redis://127.0.0.1:6379/0"
alembic -c infra/alembic.ini upgrade head
projectforge-api
# In another terminal, with the same environment:
projectforge-worker
```

### Model-backed agent configuration

The API and RQ worker use the same environment-driven `ModelConfig`; the shared tier
is `strong` by default, and Requirement, Planner, Coder, Tester, and Debugger use it
unless an agent-specific model override is set. Dahl is the initial provider. Its
model catalog rotates, so fetch a current exact model ID from the `/v1/models` endpoint
rather than relying on a hard-coded example:

```bash
export PROJECTFORGE_LLM_PROVIDER=dahl
export PROJECTFORGE_LLM_TIER=strong
export PROJECTFORGE_DAHL_STRONG_MODEL='<current model id>'
export DAHL_API_KEY='<Dahl API key>'
# Optional: defaults to https://inference.dahl.global/v1
export PROJECTFORGE_LLM_BASE_URL='https://inference.dahl.global/v1'
```

`PROJECTFORGE_LLM_MODEL` can set the shared exact model ID directly. These optional
variables override it per agent: `PROJECTFORGE_REQUIREMENT_MODEL`,
`PROJECTFORGE_PLANNER_MODEL`, `PROJECTFORGE_CODER_MODEL`, `PROJECTFORGE_TESTER_MODEL`,
and `PROJECTFORGE_DEBUGGER_MODEL`. `PROJECTFORGE_LLM_ROUTE_PREFIX` defaults to
LiteLLM's `openai` adapter for Dahl's OpenAI-compatible endpoint. Model responses are
locally validated against strict Pydantic contracts; structured provider output is not
assumed.

Before each model call, ProjectForge atomically reserves budget in PostgreSQL. The
per-call token ceiling defaults to `PROJECTFORGE_LLM_RESERVATION_TOKENS=60000`; each
hold is sized from a conservative, network-free UTF-8 prompt-byte upper bound plus
maximum output, capped at that ceiling. The per-call spend ceiling defaults to
`PROJECTFORGE_LLM_RESERVATION_SPEND=0.50` USD with a 900-second reservation TTL;
`PROJECTFORGE_LLM_MAX_OUTPUT_TOKENS=4096` and
`PROJECTFORGE_LLM_TIMEOUT_SECONDS=60` bound each request. Tune the spend ceiling to at
least the provider's maximum possible charge for the configured model and token limit,
while keeping it within each project's spend cap. When provider usage or cost is
missing, the audit ledger conservatively settles the full per-call hold. A response
exceeding the hold is still recorded and immediately escalates the workflow.
`PROJECTFORGE_REQUIREMENT_CONFIDENCE=0.70` is the default threshold below which
Requirement must ask the user rather than persist a guessed specification.

Override `PROJECTFORGE_HOST` only in a protected deployment. Do not expose this
unauthenticated API or its human-approval routes to a public network. Database and
Redis connection settings are read from environment variables, not source control.

### Phase 4 — isolated code execution

Install the optional Docker SDK and build the preconfigured runner/proxy images from
the repository root:

```bash
python -m pip install -e '.[projectforge-sandbox]'
docker build -f infra/sandbox/python/Dockerfile -t projectforge/python-test:3.11 .
docker build -f infra/sandbox/node/Dockerfile -t projectforge/node-test:20 .
docker build -f infra/sandbox/egress-proxy/Dockerfile -t projectforge/sandbox-egress-proxy:3.11 .
```

`projectforge_sandbox.SandboxManager.run()` accepts a Python or Node runtime, a shell
string or argument vector, and either a relative-path-to-content mapping or a project
worktree. It copies the input through Docker's archive API (never a host bind mount),
returns separate stdout/stderr, exit code, timeout status, duration, and container ID,
and always removes the temporary container in `finally`. `reap_orphans()` is available
for worker-startup cleanup after a process crash and only removes labeled resources
older than the maximum lease plus a five-minute safety margin. `project_dir` excludes VCS data,
virtual environments, caches, common secret files/directories, and rejects symlinks;
curated `project_files` mappings are the preferred input when the worktree may contain
credentials.

The defaults are one CPU core, 512 MiB memory with swap disabled, 60 seconds, 128
processes, a 100 MiB input-workspace ceiling, and bounded output. Per-run limits can be
lowered or raised only within hard caps (4 CPU cores, 4 GiB, and one hour). Containers
run as an unprivileged UID with a read-only root filesystem, ephemeral workspace/tmp
mounts, dropped Linux capabilities, `no-new-privileges`, and no Docker socket or host
mounts. Python images include pytest 8.3.5; Node images include Jest 29.7.0. Pin image
digests in production deployments rather than relying on mutable tags. The images
contain only the test runners; project dependencies must be baked into an approved
image or staged with the project (network access is not needed for ordinary runs). Run
the manager only on a dedicated worker host or isolated/rootless Docker daemon: access
to the manager's Docker socket is privileged, and that socket is never mounted into a
sandbox.

Network access is `none` by default. For an explicit `network_allowlist`, the sandbox
is attached only to an internal Docker network and HTTP(S) requests must pass through
a per-run egress proxy attached to a separate outbound bridge. Exact host rules and
`*.example.org` subdomain rules are supported; host rules allow ports 80/443, while a
rule with an explicit port allows only that port. The proxy pins DNS results to public
IP addresses and rejects private/reserved destinations. Non-proxy-aware clients and
direct socket attempts remain blocked; allowlisted egress is not a general network
bridge.

Run Docker-backed acceptance tests after building the images:

```bash
python -m pytest tests/test_projectforge_sandbox.py -m docker -q
```

Those tests run passing and failing pytest files, kill an infinite loop at its deadline,
verify default egress is blocked, and smoke-test the Node/Jest image. They skip when the
Docker SDK/daemon or built images are absent; CI installs the optional extra, builds the
images, and runs them. This workspace has no Docker Engine, so live container and
network-isolation tests have not run here.

### Phase 5 — task-scoped build agents

The production suite now uses real Coder, Tester, and Debugger agents inside the
Postgres-backed `BUILD_LOOP`. Coder receives one persisted `TaskNode`, bounded relevant
repository files, Debugger feedback from the previous failed attempt (when present),
and at most three ProjectSpec excerpts selected through pgvector cosine similarity;
that bound guarantees the full specification is never sent to Coder.
Spec chunks use a deterministic 384-dimensional local feature embedding, avoiding a
second provider call and extra runtime dependency. The HNSW pgvector index and task
runtime metadata are added by migration `0004_build_loop_context`.

Set `PROJECTFORGE_REPO_ROOT` to change the default `/data/repos` workspace root. Each
project gets its own Git repository at `{root}/{project_id}`. Coder writes are checked
against the task's `write_set`; Tester may create tests only under `tests/`. Every
changed file batch is applied atomically and committed with an agent/task-specific
message. Tester creates a test suite only when no discoverable tests exist, then runs
`pytest -q` or Jest in the Docker sandbox. The result includes pass/fail, exit code,
timeout state, bounded stdout/stderr, tested files, and execution evidence. Debugger
receives those logs plus relevant code, returns actionable structured repair guidance,
and never writes files itself. The FSM permits at most three Debugger-guided retries
per task before escalating; retries and agent calls remain persisted and auditable.

Run the offline build-agent, RAG, Git, and feedback-propagation tests with:

```bash
python -m pytest tests/test_projectforge_build_agents.py -q
```

The same test module includes a Docker-gated end-to-end run of the Phase 3
`simple_script.json` fixture through PLAN, human plan approval, and BUILD_LOOP. It
asserts the generated tests pass in the isolated sandbox and checks the final project
repository. That live test skips locally without Docker and is required by CI alongside
the Phase 4 sandbox suite.

### CRUD surface

- `POST /api/projects` (accepts `raw_brief_text` and optional `rubric_text`, or a legacy validated `spec`), `GET /api/projects`, `GET /api/projects/{project_id}`
- `PATCH /api/projects/{project_id}`, `DELETE /api/projects/{project_id}`
- `POST /api/projects/{project_id}/task-graphs`
- `GET /api/projects/{project_id}/task-graphs` and `GET .../{graph_id}`
- `PUT /api/projects/{project_id}/task-graphs/{graph_id}`
- `DELETE /api/projects/{project_id}/task-graphs/{graph_id}`
- `POST /api/projects/{project_id}/workflows` (create and queue a run; a seed graph is optional)
- `GET /api/workflows/{workflow_id}` and `GET .../events`
- `POST /api/workflows/{workflow_id}/clarifications` (resume a Requirement-agent clarification gate)
- `POST /api/workflows/{workflow_id}/approvals` (`plan` or `final`)
- `POST /api/workflows/{workflow_id}/cancellation`
- `POST /api/workflows/{workflow_id}/dispatch` (retry a persisted queue dispatch)
- `GET /health`

Task graph contracts reject duplicate IDs, missing dependencies, mismatched edges,
and cycles. Resource declarations use relative paths; undeclared workspace access is
exclusive, and read/write overlaps block parallel scheduling. A graph version becomes
immutable once a workflow references it. Task nodes and workflow runs persist retries (hard-capped at 3), timeouts, cancellation
requests, budget usage, states, and audit events. Workflow jobs pause at
`AWAITING_CLARIFICATION`, `AWAITING_PLAN_APPROVAL`, and
`AWAITING_FINAL_APPROVAL`. Clarification answers resume Requirement processing;
approval is the only way to proceed past the two human gates. Completion stores an
AI-usage disclosure derived from
that project's persisted agent-run audit trail. API/orchestrator tests use SQLite for
isolation; the Alembic migrations target PostgreSQL/JSONB and have not been applied to
a live database in this environment. Phase 3 evaluations use mocked completions for
simple-script, web-app, and data-analysis briefs; no billable Dahl request or live
PostgreSQL/Redis workflow was run here. Dahl `max_tokens` and provider usage/cost fields
still require a configured deployment smoke test because the inspected Dahl API docs
do not establish those response details.

### ProjectForge code map

```text
packages/schemas/                 Shared Pydantic v2 contracts
packages/agents/requirement.py     Model-backed brief normalization and clarification
packages/agents/planner.py         Model-backed validated DAG generation
packages/agents/build_agents.py    Structured model-backed Coder/Tester/Debugger workers
packages/agents/rag.py             ProjectSpec chunking and pgvector cosine retrieval
packages/agents/git_workspace.py   Scoped per-project Git workspaces and agent commits
packages/agents/litellm_client.py  Configured Dahl/OpenAI-compatible LiteLLM adapter
packages/agents/langgraph_runner.py One-node linear LangGraph call; no control-flow decisions
packages/agents/prompts/           Versioned Requirement/Planner/Coder/Tester/Debugger prompts
packages/agents/stubs.py           Deterministic stubs for integration/review/packaging/tests
packages/orchestrator/engine.py    Postgres FSM, build retries, clarification gates, DAG persistence
packages/orchestrator/budget.py    Atomic provider budget reservations and settlement
packages/orchestrator/scheduler.py Topological/resource-conflict task scheduler
packages/orchestrator/queue.py    RQ/Redis dispatch adapter
packages/sandbox/manager.py        Ephemeral Docker execution, limits, cleanup, and network policy
packages/sandbox/egress_proxy.py   Allowlist-only HTTP(S) egress proxy
infra/sandbox/                    Python/pytest, Node/Jest, and egress-proxy images
apps/api/main.py                   FastAPI CRUD and workflow/approval endpoints
apps/api/db.py                     Engine/session configuration
apps/api/orm.py                    SQLAlchemy persistence mappings
apps/api/cli.py                    Local API launcher (127.0.0.1 by default)
infra/alembic.ini                  Alembic configuration
infra/migrations/                  PostgreSQL migrations (Phases 1–5)
infra/docker-compose.yml           Local Postgres/pgvector + Redis services
tests/test_projectforge_*.py       Contract, API, persistence, scheduler, and FSM tests
```
