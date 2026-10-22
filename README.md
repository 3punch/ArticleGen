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
# Verify is local; article generation requires an LLM provider key.
python3 -m articlegen verify examples/brief_example.json   # gate only, zero tokens
export DAHL_API_KEY='<Dahl API key>'
export DAHL_MODEL='zai-org/GLM-5.3-Flash'                  # optional; default shown
python3 -m articlegen demo                                 # full loop on the example brief
python3 -m articlegen run path/to/brief.json --iterations 4
python3 -m articlegen log                                  # summarise change_log.jsonl
```

Without a provider key, a run stops before creating a draft. There is no offline
article-generation fallback.

Or with the console script after `pip install -e .`:

```bash
articlegen run path/to/brief.json --target 0.9 --provider auto
```

### One prompt → full article (`auto`)

The `auto` command takes a single natural-language prompt, builds a complete brief
from it, and runs the **entire** pipeline autonomously — plan, retrieve/verify
sources, execute experiments, write, peer-review, and revise — looping until the
research quality gate reaches *Ready for final review* (or until a re-run makes no
progress, so a blocker such as an unavailable source provider cannot burn tokens
forever). The client is made self-healing: on a rate limit (HTTP 429) it retries
with backoff and then falls back to alternative models.

```bash
# one sentence in, a full validated article out
articlegen auto "Write a computational article explaining how to verify the \
Erdős–Gyárfás conjecture on a sample of cubic graphs, with a runnable Python experiment"

# recover from a capacity-limited model automatically
export ARTICLEGEN_FALLBACK_MODELS='deepseek-ai/DeepSeek-V4-Flash-0731,MiniMaxAI/MiniMax-M2.7'
articlegen auto "your prompt here" --out article.md
```

The Streamlit dashboard exposes the same flow as a **⚡ One-prompt mode** expander at
the top of the page: paste one prompt, press *Generate full article*.

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
change-log path) live in the sidebar. The Dahl section accepts the API key in a
masked password field plus a model ID; the key remains in session state only and is
never copied to the brief, config file, or run log. The Run button stays disabled until
the selected provider has a key.

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
| Per-agent input/output ceilings | 1800–2400 / 4096–8192 | `TokenBudget.agents` |
| Whole-run token ceiling | 80000 tokens | `TokenBudget.run_token_ceiling` |
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
* **Repairs converge.** Existing draft text can be cleaned mechanically (filler
  removal, long-sentence/paragraph splitting, contraction expansion, word dedup).
  When new content is needed, the Writer requests it from the configured model; there
  is no static article scaffold or offline drafting fallback.

The savings depend on the brief and provider: skipped calls and smaller delta packets
reduce spend, while all new writing still comes from the configured LLM.

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
movement, findings applied/deferred, token spend, and any context eviction. The values
below are an illustrative model-backed log shape:

```markdown
## [2026-10-11T16:43:51Z] task=demo-article · iteration 2 · decision=CONTINUE

- gate: PASS (5 minor) at draft_verification
- agents: writer=ok/0f/dahl:zai-org/GLM-5.3-Flash · tester=ok/0f/dahl:zai-org/GLM-5.3-Flash · …
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
| `dahl` | `DAHL_API_KEY` set (or `--provider dahl`) | real model calls via `urllib` against the Dahl inference API |
| `unconfigured` | no usable provider key | run stops before the agent loop; no draft is generated |

The `dahl` provider speaks the OpenAI-compatible chat-completions protocol at
`https://inference.dahl.global/v1/chat/completions` (Bearer auth). Defaults and
overrides:

```bash
export DAHL_API_KEY='<Dahl API key>'          # required
export DAHL_MODEL='zai-org/GLM-5.3-Flash'     # optional; this is the default model
export DAHL_BASE_URL='https://inference.dahl.global/v1'   # optional; the default
export ARTICLEGEN_LLM_TIMEOUT_SECONDS=600       # optional; request timeout defaults to 300 seconds
# or per run:
articlegen run brief.json --provider dahl --model zai-org/GLM-5.3-Flash
```

ArticleGen does **not** generate articles offline. If no usable API key is configured,
the orchestrator stops before creating a draft and explains how to configure a provider.
The deterministic checkers remain available to unit tests, but they are not a writing
engine and the Writer contains no static sentence-template pool.

In the dashboard sidebar, choose `dahl`, enter the API key in the password field, and
set the model ID (default: `zai-org/GLM-5.3-Flash`). The key is held only in the
Streamlit session and is not written to the brief, config, or run logs. For CLI runs,
set `DAHL_API_KEY` and optionally `DAHL_MODEL` in the environment. Provider calls have
a 300-second socket timeout by default; set `ARTICLEGEN_LLM_TIMEOUT_SECONDS` (or pass
`timeout` to `resolve_client`) to allow longer completions. A timed-out call is recorded
as a degraded agent with an actionable message rather than an uncaught UI traceback.

**Verification status:** provider request formats are covered by stubbed HTTP tests;
this sandbox cannot call the live Dahl/Anthropic/OpenAI endpoints.

---

## CI

`.github/workflows/pipeline.yml` runs the Python suite and Docker-backed sandbox/E2E
tests, typechecks and builds the Next.js workspace, verifies the valid/invalid example
briefs and prompt templates, then executes the orchestrated loop and uploads `runs/` as
an artefact. `workflow_dispatch` accepts a brief path, iteration count and provider;
add `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `DAHL_API_KEY` as a repository secret to
run against a real model.

---

## ProjectForge — Phases 1–7

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
only the top relevant excerpts rather than the entire specification. Phase 6 adds the
evidence-bearing Reviewer, task-scoped human rework decisions, deterministic integration
checks, README/report generation, audit-trail AI-usage disclosure, and bounded ZIP
export. Phase 7 adds the Next.js + Tailwind human workspace for clarification, plan and
final-review checkpoints, live task progress/logs, and package download. Orchestration
and control flow remain deterministic code; no LLM chooses state transitions. The
workflow engine is exposed through a swappable `WorkflowEngine` interface. Production
tables are not created on API startup; apply the Alembic migrations explicitly. The API
defaults to loopback because authentication is not configured and must not be exposed
publicly without an authenticated deployment.

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

### Run the Next.js workspace

In another terminal, set the server-side API target and start the web app:

```bash
cd apps/web
npm ci
cp .env.example .env.local
npm run dev
```

The browser uses same-origin `/api/...` calls; Next.js proxies them to the FastAPI
origin configured by `PROJECTFORGE_API_URL` (default `http://127.0.0.1:8000`). For a
container network, point it at the API service name instead. The frontend requires the
API, PostgreSQL, Redis/RQ worker, and configured ProjectForge agents for workflow
actions; it does not substitute local mock execution when those services are absent.
The Next.js dev server binds to `0.0.0.0` for the workspace preview.

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
`PROJECTFORGE_DEBUGGER_MODEL`, `PROJECTFORGE_REVIEWER_MODEL`, and
`PROJECTFORGE_PACKAGER_MODEL`. `PROJECTFORGE_LLM_ROUTE_PREFIX` defaults to
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

### Phase 6 reviewer, human gates, and packaging

The Reviewer assesses automatable rubric items using bounded source/test evidence and
static Python parsing. Subjective items remain `human_review_required`, with a
recommendation and evidence for the human reviewer; they are never persisted as a
machine pass. The approval API returns rejected plans to `PLAN` and routes final
change requests to `BUILD_LOOP` with feedback attached to selected TaskNodes and
relevant dependents. The Packager commits a README and Markdown report to the project
Git workspace; the ZIP includes test/review evidence, the disclosure in Markdown and
JSON, a checksum manifest, and no Git metadata or excluded workspace files. The final
package download endpoint verifies the persisted size and SHA-256 before serving it.

`tests/test_projectforge_phase6_e2e.py` exercises all three Phase 3 fixtures through
both human gates and asserts the downloadable bundle contains README, report,
disclosure, tests, and passing test evidence. Its Docker-backed test is mandatory in
CI; offline API/workflow/package tests use an explicit sandbox stub and do not execute
project code.

### CRUD surface

- `POST /api/projects` (accepts `raw_brief_text` and optional `rubric_text`, or a legacy validated `spec`), `GET /api/projects`, `GET /api/projects/{project_id}`
- `PATCH /api/projects/{project_id}`, `DELETE /api/projects/{project_id}`
- `POST /api/projects/{project_id}/task-graphs`
- `GET /api/projects/{project_id}/task-graphs` and `GET .../{graph_id}`
- `PUT /api/projects/{project_id}/task-graphs/{graph_id}`
- `DELETE /api/projects/{project_id}/task-graphs/{graph_id}`
- `POST /api/projects/{project_id}/workflows` (create and queue a run; a seed graph is optional)
- `GET /api/workflows/{workflow_id}` and `GET .../events`
- `GET /api/workflows/{workflow_id}/review` (current plan or final-review diff/checklist)
- `GET /api/workflows/{workflow_id}/tasks/{task_id}/logs` (bounded coder/tester/debugger audit output)
- `POST /api/workflows/{workflow_id}/clarifications` (resume a Requirement-agent clarification gate)
- `POST /api/workflows/{workflow_id}/approvals` (`plan` or `final`, with task-scoped rework feedback)
- `POST /api/workflows/{workflow_id}/cancellation`
- `POST /api/workflows/{workflow_id}/dispatch` (retry a persisted queue dispatch)
- `GET /api/workflows/{workflow_id}/package/metadata` and `GET .../package` (integrity-checked ZIP download)
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
packages/agents/prompts/           Versioned agent prompts
packages/agents/reviewer.py         Evidence-based rubric assessment and human-review recommendations
packages/agents/packager.py         README and Markdown report generation
packages/agents/package_export.py   Bounded, integrity-checked ZIP export and disclosure artifacts
packages/agents/stubs.py            Deterministic test stubs; production roles use scoped agents
packages/orchestrator/engine.py     Postgres FSM, build retries, human gates, and audit disclosure
packages/orchestrator/budget.py     Atomic provider budget reservations and settlement
packages/orchestrator/scheduler.py  Topological/resource-conflict task scheduler
packages/orchestrator/queue.py      RQ/Redis dispatch adapter
packages/sandbox/manager.py         Ephemeral Docker execution, limits, cleanup, and network policy
packages/sandbox/egress_proxy.py    Allowlist-only HTTP(S) egress proxy
apps/web/                            Next.js + Tailwind human workspace and same-origin API proxy
apps/api/main.py                    FastAPI CRUD, workflow, approval, task-log, and export endpoints
apps/api/db.py                      Engine/session configuration
apps/api/orm.py                     SQLAlchemy persistence mappings
apps/api/cli.py                     Local API launcher (127.0.0.1 by default)
infra/sandbox/                      Python/pytest, Node/Jest, and egress-proxy images
infra/alembic.ini                   Alembic configuration
infra/migrations/                   PostgreSQL migrations (Phases 1–6)
infra/docker-compose.yml            Local Postgres/pgvector + Redis services
tests/test_projectforge_*.py        Contract, API, persistence, scheduler, FSM, and packaging tests
```
