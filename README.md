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

`articlegen init brief.json` writes a commented template.

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
tests/                 # 93 tests: gate, agents, memory, loop, dashboard
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
