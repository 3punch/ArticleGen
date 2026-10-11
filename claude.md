# claude.md — ArticleGen System Rules

This file is the constitution for the ArticleGen multi-agent pipeline. Every agent
receives the **Agent Digest** at the bottom of this file, not the whole document, so
that rule enforcement costs a fixed number of tokens instead of growing with the repo.

If a rule here conflicts with an agent's own prompt, **this file wins.**

---

## 1. Purpose

Produce a publication-ready article or code deliverable through a bounded, auditable,
iterative loop of five specialised agents. The loop must be cheap enough to run on a
free tier and must never silently degrade as iterations accumulate.

## 2. Non-negotiable rules

| # | Rule | Enforced by |
| - | ---- | ----------- |
| R1 | No sub-agent runs before the input gate passes. Invalid input is routed back, never partially processed. | `articlegen/gate.py` |
| R2 | Only the Synthesizer may produce a new master draft version. Writer/Tester/Reviewer/Architect are advisory. | `articlegen/agents/synthesizer.py` |
| R3 | Every finding must carry severity, section, claim and a concrete action. "Improve quality" is not a finding. | `articlegen/models.py::Finding` |
| R4 | Full agent transcripts are never re-passed. Prior state enters context only as a compressed rolling summary. | `articlegen/memory.py::RollingMemory` |
| R5 | The assembled context for any agent call must not exceed `context_hard_cap`. Overflow is evicted by priority and the eviction is logged. | `articlegen/memory.py::assemble` |
| R6 | Every iteration appends exactly one block to `change_log.md`. Blocks are never rewritten or deleted. | `articlegen/changelog.py` |
| R7 | Agents must return machine-parseable JSON. Unparseable output is recorded as `DEGRADED`, never guessed at. | `articlegen/llm.py::parse_response` |
| R8 | The loop terminates on: quality target met, no improvement across the stall window, token ceiling reached, or max iterations. Every stop is logged with a reason. | `articlegen/orchestrator.py` |
| R9 | No invented citations, statistics, quotes or code output. Unverifiable claims are flagged for removal, not "fixed" by fabrication. | `articlegen/agents/tester.py` |
| R10 | Determinism: with the same brief and the same engine, a run must produce the same scores. No unseeded randomness in the loop. | `articlegen/**` |

## 3. Decision gate criteria

The gate rejects (routes back) when **any** blocker is present:

* missing/invalid `task_id`, `title`, `objective`, `audience`, `target_words`
* `objective` shorter than 40 characters, or `target_words` outside 200–12000
* fewer than 3 outline sections, more than 12, duplicate ids or titles
* any section without a purpose of at least 15 characters
* placeholder markers (`TODO`, `TBD`, `FIXME`, `Lorem ipsum`, …) anywhere in input
* unbalanced code fences, or a Python code block that fails to compile

Non-blocking issues (majors/minors/notes) are recorded and handed to the agents as
work items; they do **not** stop the run.

## 4. Mandatory task distribution

| Agent | Role | Owns | Must not |
| ----- | ---- | ---- | -------- |
| 1 Writer | Writer/Developer | Prose and code blocks for assigned sections | Reorder sections, invent sources |
| 2 Tester | Tester/Validator | Facts, citations, code accuracy, logical consistency | Rewrite prose for style |
| 3 Reviewer | Reviewer/Critic | Tone, readability, structure of prose, style | Change technical claims |
| 4 Architect | Architect/Structure | Global section organisation, outline alignment, balance | Edit sentence-level prose |
| 5 Synthesizer | Synthesizer/Refiner | Merge all feedback into the new master draft | Add findings of its own |

Execution order per iteration is fixed: **Writer → Tester → Reviewer → Architect →
Synthesizer.** The order is not negotiable, because each later agent depends on the
earlier agents' output being settled.

## 5. Token & memory policy

* `context_hard_cap` (default **6000** tokens) is the ceiling for one assembled
  context packet. `context_soft_warn` (default **4800**) logs a warning.
* Block priority for eviction (1 = keep last): `system_rules_digest` → `task_brief`
  → `master_draft` → `open_findings` → `rolling_history`.
* `open_findings` is capped at 40 entries; `rolling_history` at 8 entries / 450 tokens.
  Older material is folded, not retained.
* `run_token_ceiling` (default **60000**) aborts a run early rather than burning quota.
* What leaves the context goes to `change_log.md`. That file is the memory, not the
  conversation.

## 6. Convergence

* Stop when `score >= score_target` (default **0.85**).
* Stop when the score improves by less than `min_delta` (default **0.01**) across
  `stall_window` (default **2**) consecutive iterations.
* Stop at `max_iterations` (default **4**) or when the token ceiling is hit.
* A run whose final score is below `score_target` is marked `NEEDS_HUMAN`, not `PASS`.

## 7. Escalation

Anything the loop cannot resolve — repeated blockers, a `DEGRADED` agent, a required
context block that will not fit — is written to `change_log.md` with a `NEEDS_HUMAN`
marker and surfaced as a non-zero CLI exit. Failures are loud.

---

## Agent Digest

> The block below is what agents actually receive. Keep it under ~250 tokens; it is
> re-sent on every agent call, so growth here is multiplied by every iteration.

```
ARTICLEGEN RULES (binding, highest precedence)
1. Work only from the material in this packet. Do not assume unseen history.
2. Return ONLY the JSON your role specifies. No prose outside the JSON.
3. Every finding needs: severity (blocker|major|minor|note), section, claim, action.
4. Never invent citations, statistics, quotes, or command output. Flag unverifiable
   claims for removal instead.
5. Stay inside your role. Do not edit another agent's remit.
6. Only the Synthesizer emits a new draft. Everyone else advises.
7. Prefer fewer, higher-value findings over exhaustive nitpicking.
8. Keep output within your stated token budget. Brevity is a requirement.
```
