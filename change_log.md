# ArticleGen Change Log

Append-only. Every pipeline iteration appends exactly one block; nothing here is
rewritten. This file is the pipeline's long-term memory substitute: instead of
re-passing full agent histories into each context window, prior state is
compressed into the blocks below and referenced from there.

Legend: `gate` = input/draft verification result, `score` = weighted quality
score (higher is better), `evicted` = context blocks dropped to respect the token
cap, `applied` = fixes the Synthesizer merged, `deferred` = findings carried
forward to the next iteration.


## [2026-10-11T16:45:46Z] RUN_START · task=demo-article

- title: Designing an Iterative Multi-Agent Writing Pipeline
- mode: article
- target_words: 1000
- sections: 4
- engine: offline-heuristic
- max_iterations: 3
- score_target: 0.85
- context_hard_cap: 6000

## [2026-10-11T16:45:46Z] task=demo-article · iteration 1 · decision=CONTINUE

- gate: PASS (5 minor) at draft_verification
- agents: writer=ok/0f/offline-heuristic · tester=ok/4f/offline-heuristic · reviewer=ok/0f/offline-heuristic · architect=ok/1f/offline-heuristic · synthesizer=ok/0f/offline-heuristic
- score: — → **0.7241** (Δ+0.0000)
- findings: 5 raised · 4 applied · 5 deferred
- draft: 1188 words at end of iteration
- context: cap=6000 evicted=none
- tokens: 11709 this iteration (run total 11709)

### Deferred (carried forward in compressed form, never re-passed in full)

- [blocker] tester/logic@introduction: section 'Introduction' is empty; the deliverable is incomplete
- [blocker] tester/logic@decision-gate: section 'The Decision Gate' is empty; the deliverable is incomplete
- [blocker] tester/logic@agent-roles: section 'Mandatory Agent Roles' is empty; the deliverable is incomplete
- [blocker] tester/logic@token-budget: section 'Token and Memory Policy' is empty; the deliverable is incomplete
- [major] architect/coverage@document: objective vocabulary largely absent from the draft (11/12 key terms uncovered)

## [2026-10-11T16:45:46Z] task=demo-article · iteration 2 · decision=CONTINUE

- gate: PASS (5 minor) at draft_verification
- agents: writer=ok/0f/offline-heuristic · tester=ok/0f/offline-heuristic · reviewer=ok/6f/offline-heuristic · architect=ok/0f/offline-heuristic · synthesizer=ok/0f/offline-heuristic
- score: 0.7241 → **0.9169** (Δ+0.1928)
- findings: 6 raised · 1 applied · 5 deferred
- draft: 1181 words at end of iteration
- context: cap=6000 evicted=none
- tokens: 15341 this iteration (run total 15341)

### Applied by Synthesizer

| agent | severity | section | claim | action |
| --- | --- | --- | --- | --- |
| reviewer | minor | document | 7 filler construction(s) add words without meaning | Delete the filler; the sentence carries the point without it. |

### Deferred (carried forward in compressed form, never re-passed in full)

- [minor] reviewer/readability@introduction: sentence runs to 42 words (ceiling 30)
- [minor] reviewer/readability@introduction: passive constructions dominate this section
- [minor] reviewer/readability@decision-gate: sentence runs to 42 words (ceiling 30)
- [minor] reviewer/readability@decision-gate: sentence runs to 42 words (ceiling 30)
- [minor] reviewer/readability@decision-gate: passive constructions dominate this section

## [2026-10-11T16:45:46Z] RUN_END · task=demo-article

- status: PASS
- stop_reason: score target met (0.9169 >= 0.85)
- iterations: 2
- final_score: 0.9169
- score_target: 0.85
- tokens_total: 27050
- draft_words: 1181
- final_draft_gate: PASS
- master_draft: runs/demo-article/master_draft.md
- open_findings: 4
