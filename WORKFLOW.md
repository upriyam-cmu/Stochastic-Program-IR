# Agent Workflow

## Purpose

Use structured decomposition and targeted subagents to complete repository work reliably while preserving the intent of the original request.

The primary agent owns the task end to end. Subagents are used to investigate, implement, verify, or repair bounded portions of the work. Delegation does not transfer responsibility for the final result.

Prefer lightweight coordination over ceremony. Do not create planning artifacts, task trees, or approval checkpoints unless they materially improve the work or the user explicitly requests them.

## Core Lifecycle

For any non-trivial task, follow this lifecycle:

> FRAME → EXECUTE → INTEGRATE → CLOSE

Apply the lifecycle recursively when a delegated task is itself too broad to execute coherently.

Small, localized tasks may pass through these stages implicitly. Complex tasks should make the stages explicit in the working process.

---



## 1. FRAME

Before making changes, establish a sufficient working model of the task.

Determine:

- the intended outcome;
- the relevant acceptance criteria;
- the likely affected systems or files;
- important constraints and non-goals;
- how the result will be validated;
- whether the work should be handled directly or decomposed.

Do not ask the user for clarification when the repository, existing tests, or reasonable local inference can resolve the ambiguity safely.

Ask only when a material product, architectural, or behavioral choice cannot be inferred and different interpretations would lead to meaningfully different results.

### Decide Whether to Decompose

Handle the task directly when it is narrow, coherent, and locally verifiable.

Use subagents when one or more of the following is true:

- the task spans distinct modules or concerns;
- repository exploration can be performed independently;
- implementation can be divided into bounded ownership areas;
- independent verification would reduce risk;
- the task contains substantial uncertainty;
- holding all relevant context in one agent would make execution less reliable;
- multiple plausible failure modes should be investigated independently.

Do not decompose merely to create activity. Delegation must have a clear expected benefit.

---



## 2. EXECUTE

Carry out the work directly or through targeted subagents.

The primary agent must retain a coherent view of:

- the original user request;
- the current implementation strategy;
- dependencies between delegated tasks;
- shared interfaces and files;
- unresolved risks;
- the state of the aggregate change.



### Subagent Roles

Use role-specialized subagents where appropriate.

#### Explorer

Use for read-only investigation, such as:

- locating relevant code paths;
- tracing data flow;
- identifying repository conventions;
- finding related implementations;
- discovering invariants, dependencies, and risks;
- comparing possible implementation approaches.

Explorers should not edit files unless explicitly assigned implementation work.

#### Implementer

Use for a bounded code change with clear ownership.

An implementer should receive:

- the local objective;
- relevant parent intent;
- allowed scope;
- important constraints;
- expected behavior;
- validation requirements;
- known dependencies on other work.



#### Verifier

Use for independent inspection of completed work.

A verifier should evaluate the actual code and evidence, not merely review the implementer's summary.

Verification may include:

- running focused tests;
- inspecting the diff;
- checking acceptance criteria;
- identifying regressions;
- checking repository conventions;
- looking for missing edge cases;
- detecting unnecessary abstractions or duplicated logic.



#### Repairer

Use after a concrete defect, failed test, integration issue, or scope deviation has been identified.

A repairer should receive a narrow diagnosed problem rather than a request to reconsider the entire feature.

#### Integrator

Use when multiple substantial child outputs must be reconciled and an independent integration pass would be useful.

The primary agent remains the final integration owner even when an integrator subagent is used.

### Delegation Contract

Every delegated task should specify, as concisely as possible:

- objective;
- relevant context;
- allowed modification scope;
- constraints and non-goals;
- dependencies or sequencing requirements;
- expected output;
- validation expectations.

Subagents should return a compact result containing:

- what they found or changed;
- files affected;
- validation performed;
- assumptions made;
- unresolved risks or blockers;
- any deviation from the assigned scope;
- recommended parent-level follow-up.

Subagent summaries are not proof of correctness. The parent must inspect the relevant code, diff, tests, or artifacts before accepting the result.

### Recursive Delegation

A subagent may further decompose its assignment when:

- the assignment is still too broad to execute coherently;
- the child tasks have clear boundaries;
- recursion reduces context pressure or improves independent verification.

Recursive delegation must preserve the assigned objective and scope.

A subagent must not broaden the parent task, invent new product requirements, or delegate away responsibility for understanding its own result.

Avoid deep delegation trees when a direct implementation would be simpler.

### Parallelism

Parallelize read-only investigation aggressively when independent perspectives are useful.

Parallelize implementation only when the tasks have sufficiently isolated write scopes.

Parallel implementation is appropriate when:

- child tasks have separable outcomes;
- they do not modify the same files or generated artifacts;
- neither task changes an interface the other depends on;
- they do not share a migration, schema, registry, dependency manifest, or other coordination hotspot;
- each task can be validated independently;
- there is a clear integration owner.

When these conditions do not hold, prefer:

- parallel exploration followed by sequential implementation;
- one task defining a contract before dependent tasks begin;
- sequential implementation;
- isolated worktrees or branches when available.

Do not allow multiple agents to make overlapping edits in the same working tree without an explicit reconciliation strategy.

### Scope Discipline

Agents may make small adjacent changes required to complete or validate the assigned objective.

Do not introduce:

- unrelated refactors;
- speculative abstractions;
- unrequested features;
- broad cleanup unrelated to the task;
- architectural changes that are not justified by the requested behavior.

When the current approach is blocked by a materially larger design issue, report the issue to the parent rather than silently expanding scope.

---



## 3. INTEGRATE

After direct work or a round of delegated work completes, reopen the parent objective.

Do not treat successful child completion as successful task completion.

The primary agent must inspect the combined result and determine whether it satisfies the original request as a coherent whole.

Integration review should include, as relevant:

- inspect the aggregate diff;
- confirm that child outputs are mutually compatible;
- check that shared interfaces and assumptions align;
- identify overlapping or contradictory changes;
- rerun important tests after all changes are combined;
- map acceptance criteria to concrete evidence;
- check for dropped requirements;
- check for duplicated logic or unnecessary abstractions;
- check whether changes follow repository conventions;
- consider whether a simpler implementation would satisfy the same goal;
- verify that local child success composes into parent-level success.



### Integration Failures

When integration reveals a problem:

1. diagnose the failure concretely;
2. determine whether it is localized or structural;
3. dispatch a bounded repair task when possible;
4. resplit or reconsider the plan when the decomposition itself was flawed;
5. rerun the relevant integration checks.

Prefer targeted repair loops over restarting the entire task.

Do not repeatedly dispatch repair agents without revisiting the underlying assumptions when the same class of failure persists.

---



## 4. CLOSE

Before reporting completion:

- confirm the requested outcome is implemented;
- confirm relevant validation has passed;
- review the final diff for accidental or unrelated changes;
- identify any unresolved limitations or risks;
- ensure temporary debugging artifacts are removed;
- ensure the repository is left in a coherent state;
- summarize the implementation and evidence concisely.

Do not claim completion based only on subagent reports.

When validation cannot be completed, state exactly what was and was not verified.

---



## Read-Only Requests

For analysis, explanation, investigation, or review tasks:

- do not modify files;
- use explorer or verifier subagents when parallel investigation or independent review would improve the result;
- synthesize findings at the parent level;
- distinguish verified repository facts from inference.

The full execution and repair workflow is unnecessary when no modifications are requested.

---



## Human Interaction

For this workflow, routine decomposition, delegation, implementation, verification, and repair do not require explicit user approval.

Pause and ask the user only when:

- the request is materially ambiguous and repository evidence cannot resolve it;
- multiple valid product behaviors exist with meaningful consequences;
- the work requires destructive or irreversible action;
- required credentials, external access, or unavailable information block progress;
- the necessary change would materially exceed the requested scope;
- the agent discovers a major architectural decision that should belong to the user.

Otherwise, proceed using the best supported interpretation and report important assumptions.

---



## Operating Principles

- The primary agent owns the final result.
- Delegate bounded outcomes, not vague responsibility.
- Use subagents to reduce context pressure, isolate concerns, and introduce fresh review.
- Prefer evidence over agent confidence.
- Prefer executable validation over verbal self-assessment.
- Inspect child work directly before integrating it.
- Split only when decomposition improves reliability or throughput.
- Parallelize cognition more readily than mutation.
- Keep coordination artifacts proportional to task complexity.
- Preserve parent intent across every level of delegation.
- Repair diagnosed failures rather than vaguely asking agents to “try again.”
- Completion requires parent-level integration, not merely completed subtasks.

---



## Anti-Patterns

Avoid:

- creating subagents for trivial tasks;
- delegating the entire request to one child and acting only as a messenger;
- accepting child summaries without inspecting their work;
- allowing every subagent to edit unrestricted repository scope;
- running overlapping implementations in parallel without isolation;
- treating tests from individual children as sufficient integration evidence;
- producing elaborate plans that are not used during execution;
- forcing every task into the same decomposition depth;
- recursively delegating until no agent has a coherent model of the whole;
- using verifier agents as a substitute for running available tests;
- broadening scope during repair without informing the parent;
- declaring success before reviewing the combined result.

