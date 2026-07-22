## Project Guidelines

Specifications for this project are present in `<root-dir>/specs/`.

Main code base is in `<root-dir>/src/`.

---

## HARD RULES

All project work must remain within the root directory of the repository
(where this file is stored). Do not modify, delete, or inspect unrelated
files outside this directory. Read-only access to explicitly provided
context outside the repository, such as Cursor-provided tool instructions,
terminal state, or user-attached files, is allowed only when needed to
perform the requested work.

---

## Agent Instructions (Workflow Protocol)

This repository follows a recursive execution workflow.

For any non-trivial task:

1. FRAME the problem. Identify objectives, validation criteria,
   integration challenges, and other relevant task data.
2. EXECUTE directly or via specialized subagents. Delegate only when
   it materially improves reliability, reduces context pressure, enables
   independent investigation, or allows safe parallel execution.
3. INTEGRATE all child work against the original requirements.
4. CLOSE only after validating the unified deliverables.

Subagents may recursively apply this workflow when appropriate.

Do not accept child summaries without independently inspecting and verifying their work.

Prefer:
- direct execution over unnecessary delegation
- delegation over oversized contexts
- evidence over confidence
- integration over local correctness
- targeted repair over restarting
- preserving task scope over opportunistic improvements

See `WORKFLOW.md` for the complete protocol and delegation guidelines.
Consult it when coordinating multiple subagents, resolving integration issues,
or when additional workflow detail would improve execution.

---

## Agent Behavioral Constraints (Always Active)

### 1. Explicit assumptions
The agent must:
- State assumptions that materially affect implementation
- Surface alternative interpretations when they would change behavior
- Ask questions instead of guessing when uncertainty cannot be resolved from repository evidence

### 2. Simplicity bias
Prefer the smallest correct solution.

Do not:
- Add unused abstractions
- Introduce configurability not requested
- Expand scope beyond the request

If a simpler solution exists, prefer it.

### 3. Minimal diff principle
Changes must be strictly scoped to the request.

If adjacent improvements, refactors, or cleanup opportunities are
discovered outside the scope of the current change:
- Do not implement them in the current phase
- If relevant, reference them in the phase review notes for future work
- Add TODO comments only when touching the directly relevant code and the
  TODO is specific, actionable, and consistent with the TODO style guide

Do not:
- Refactor unrelated code
- Reformat unrelated sections
- Remove pre-existing dead code unless directly impacted

Only clean up artifacts directly caused by the change.

### 4. Verifiable progress
All work must be tied to a validation mechanism:
- tests
- reproducible behavior
- explicit acceptance criteria

Every phase should define:
- how success is verified
- what “done” means concretely

### 5. Completion criteria
A phase is considered complete only when all of the following are satisfied:
- All explicitly defined success criteria for the phase are met
- All required tests pass (if applicable)
- No unaddressed errors, type-check failures, or lint violations remain
- All changes remain within the approved scope of the phase plan

If any criterion cannot be satisfied, the agent must stop and report the blocking issue.

---

## Code Style & Tooling

### Python
- Lint: ruff
- Type check: ty

### TODO Style Guide
TODO comments must include:
- a short tag identifying the type of issue
- a concise description of the problem
- optional context only when necessary for clarity

Optional: include a brief note on intended direction for future resolution.

Example:
```py
# TODO(todo-tag): This is a brief description of the problem
# this todo tag is highlighting. This is a brief list of additional
# relevant considerations, and an optional list of potential future
# remedies.
```
