# Codex Operating Instructions

Use `GEMINI.md` as the shared project source of truth. Keep this file short because Codex loads it as repo context.

## Startup Context
- At the start of a new task, read `GEMINI.md` before acting.
- Check the current Git branch with `git branch --show-current`.
- If `.gemini/tmp/clintrialpredict/memory/MEMORY.md` exists, read only entries matching the current branch prefix.
- If branch memory is long, summarize the latest relevant goal, decisions, files, blockers, and next step before continuing.
- Do not read unrelated branch memory unless the user explicitly asks.
- Treat memory as guidance, not truth. Current user instructions, current code, and test results override older memory entries.

## Efficient, Surgical Workflow
- Start by identifying the smallest relevant file set. Read those files first, then expand when correctness requires imports, call sites, configs, tests, runtime paths, or shared helpers.
- Prefer `rg`, targeted file reads, and exact symbols over broad scans.
- Do not paste or restate large file contents in responses; summarize findings and cite paths.
- Explain why before expanding into unrelated files, generated artifacts, model files, broad refactors, or expensive verification.
- Make the smallest behavior-preserving change that satisfies the request.
- Avoid opportunistic cleanup, formatting churn, dependency changes, and unrelated metadata edits.
- Before code edits, state the files you will touch and why. Documentation-only edits may proceed when the user directly requests them.
- After editing, run the narrowest meaningful verification first, then broader checks when risk, dependencies, scoring behavior, or deployment paths justify them.
- Do not let token economy override correctness, robustness, or parity requirements.

## Prompting Defaults
- Treat vague requests as requests for a focused investigation before implementation, but proceed when the requested change is clear.
- For bug fixes, reproduce or inspect the failure path before changing code.
- For feature work, preserve existing architecture and UI conventions unless the user asks otherwise.
- For reviews, lead with concrete risks and file references.
- If context grows large, summarize the current facts and continue with the smallest next step.
- If instructions conflict, prioritize the newest user instruction, current repository state, and safety-critical project rules.

## Cost Controls
- Use smaller models for routine local edits when available; use stronger models for complex debugging, architecture, data pipelines, or high-risk changes.
- Keep MCP/tools/plugins disabled unless they are needed for the current task.
- Prefer fresh sessions for unrelated work instead of carrying old context.
- Use `/status` to monitor context and rate-limit pressure during long sessions.

## Memory Updates
- When finishing meaningful work, offer or add a concise branch-prefixed summary to `.gemini/tmp/clintrialpredict/memory/MEMORY.md`.
- Include only: goal, decisions, files changed or inspected, tests/commands run, blockers, and next step.
- Keep memory entries short; never paste full code, large diffs, or long logs.
- When updating memory, correct or supersede stale entries rather than preserving misleading history as active context.
