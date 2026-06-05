# Clinical Trial Prediction: Shared Agent Instructions (v56.0)

This file is the **Single Source of Truth** for all AI agents (Gemini CLI, Antigravity/agy, Codex CLI).

## 1. Project Context
- **Goal**: Transitioning from historical discovery (v1.0 "Steel Shield") to interactive strategic forecasting (v2.0 "Strategic Forecaster").
- **Technical Reference**: Detailed architecture, simulation roadmaps, and artifact registries are maintained in [ARCHITECTURAL_LOG.md](./ARCHITECTURAL_LOG.md).

## 2. Agent Operational Guidelines

### Request Routing
- Classify every non-trivial request before acting: `core_scoring`, `trial_audit`, `trial_edit`, `trial_simulator`, `operational_estimation`, `narratives_llm`, `deployment`, or `docs_memory`.
- Use the classification to choose the smallest architecture file to inspect or update:
  - `core_scoring`, shared artifacts, variants, deployment topology: `ARCHITECTURAL_LOG.md`
  - `trial_edit` / `trial_simulator` UI and live `/predict` workflow: `docs/architecture_edit.md`
  - `operational_estimation`: `docs/architecture_estimation.md`
  - `narratives_llm`: `docs/architecture_narratives.md`
  - agent behavior and memory policy: `GEMINI.md` / `AGENTS.md`
- When a request crosses scopes, name the primary scope first and update secondary docs only for durable cross-scope decisions.

### Default Behavior
- **Read-Only**: Treat the repository as read-only unless explicit approval is given for a specific code edit.
- **Surgical Changes**: Do not create, modify, or move files unless explicitly asked.
- **Pre-Flight Inspection**: Always inspect relevant files and propose a minimal plan before making any change.

### Advisory & Planning Protocols
- **Advisory Mode**: Default to explaining and reviewing risks before editing.
- **Planning Mode**: Strictly follow the Planning Mode workflow. Create/update `implementation_plan.md` and obtain user approval before execution.
- **Reply Quality**: Final replies should state the architecture scope, files changed, verification run, remaining risk, and next action. Do not restate long history unless the user asks.

### Parity & Regression Safeguards
Before staging or committing any code edits to model coefficients, categories, preprocess pipeline scripts (`src/prep/`), or scoring functions:
1. Re-generate the local Streamlit database: `python refresh_registry.py`
2. Run the mathematical audit verification: `python audit_parity.py`
3. **Requirement**: 100% Perfect Parity must be achieved before deployment.

## 3. Multi-CLI Memory Strategy
- All CLIs share this `GEMINI.md` for instructions.
- Project-specific private notes are stored in `.gemini/tmp/clintrialpredict/memory/MEMORY.md`.
- **Architecture-First Memory**: Memory entries MUST start with the architecture scope, then the branch context, using `[scope][branch] YYYY-MM-DD: ...`. Example: `[architecture_estimation][trial-edit] 2026-06-05: ...`.
- Memory is a short handoff layer, not the architecture source of truth. Each meaningful entry should identify the relevant architecture scope/file, such as `architecture_edit`, `architecture_estimation`, `architecture_narratives`, `deployment`, `branch_history`, or `ARCHITECTURAL_LOG`.
- Durable decisions belong in the relevant architecture file. Memory should only summarize the latest goal, decisions, changed/inspected files, verification, blockers, and next step.
- When reading memory, prioritize entries matching the relevant architecture scope first, then current branch. Treat branch context as provenance, not ownership of the architecture.
- If memory and architecture docs conflict, trust current code/tests first, then the relevant architecture file, then the newest matching memory entry.
