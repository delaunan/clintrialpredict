# Clinical Trial Prediction: Shared Agent Instructions (v56.0)

This file is the **Single Source of Truth** for all AI agents (Gemini CLI, Antigravity/agy, Codex CLI).

## 1. Project Context
- **Goal**: Transitioning from historical discovery (v1.0 "Steel Shield") to interactive strategic forecasting (v2.0 "Strategic Forecaster").
- **Technical Reference**: Detailed architecture, simulation roadmaps, and artifact registries are maintained in [ARCHITECTURAL_LOG.md](./ARCHITECTURAL_LOG.md).

## 2. Agent Operational Guidelines

### Default Behavior
- **Read-Only**: Treat the repository as read-only unless explicit approval is given for a specific code edit.
- **Surgical Changes**: Do not create, modify, or move files unless explicitly asked.
- **Pre-Flight Inspection**: Always inspect relevant files and propose a minimal plan before making any change.

### Advisory & Planning Protocols
- **Advisory Mode**: Default to explaining and reviewing risks before editing.
- **Planning Mode**: Strictly follow the Planning Mode workflow. Create/update `implementation_plan.md` and obtain user approval before execution.

### Parity & Regression Safeguards
Before staging or committing any code edits to model coefficients, categories, preprocess pipeline scripts (`src/prep/`), or scoring functions:
1. Re-generate the local Streamlit database: `python refresh_registry.py`
2. Run the mathematical audit verification: `python audit_parity.py`
3. **Requirement**: 100% Perfect Parity must be achieved before deployment.

## 3. Multi-CLI Memory Strategy
- All CLIs share this `GEMINI.md` for instructions.
- Project-specific private notes are stored in `.gemini/tmp/clintrialpredict/memory/MEMORY.md`.
- **Branch-Aware Memory**: To prevent context contamination, all agents MUST prefix every entry in `MEMORY.md` with the current Git branch name (e.g., `[branch-name]: progress note`). When reading `MEMORY.md`, agents MUST prioritize entries matching the current branch and treat others as historical or out-of-scope.
- Use shared memory for cross-turn persistence to avoid duplicating session info.
