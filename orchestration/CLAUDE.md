# Orchestration Rules

## This Directory
Schema-aware training data generation — teaches small models to orchestrate tools
by collecting and scoring real agent trajectories.

## Key Concept: Schema-Aware Training
Tool descriptions are embedded INSIDE instruction context. The model learns
*routing patterns*, not specific tool names. This makes fine-tuned models
more generalizable across different tool namespaces.

## Workflow (`orchestration_trainer.py` → `OrchestrationDataService`)
1. **Generate problems** — synthetic domain-aware problems with tool schemas embedded
2. **Collect trajectories** — run agent N times per problem at varying temperatures
3. **Score** — `OrchestrationRewardFunction`: accuracy (LLM-as-judge) + cost + latency
4. **Format** — convert to SFT (best), DPO (best vs worst), or GRPO (all + rewards)
5. **Train** — fine-tune via `finetuning_pipeline`
6. **Deploy** (optional) — via `hosting_pipeline`

## Key Classes
- `OrchestrationDataService` — main service; entry point for all above steps
- `TrajectoryRecorder` — records agent runs as `Trajectory` objects
- `Trajectory` — structured run: turns, tool calls, final_answer, metadata
- `TurnRecord` — per-turn: tool calls, token counts, latency (ms), cost ($)
- `OrchestrationRewardFunction` — multi-objective: accuracy + cost + latency

## Cost Model (hardcoded in `rewards.py`)
| Model | Rate |
|-------|------|
| gpt-4o | $0.005 / 1k tokens |
| gpt-4o-mini | $0.00015 / 1k tokens |
| o3-mini | $0.00110 / 1k tokens |
| "local" | $0 (free) |

## Output Formats
- **SFT**: best trajectory only
- **DPO**: best vs worst trajectory pair
- **GRPO**: all trajectories with reward scores

## Tests
`tests/orchestration/test_orchestration.py` — 20+ real tests, NOT stubs.
**Use this file as the reference style** for all new test files in the project.
Pattern: `fake_agent_run_events()` for mock agents, `@pytest.fixture` for shared state.

## Rules
- Orchestrator imports services — never imports pipeline internals directly
- `OrchestrationConfig` drives all hyperparameters (temperatures, N runs, budget)
- Reward weights are configurable in `OrchestrationConfig`, not hardcoded in `rewards.py`
