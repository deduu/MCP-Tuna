# Frontend MCP Remediation Matrix

## Scope
- Backend source of truth: `mcp_gateway.py` tool registrations.
- Frontend scope: hardcoded tool calls in `frontend/src`.
- Date: 2026-03-06.

## Coverage Snapshot
- Before remediation:
  - Hardcoded frontend tool IDs: 80
  - Matching gateway IDs: 54
  - Stale/non-existent IDs in frontend: 26
- After remediation:
  - Hardcoded frontend tool IDs: 69
  - Matching gateway IDs: 69
  - Stale/non-existent IDs in frontend: 0

## Tool Mapping (Old -> New)
| Old frontend call | New aligned call/flow | Updated in |
|---|---|---|
| `dataset.get_stats` | `dataset.info(file_path)` | `frontend/src/api/hooks/useDatasets.ts` |
| `dataset.export` | `dataset.info(file_path)` (informational export fallback) | `frontend/src/components/datasets/DatasetCard.tsx` |
| `dataset.delete` | `dataset.delete(file_path)` | `frontend/src/components/datasets/DatasetCard.tsx` |
| `host.get_status` | `host.health(deployment_id)` | `frontend/src/api/hooks/useDeployments.ts`, `frontend/src/components/deployments/DeploymentDetail.tsx` |
| `host.get_logs` | `host.health(deployment_id)` (rendered as JSON log line) | `frontend/src/api/hooks/useDeployments.ts` |
| `host.undeploy` | `host.stop(deployment_id)` | `frontend/src/api/hooks/useDeployments.ts` |
| `workflow.list_jobs` | Removed (no job API) | `frontend/src/api/hooks/usePipeline.ts` |
| `workflow.get_status` | Removed (no job API) | `frontend/src/api/hooks/usePipeline.ts` |
| `workflow.cancel` | Removed (no job API) | `frontend/src/api/hooks/usePipeline.ts` |
| `workflow.get_result` | Removed (use in-memory `job.result`) | `frontend/src/components/pipeline/PipelineJobCard.tsx` |
| `judge.get_config` | `judge.list_types` | `frontend/src/api/hooks/useEvaluation.ts` |
| `judge.list_criteria` | Removed (no direct list API) | `frontend/src/api/hooks/useEvaluation.ts` |
| `judge.configure_judge` | Removed; replaced with capabilities viewer | `frontend/src/components/evaluation/JudgeConfig.tsx` |
| `judge.calibrate` | Removed; replaced with capabilities viewer | `frontend/src/components/evaluation/JudgeConfig.tsx` |
| `judge.create_criterion` | `judge.create_rubric` | `frontend/src/components/evaluation/CriteriaManager.tsx` |
| `judge.batch_judge` | `dataset.load` -> `judge.evaluate_batch(test_data)` | `frontend/src/components/evaluation/JudgeTab.tsx` |
| `judge.evaluate_response` | `judge.evaluate` (`judge_type=pointwise`) | `frontend/src/components/evaluation/SingleEvalForm.tsx` |
| `judge.evaluate_with_rubric` | `judge.evaluate` (`judge_type=rubric`, parsed rubric JSON) | `frontend/src/components/evaluation/SingleEvalForm.tsx` |
| `judge.compare_responses` | `judge.compare_pair` | `frontend/src/components/evaluation/CompareForm.tsx` |
| `ft_eval.evaluate_finetune` | `dataset.load` -> optional `test.inference` -> `ft_eval.batch` | `frontend/src/components/evaluation/FtEvalTab.tsx` |
| `ft_eval.get_metrics` | `ft_eval.summary(results)` | `frontend/src/components/evaluation/FtEvalTab.tsx` |
| `ft_eval.compare_models` | `test.compare_models` for A/B model checks | `frontend/src/components/evaluation/FtEvalTab.tsx` |
| `evaluate_model.compare_models` | Removed; compare built from `evaluate_model.batch` runs | `frontend/src/components/evaluation/BenchmarkTab.tsx` |
| `evaluate_model.export_results` | `evaluate_model.export` | `frontend/src/components/evaluation/BenchmarkTab.tsx` |
| `ft_eval.export_results` | `ft_eval.export` | `frontend/src/components/evaluation/FtEvalTab.tsx` |
| `test.evaluate_output` | `test.inference` | `frontend/src/components/deployments/InferenceTest.tsx` |
| `orchestration.score_trajectories` | `orchestration.build_training_data` | `frontend/src/components/pipeline/OrchestrationStep.tsx` |
| `orchestration.format_dataset` | `orchestration.train_orchestrator` (step 4) | `frontend/src/components/pipeline/OrchestrationStep.tsx` |

## Parameter Contract Fixes
- `generate.get_schema` now always includes required `technique`.
- `generate.from_document` now uses `file_path` (not `document_path`).
- `generate.batch` now sends `file_paths: string[]`.
- `generate.from_page` now sends required `page_text`, `page_index`, `file_name`.
- Clean/Normalize/Evaluate flows now load datasets to `data_points` and pass arrays (not `dataset_path`).
- `dataset.split` now sends `output_dir`, `train_ratio`, `val_ratio`, `test_ratio`.
- `dataset.merge` now sends `output_path` (not `output_name`).
- `workflow.full_pipeline` now sends `file_path` and optional `base_model`.
- `workflow.run_pipeline` now sends JSON string of `{tool, params}` steps.

## Remaining Functional Gaps (Backend-Limited)
- No MCP workflow async job list/status/cancel APIs exist.
- No host logs API exists (only `host.health`).
