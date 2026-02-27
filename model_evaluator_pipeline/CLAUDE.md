# Model Evaluator Pipeline Rules

## This Directory
Post-training model output quality evaluation. Scores generated outputs against
ground truth references using ROUGE, BERTScore, and LLM-as-Judge metrics.

**Not to be confused with `data_evaluator_pipeline/`** which evaluates
*dataset quality before training*. This pipeline evaluates *model output quality
after training*.

## Structure
```
model_evaluator_pipeline/
├── __init__.py
├── CLAUDE.md
├── metrics/
│   ├── __init__.py
│   ├── rouge.py          # ROUGE-1/2/L/Lsum via HF evaluate
│   ├── bertscore.py      # BERTScore P/R/F1 via bert-score
│   ├── llm_judge.py      # 5-criteria LLM-as-Judge (1-10 scores)
│   └── perplexity.py     # Perplexity via OpenAI logprobs
├── models/
│   ├── __init__.py
│   └── ft_evaluator.py   # Pydantic models: Verdict, FailureType, FTEvalResult, etc.
├── prompts/
│   ├── __init__.py
│   └── ft_evaluator_prompt.md  # Domain Knowledge Judge system prompt
├── judges/                # Advanced LLM-as-a-Judge system
│   └── ...
└── services/
    ├── __init__.py
    ├── evaluation_service.py      # Step 1: inference + scoring (ROUGE/BERT/Judge/Perplexity)
    ├── ft_evaluator_service.py    # Step 2: domain knowledge PASS/FAIL + K-Type taxonomy
    └── judge_service.py           # Advanced multi-judge orchestration
```

## Metrics

| Metric | Function | Returns |
|--------|----------|---------|
| ROUGE | `compute_rouge(gen, ref)` | `{rouge1, rouge2, rougeL, rougeLsum}` |
| BERTScore | `compute_bertscore(gen, ref)` | `{bert_precision, bert_recall, bert_f1}` |
| LLM Judge | `llm_judge(q, gen, ref, llm)` | 5 criteria × `{score: 1-10, reason}` |
| Perplexity | `compute_perplexity(q, gen, ref, model)` | `{perplexity, mean_logprob, num_tokens}` |

## LLM-as-Judge Criteria
correctness, completeness, factuality, structure, hallucination_resistance
Each scored 1-10 with a reason string.

## Service API (`services/evaluation_service.py`) — Step 1
- `evaluate_single(question, generated, reference, metrics)` → scored result
- `evaluate_batch(test_data, metrics, model_path?, adapter_path?, flatten?)` → all rows + summary
- `flatten_result(result)` → flat dict (nested scores → top-level keys)
- `compute_summary(results)` → per-metric {min, max, mean, stdev}
- `export_results(results, path, format)` → JSONL / JSON / Excel

## Service API (`services/ft_evaluator_service.py`) — Step 2
- `evaluate_single(instruction, generated, reference, judge_model?, ksmi_label?)` → FTEvalVerdict
- `evaluate_multi_judge(instruction, generated, reference, ksmi_label?, judge_models?)` → FTEvalResult
- `evaluate_batch(test_data, judge_models?)` → {success, results[], summary, count}
- `compute_summary(results)` → FTEvalSummary
- `export_results(results, output_path, format)` → {success, output_path, format, num_results}

## MCP Tools
`evaluate_model.single`, `evaluate_model.batch`, `evaluate_model.export`, `evaluate_model.summary`
`ft_eval.single`, `ft_eval.batch`, `ft_eval.summary`, `ft_eval.export`

## Config (`shared/config.py`)
```python
ModelEvaluationConfig(
    model="gpt-4o",         # LLM for judge
    metrics=["rouge", "bertscore", "llm_judge"],
    max_new_tokens=1024,
    temperature=0.1,
    judge_model="gpt-4o",
    bertscore_model="roberta-large",
    api_key=None,            # For perplexity metric (falls back to env)
    api_base=None,
)

FTEvaluatorConfig(
    judge_models=["gpt-4o"],
    temperature=0.0,
    max_tokens=2048,
    system_prompt_path=None,  # Override bundled prompt
)
```

## Dependencies
`evaluate>=0.4`, `bert-score>=0.3`, `rouge-score>=0.1`, `openpyxl>=3.1`
(in `[project.optional-dependencies] eval`)

## Rules
- Metrics are async functions (not classes) — they offload heavy compute to threads
- LLM judge scores are clamped to [1, 10]; parse failures return zeros
- Perplexity uses direct `AsyncOpenAI` (not agentsoul's BaseLLM) — logprobs not exposed by BaseLLM
- FTEvaluatorService uses direct `AsyncOpenAI` with `response_format={"type": "json_object"}`
- Service can optionally run inference via `finetuning_pipeline.InferenceService`
  when `model_path` is provided (MCP boundary: direct import is acceptable here
  because both are in the same process when called from the gateway)
- Tests mock all external I/O: `bert_score.score`, `evaluate.load`, LLM calls, `AsyncOpenAI`
