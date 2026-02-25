# Testing Rules

## Framework
- pytest + pytest-asyncio (`asyncio_mode = "auto"` in `pyproject.toml`)
- All async tests: use `async def test_...` — no `@pytest.mark.asyncio` decorator needed

## Reference Implementation
`tests/orchestration/test_orchestration.py` is the gold standard for test style in this project.
Read it before writing any new test file. Key patterns to follow:
- `fake_agent_run_events()` generator for mock agents
- `@pytest.fixture` with `scope="function"` for isolated state
- Descriptive test names: `test_<what>_<condition>_<expected>`

## Structure
Test files mirror source files 1:1:
```
tests/
├── conftest.py                         # Shared fixtures: mock LLM, sample datapoints
├── orchestration/test_orchestration.py # Existing (real, 20+ tests)
├── data_generator_pipeline/            # To be created (TDD)
├── data_evaluator_pipeline/            # To be created (TDD)
├── finetuning_pipeline/                # To be fixed (many stubs currently)
├── app/                                # To be created (TDD)
└── shared/                             # To be created (TDD)
```

## Rules
1. **TDD.** Test file must exist with failing tests BEFORE the implementation file.
2. **Mock ALL external I/O.** LLM API calls, PostgreSQL, HuggingFace, file system, GPU.
3. **No real API calls.** Never hit OpenAI, Cohere, or HuggingFace in unit tests.
4. **No real GPU.** Mock `torch` operations in finetuning tests.
5. **No commented-out tests.** Delete unused tests rather than comment them out.
6. **One test file per source file.** `services/pipeline_service.py` → `tests/*/test_pipeline_service.py`

## Mocking DiagnosticWriter
`shared/diagnostics` uses a module-level singleton `_writer`. In tests, the writer is `None`
by default (since `init_diagnostics()` is never called), so all `emit_*` helpers are no-ops.
**No setup needed** — diagnostics are automatically silent in tests.

If you need to assert that events were emitted, patch the writer:
```python
from unittest.mock import AsyncMock, patch
import shared.diagnostics as diag

@pytest.fixture
def mock_writer(monkeypatch):
    writer = AsyncMock()
    monkeypatch.setattr(diag, "_writer", writer)
    return writer

async def test_emit_called(mock_writer):
    await diag.emit_request_start("hello", "claude-sonnet-4-6", "agent", True, [])
    mock_writer.emit.assert_awaited_once()
```

## Fixture Conventions (`conftest.py`)
```python
@pytest.fixture
def mock_llm():
    """Returns a BaseLLM mock that returns predictable JSON output."""
    ...

@pytest.fixture
def sample_sft_datapoint():
    """Returns a valid BaseDataPoint for test input."""
    ...

@pytest.fixture
def sample_dataset(sample_sft_datapoint):
    """Returns a list of 3 BaseDataPoint for batch testing."""
    ...
```

## Mocking LLMs
Use `fake_agent_run_events()` pattern from `tests/orchestration/test_orchestration.py`.
For service-level tests: mock the LLM at the `BaseLLM.generate()` level using `unittest.mock.AsyncMock`.
