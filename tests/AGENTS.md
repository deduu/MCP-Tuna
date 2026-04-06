# AGENTS.md

Rules for tests in `tests/`.

- Mirror source layout when adding new tests.
- Use `tests/orchestration/test_orchestration.py` as the reference style for fixtures, async tests, and naming.
- Mock all external I/O: LLMs, HuggingFace, databases, filesystem writes, and GPU operations.
- Prefer `async def test_...` with the existing pytest configuration instead of extra decorators.
- Add or update the smallest test slice that proves the behavioral change.
- Keep tests focused on observable behavior and boundary contracts, not implementation trivia.
