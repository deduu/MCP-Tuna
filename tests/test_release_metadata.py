from __future__ import annotations

from pathlib import Path


PUBLIC_METADATA_FILES = [
    "README.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "server.json",
    "docker-compose.yml",
]


def test_public_metadata_has_no_legacy_product_names() -> None:
    legacy_terms = ["transcendence-gateway", "transcendence-data", "agenty"]
    repo_root = Path(__file__).resolve().parent.parent

    for relative_path in PUBLIC_METADATA_FILES:
        text = (repo_root / relative_path).read_text(encoding="utf-8").lower()
        for term in legacy_terms:
            assert term not in text, f"{term} leaked into {relative_path}"
