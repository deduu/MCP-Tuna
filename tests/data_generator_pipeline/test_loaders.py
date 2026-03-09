import json

import pytest

from data_generator_pipeline.loaders import JsonlPageLoader


def test_jsonl_page_loader_uses_text_field(tmp_path):
    file_path = tmp_path / "generated_dataset.jsonl"
    rows = [
        {"instruction": "Q1", "output": "A1", "text": "First source page"},
        {"instruction": "Q2", "output": "A2", "text": "Second source page"},
    ]
    file_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    file_name, pages = JsonlPageLoader().load(str(file_path))

    assert file_name == "generated_dataset"
    assert pages == [
        {"index": 0, "markdown": "First source page"},
        {"index": 1, "markdown": "Second source page"},
    ]


def test_jsonl_page_loader_requires_text_like_field(tmp_path):
    file_path = tmp_path / "bad_dataset.jsonl"
    file_path.write_text(
        json.dumps({"instruction": "Q1", "output": "A1"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="JSONL document rows must include one of"):
        JsonlPageLoader().load(str(file_path))
