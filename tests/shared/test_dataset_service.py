"""Tests for shared.dataset_service — TDD, written before implementation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.dataset_service import DatasetService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def svc() -> DatasetService:
    return DatasetService()


@pytest.fixture
def sft_data() -> list[dict]:
    return [
        {"instruction": "Explain gravity.", "input": "", "output": "Gravity is a force."},
        {"instruction": "What is water?", "input": "", "output": "Water is H2O."},
        {"instruction": "Describe light.", "input": "", "output": "Light is electromagnetic radiation."},
    ]


@pytest.fixture
def dpo_data() -> list[dict]:
    return [
        {"prompt": "Explain AI", "chosen": "AI is...", "rejected": "I don't know"},
        {"prompt": "What is ML?", "chosen": "ML is...", "rejected": "Not sure"},
    ]


@pytest.fixture
def grpo_data() -> list[dict]:
    return [
        {"prompt": "Solve x+1=3", "responses": ["x=2", "x=3"], "rewards": [1.0, 0.0]},
    ]


@pytest.fixture
def kto_data() -> list[dict]:
    return [
        {"prompt": "Hello", "completion": "Hi there!", "label": True},
        {"prompt": "Hello", "completion": "Go away", "label": False},
    ]


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

async def test_save_jsonl_creates_file(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "data.jsonl"
    result = await svc.save(sft_data, str(out), format="jsonl")
    assert result["success"] is True
    assert out.exists()
    assert result["row_count"] == 3
    assert result["dataset_id"] == "data"


async def test_save_json_creates_file(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "data.json"
    result = await svc.save(sft_data, str(out), format="json")
    assert result["success"] is True
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert len(loaded) == 3


async def test_save_parquet_creates_file(svc: DatasetService, sft_data: list, tmp_path: Path):
    pytest.importorskip("pandas")
    out = tmp_path / "data.parquet"
    result = await svc.save(sft_data, str(out), format="parquet")
    assert result["success"] is True
    assert out.exists()
    assert result["row_count"] == 3


async def test_save_creates_parent_dirs(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "nested" / "dir" / "data.jsonl"
    result = await svc.save(sft_data, str(out))
    assert result["success"] is True
    assert out.exists()


async def test_save_returns_correct_dataset_id(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "my_sft_data.jsonl"
    result = await svc.save(sft_data, str(out))
    assert result["dataset_id"] == "my_sft_data"


async def test_save_unsupported_format_returns_error(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "data.xml"
    result = await svc.save(sft_data, str(out), format="xml")
    assert result["success"] is False
    assert "format" in result["error"].lower()


async def test_save_empty_dataset(svc: DatasetService, tmp_path: Path):
    out = tmp_path / "empty.jsonl"
    result = await svc.save([], str(out))
    assert result["success"] is True
    assert result["row_count"] == 0


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

async def test_load_jsonl_returns_data_points(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "data.jsonl"
    await svc.save(sft_data, str(out))
    result = await svc.load(str(out))
    assert result["success"] is True
    assert len(result["data_points"]) == 3
    assert result["data_points"][0]["instruction"] == "Explain gravity."


async def test_load_json_returns_data_points(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "data.json"
    await svc.save(sft_data, str(out), format="json")
    result = await svc.load(str(out))
    assert result["success"] is True
    assert len(result["data_points"]) == 3


async def test_load_parquet_returns_data_points(svc: DatasetService, sft_data: list, tmp_path: Path):
    pytest.importorskip("pandas")
    out = tmp_path / "data.parquet"
    await svc.save(sft_data, str(out), format="parquet")
    result = await svc.load(str(out))
    assert result["success"] is True
    assert len(result["data_points"]) == 3


async def test_load_nonexistent_file_returns_error(svc: DatasetService):
    result = await svc.load("/nonexistent/path.jsonl")
    assert result["success"] is False
    assert "not found" in result["error"].lower() or "not exist" in result["error"].lower()


async def test_load_auto_detects_format(svc: DatasetService, sft_data: list, tmp_path: Path):
    """Save as .jsonl, load by path — format detected from extension."""
    out = tmp_path / "auto.jsonl"
    await svc.save(sft_data, str(out))
    result = await svc.load(str(out))
    assert result["success"] is True
    assert result["row_count"] == 3


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------

async def test_preview_returns_first_n_rows(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "data.jsonl"
    await svc.save(sft_data, str(out))
    result = await svc.preview(str(out), n=2)
    assert result["success"] is True
    assert len(result["rows"]) == 2
    assert result["total_rows"] == 3


async def test_preview_with_fewer_rows_than_n(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "data.jsonl"
    await svc.save(sft_data, str(out))
    result = await svc.preview(str(out), n=100)
    assert result["success"] is True
    assert len(result["rows"]) == 3


async def test_preview_returns_columns(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "data.jsonl"
    await svc.save(sft_data, str(out))
    result = await svc.preview(str(out))
    assert "columns" in result
    assert "instruction" in result["columns"]


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

async def test_info_returns_metadata(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "data.jsonl"
    await svc.save(sft_data, str(out))
    result = await svc.info(str(out))
    assert result["success"] is True
    meta = result["metadata"]
    assert meta["row_count"] == 3
    assert meta["dataset_id"] == "data"
    assert meta["size_bytes"] > 0


async def test_info_detects_sft_technique(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "sft.jsonl"
    await svc.save(sft_data, str(out))
    result = await svc.info(str(out))
    assert result["metadata"]["technique"] == "sft"


async def test_info_detects_dpo_technique(svc: DatasetService, dpo_data: list, tmp_path: Path):
    out = tmp_path / "dpo.jsonl"
    await svc.save(dpo_data, str(out))
    result = await svc.info(str(out))
    assert result["metadata"]["technique"] == "dpo"


async def test_info_detects_grpo_technique(svc: DatasetService, grpo_data: list, tmp_path: Path):
    out = tmp_path / "grpo.jsonl"
    await svc.save(grpo_data, str(out))
    result = await svc.info(str(out))
    assert result["metadata"]["technique"] == "grpo"


async def test_info_detects_kto_technique(svc: DatasetService, kto_data: list, tmp_path: Path):
    out = tmp_path / "kto.jsonl"
    await svc.save(kto_data, str(out))
    result = await svc.info(str(out))
    assert result["metadata"]["technique"] == "kto"


async def test_info_unknown_columns_returns_none_technique(svc: DatasetService, tmp_path: Path):
    data = [{"foo": "bar", "baz": 1}]
    out = tmp_path / "unknown.jsonl"
    await svc.save(data, str(out))
    result = await svc.info(str(out))
    assert result["metadata"]["technique"] is None


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

async def test_delete_removes_dataset_file(svc: DatasetService, sft_data: list, tmp_path: Path):
    out = tmp_path / "delete_me.jsonl"
    await svc.save(sft_data, str(out))

    result = await svc.delete(str(out))

    assert result["success"] is True
    assert result["deleted"] is True
    assert out.exists() is False


async def test_delete_missing_file_returns_error(svc: DatasetService, tmp_path: Path):
    result = await svc.delete(str(tmp_path / "missing.jsonl"))
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

async def test_split_creates_three_files(svc: DatasetService, tmp_path: Path):
    data = [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(20)]
    src = tmp_path / "data.jsonl"
    await svc.save(data, str(src))
    out_dir = tmp_path / "splits"

    result = await svc.split(str(src), str(out_dir), train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert result["success"] is True
    assert Path(result["splits"]["train"]["path"]).exists()
    assert Path(result["splits"]["val"]["path"]).exists()
    assert Path(result["splits"]["test"]["path"]).exists()


async def test_split_respects_ratios(svc: DatasetService, tmp_path: Path):
    data = [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(100)]
    src = tmp_path / "data.jsonl"
    await svc.save(data, str(src))
    out_dir = tmp_path / "splits"

    result = await svc.split(str(src), str(out_dir), train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    total = (
        result["splits"]["train"]["count"]
        + result["splits"]["val"]["count"]
        + result["splits"]["test"]["count"]
    )
    assert total == 100
    assert result["splits"]["train"]["count"] == 80


async def test_split_with_seed_is_reproducible(svc: DatasetService, tmp_path: Path):
    data = [{"instruction": f"q{i}", "input": "", "output": f"a{i}"} for i in range(50)]
    src = tmp_path / "data.jsonl"
    await svc.save(data, str(src))

    r1 = await svc.split(str(src), str(tmp_path / "s1"), seed=42)
    r2 = await svc.split(str(src), str(tmp_path / "s2"), seed=42)

    # Load and compare train splits
    t1 = await svc.load(r1["splits"]["train"]["path"])
    t2 = await svc.load(r2["splits"]["train"]["path"])
    assert t1["data_points"] == t2["data_points"]


async def test_split_invalid_ratios_returns_error(svc: DatasetService, sft_data: list, tmp_path: Path):
    src = tmp_path / "data.jsonl"
    await svc.save(sft_data, str(src))
    result = await svc.split(str(src), str(tmp_path / "out"), train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
    assert result["success"] is False


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

async def test_merge_combines_files(svc: DatasetService, tmp_path: Path):
    d1 = [{"instruction": "q1", "input": "", "output": "a1"}]
    d2 = [{"instruction": "q2", "input": "", "output": "a2"}]
    f1 = tmp_path / "d1.jsonl"
    f2 = tmp_path / "d2.jsonl"
    await svc.save(d1, str(f1))
    await svc.save(d2, str(f2))

    out = tmp_path / "merged.jsonl"
    result = await svc.merge([str(f1), str(f2)], str(out))
    assert result["success"] is True
    assert result["total_rows"] == 2


async def test_merge_with_deduplication(svc: DatasetService, tmp_path: Path):
    dup = {"instruction": "same", "input": "", "output": "answer"}
    f1 = tmp_path / "d1.jsonl"
    f2 = tmp_path / "d2.jsonl"
    await svc.save([dup], str(f1))
    await svc.save([dup], str(f2))

    out = tmp_path / "merged.jsonl"
    result = await svc.merge([str(f1), str(f2)], str(out), deduplicate=True, dedup_key="instruction")
    assert result["success"] is True
    assert result["total_rows"] == 1


async def test_merge_empty_file_list_returns_error(svc: DatasetService, tmp_path: Path):
    out = tmp_path / "merged.jsonl"
    result = await svc.merge([], str(out))
    assert result["success"] is False
