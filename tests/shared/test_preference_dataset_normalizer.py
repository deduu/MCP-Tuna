from __future__ import annotations

from shared.preference_dataset_normalizer import normalize_preference_dataset


def test_normalize_preference_dataset_trims_dpo_fields_and_preserves_extra_columns():
    rows = [
        {
            "prompt": "Prompt with trailing space  \n",
            "chosen": " Chosen answer ",
            "rejected": "Rejected answer\t",
            "metadata": {"source": "test"},
        }
    ]

    result = normalize_preference_dataset(rows, "dpo")

    assert result.dataset[0]["prompt"] == "Prompt with trailing space"
    assert result.dataset[0]["chosen"] == "Chosen answer"
    assert result.dataset[0]["rejected"] == "Rejected answer"
    assert result.dataset[0]["metadata"] == {"source": "test"}
    assert result.summary["trimmed_row_count"] == 1
    assert result.summary["trimmed_scalar_value_count"] == 3
    assert result.summary["preserved_extra_columns"] == ["metadata"]


def test_normalize_preference_dataset_flags_invalid_kto_rows_after_strip():
    rows = [
        {"prompt": "  ", "completion": "valid", "label": True},
        {"prompt": "ok", "completion": "\n", "label": False},
    ]

    result = normalize_preference_dataset(rows, "kto")

    assert result.summary["invalid_row_count"] == 2
    assert result.summary["invalid_rows"][0]["index"] == 0
    assert "prompt: empty after strip" in result.summary["invalid_rows"][0]["issues"]
    assert "completion: empty after strip" in result.summary["invalid_rows"][1]["issues"]


def test_normalize_preference_dataset_trims_grpo_response_variants():
    rows = [
        {
            "prompt": " Pertanyaan ",
            "responses": [" Bagus ", " Buruk\t"],
            "rewards": [1.0, 0.0],
        }
    ]

    result = normalize_preference_dataset(rows, "grpo")

    assert result.dataset[0]["prompt"] == "Pertanyaan"
    assert result.dataset[0]["responses"] == ["Bagus", "Buruk"]
    assert result.summary["trimmed_scalar_value_count"] == 1
    assert result.summary["trimmed_list_value_count"] == 2
