from __future__ import annotations

from shared.dpo_training_diagnostics import (
    serialize_config_object,
    summarize_dpo_preprocessing,
    summarize_dpo_trainer_dataset,
)


class _WhitespaceTokenizer:
    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        return {"input_ids": [index for index, _ in enumerate(str(text).split(), start=1)]}


def test_summarize_dpo_preprocessing_reports_token_overflow_ratios():
    dataset = [
        {
            "prompt": "satu dua tiga empat lima enam",
            "chosen": "alpha beta gamma delta epsilon",
            "rejected": "pendek sekali",
        },
        {
            "prompt": "pendek prompt",
            "chosen": "jawaban ringkas",
            "rejected": "jawaban yang lebih panjang sedikit",
        },
    ]

    summary = summarize_dpo_preprocessing(
        dataset,
        _WhitespaceTokenizer(),
        max_prompt_length=4,
        max_length=7,
        sample_limit=2,
    )

    assert summary["sample_size"] == 2
    assert summary["prompt_overflow_ratio"] == 0.5
    assert summary["chosen_response_overflow_ratio"] == 0.5
    assert summary["combined_chosen_overflow_ratio"] == 0.5
    assert len(summary["samples"]) == 2
    assert summary["samples"][0]["prompt_overflow"] is True


def test_serialize_config_object_prefers_to_dict():
    class _Config:
        def to_dict(self):
            return {"alpha": 1, "nested": {"beta": [1, 2]}}

    assert serialize_config_object(_Config()) == {"alpha": 1, "nested": {"beta": [1, 2]}}


def test_summarize_dpo_trainer_dataset_reports_postprocessed_columns():
    dataset = [
        {
            "prompt_input_ids": [1, 2, 3],
            "chosen_input_ids": [4, 5],
            "rejected_input_ids": [6],
            "prompt": "hello world",
        }
    ]

    class _Dataset:
        column_names = [
            "prompt_input_ids",
            "chosen_input_ids",
            "rejected_input_ids",
            "prompt",
        ]

        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            return dataset[index]

    summary = summarize_dpo_trainer_dataset(_Dataset())

    assert summary["num_rows"] == 1
    assert "prompt_input_ids" in summary["column_names"]
    sample = summary["samples"][0]["fields"]
    assert sample["prompt_input_ids"]["length"] == 3
    assert sample["chosen_input_ids"]["preview"] == [4, 5]
