import pytest

from data_normalization_pipeline.services.normalization_service import DataNormalizationService


@pytest.mark.asyncio
async def test_normalize_dataset_reports_change_breakdown():
    service = DataNormalizationService()
    data_points = [
        {
            "instruction": "  What is KSMI?  ",
            "input": " Domain context ",
            "response": "A framework.",
        },
        {
            "instruction": "Already clean",
            "input": "",
            "output": "Stable output",
        },
    ]

    result = await service.normalize_dataset(data_points)

    assert result["count"] == 2
    assert result["target_format"] == "sft"
    assert result["changed_rows"] == 1
    assert result["unchanged"] is False
    assert result["steps"]["strip_text"]["changed_fields"] == 2
    assert result["steps"]["merge_fields"]["merged_rows"] == 1
    assert result["steps"]["standardize_keys"]["renamed_fields"] == 1
