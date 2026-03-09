import pytest

from data_cleaning_pipeline.services.cleaning_service import DataCleaningService


@pytest.mark.asyncio
async def test_clean_dataset_reports_step_breakdown():
    service = DataCleaningService()
    data_points = [
        {"instruction": "alpha instruction", "output": "valid output that is long enough"},
        {"instruction": "alpha instruction", "output": "valid output that is long enough"},
        {"instruction": "beta", "output": ""},
        {"instruction": "short", "output": "tiny"},
    ]

    result = await service.clean_dataset(data_points)

    assert result["original_count"] == 4
    assert result["cleaned_count"] == 1
    assert result["removed"] == 3
    assert result["unchanged"] is False
    assert result["steps"]["remove_empty_fields"]["removed"] == 1
    assert result["steps"]["deduplicate"]["removed"] == 1
    assert result["steps"]["remove_short_entries"]["removed"] == 1
