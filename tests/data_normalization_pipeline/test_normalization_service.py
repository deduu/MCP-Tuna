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


@pytest.mark.asyncio
async def test_remap_fields_converts_chat_triplet_to_sft():
    service = DataNormalizationService()

    result = await service.remap_fields([
        {
            "system": "  You are concise.  ",
            "user": "  Explain Salestify. ",
            "assistant": " It helps sales teams manage WhatsApp chats. ",
        }
    ])

    assert result["success"] is True
    assert result["target_format"] == "sft"
    assert result["changed_rows"] == 1
    assert result["created_fields"] == ["instruction", "input", "output"]
    assert result["dropped_fields"] == ["assistant", "system", "user"]
    assert result["data_points"] == [
        {
            "instruction": "System: You are concise.\n\nUser: Explain Salestify.",
            "input": "",
            "output": "It helps sales teams manage WhatsApp chats.",
        }
    ]


@pytest.mark.asyncio
async def test_remap_fields_keeps_extra_fields_when_requested():
    service = DataNormalizationService()

    result = await service.remap_fields(
        [{"prompt": "Hi", "response": "Hello", "source": "demo"}],
        preset="prompt_response_to_sft",
        keep_unmapped_fields=True,
    )

    assert result["success"] is True
    assert result["dropped_fields"] == []
    assert result["data_points"][0]["instruction"] == "Hi"
    assert result["data_points"][0]["output"] == "Hello"
    assert result["data_points"][0]["source"] == "demo"
