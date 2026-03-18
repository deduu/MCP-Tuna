from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetuning_pipeline.services.model_discovery_service import ModelDiscoveryService


@pytest.fixture
def svc() -> ModelDiscoveryService:
    return ModelDiscoveryService()


def test_get_model_info_reports_vlm_capabilities(svc: ModelDiscoveryService, tmp_path: Path):
    model_dir = tmp_path / "qwen2.5-vl"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_5_vl",
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "vision_config": {"image_size": 448},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    result = svc.get_model_info(str(model_dir))

    assert result["success"] is True
    assert result["modality"] == "vision-language"
    assert result["supports_training"] is True
    assert result["supports_inference"] is True
    assert result["supports_deployment"] is True
    assert result["supported_techniques"] == ["vlm_sft"]


@pytest.mark.asyncio
async def test_list_available_base_models_includes_capabilities(svc: ModelDiscoveryService, tmp_path: Path):
    hub_dir = tmp_path / "hub" / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / "1234"
    hub_dir.mkdir(parents=True)
    (hub_dir / "config.json").write_text("{}", encoding="utf-8")
    (hub_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    result = await svc.list_available_base_models(hf_home=str(tmp_path))

    assert result["success"] is True
    assert len(result["models"]) == 1
    model = result["models"][0]
    assert model["modality"] == "vision-language"
    assert model["supported_techniques"] == ["vlm_sft"]
