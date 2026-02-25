import pytest
from pathlib import Path
from finetuning_pipeline.services.pipeline_service import FineTuningService

# @pytest.mark.asyncio
# async def test_load_dataset_from_file_jsonl():
#     service = FineTuningService()

#     result = await service.load_dataset_from_file(
#         "./data/sft/data_sft.jsonl",
#         format="jsonl"
#     )
#     print(result)
#     assert result["success"]
#     assert result["num_examples"] > 0
#     assert "instruction" in result["columns"]
#     assert "input" in result["columns"]
#     assert "output" in result["columns"]

# @pytest.mark.asyncio
# async def test_run_inference_gpu():
#     service = FineTuningService()

#     result = await service.run_inference(
#         prompts=["Hello"],
#         model_path="meta-llama/Llama-3.2-3B-Instruct",
#         max_new_tokens=10,
#         do_sample=False
#     )
#     print(result)

#     assert result["success"]
#     assert len(result["results"]) == 1
#     assert isinstance(result["results"][0]["response"], str)

# @pytest.mark.asyncio
# async def test_train_model_real():
#     tmp_path = "./Llama-3.2-3B-Instruct-sft-output"
#     service = FineTuningService(
#         default_base_model="meta-llama/Llama-3.2-3B-Instruct"
#     )

#     dataset_result = await service.load_dataset_from_file(
#         "./data/sft/data_sft.jsonl",
#         format="jsonl"
#     )
    
#     dataset = dataset_result["dataset_object"]

#     result = await service.train_model(
#         dataset=dataset,
#         output_dir=str(tmp_path),
#         num_epochs=1,
#         enable_evaluation=False,

#     )

#     print(result)

# @pytest.mark.asyncio
# async def test_search_huggingface_models():
#     import json
#     service = FineTuningService()
#     result = await service.search_huggingface_models(query="qwen")
#     print(json.dumps(result, indent=2))
#     assert result["success"]
#     assert result["count"] > 0
#     assert "id" in result["models"][0]
#     assert "author" in result["models"][0]
#     assert "downloads" in result["models"][0]
#     assert "likes" in result["models"][0]
#     assert "tags" in result["models"][0]
#     assert "library" in result["models"][0]
#     assert "created_at" in result["models"][0]

@pytest.mark.asyncio
async def test_list_available_base_models():
    import json
    service = FineTuningService()
    result = await service.list_available_base_models(query="llama")
    print(json.dumps(result, indent=2))
    assert result["success"]
 
    assert "models" in result
    assert "id" in result["models"][0]
    assert "model_path" in result["models"][0]
    assert "usable_for" in result["models"][0]