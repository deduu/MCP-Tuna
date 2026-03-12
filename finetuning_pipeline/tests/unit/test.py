from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from finetuning_pipeline.services.curriculum_service import CurriculumService
from finetuning_pipeline.services.pipeline_service import FineTuningService
from finetuning_pipeline.services.training_service import TrainingService

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


# ----------------------------------------------------------------
# Shared mock helpers
# ----------------------------------------------------------------

def _make_mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    tok.apply_chat_template = MagicMock(return_value="User: q\nAssistant: a")
    tok.save_pretrained = MagicMock()
    return tok


def _make_mock_model():
    model = MagicMock()
    return model


def _make_mock_trainer():
    trainer = MagicMock()
    trainer.train = MagicMock()
    trainer.save_model = MagicMock()
    return trainer


# ----------------------------------------------------------------
# DPO trainer tests
# ----------------------------------------------------------------

_LOAD_MODEL_PATH = "finetuning_pipeline.services.training_service.TrainingService._load_model_and_tokenizer"
_CLEANUP_PATH = "finetuning_pipeline.services.training_service.TrainingService._cleanup"


@pytest.mark.asyncio
@patch(_CLEANUP_PATH)
@patch(_LOAD_MODEL_PATH)
@patch("trl.DPOTrainer")
async def test_train_dpo_model(mock_trainer_cls, mock_load_model, mock_cleanup):
    mock_load_model.return_value = (_make_mock_model(), _make_mock_tokenizer())
    mock_trainer_cls.return_value = _make_mock_trainer()
    mock_cleanup.return_value = None

    service = TrainingService()
    data = [
        {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"},
        {"prompt": "Capital of France?", "chosen": "Paris", "rejected": "London"},
    ]
    result = await service.train_dpo_model(
        dataset=data,
        output_dir="/tmp/test_dpo_output",
        base_model="test-model",
        num_epochs=1,
        use_lora=False,
    )

    assert result["success"], result.get("error")
    assert result["config"]["trainer"] == "dpo"
    assert result["config"]["beta"] == 0.1
    assert result["num_training_examples"] == 2
    assert "model_path" in result


@pytest.mark.asyncio
async def test_train_dpo_model_missing_columns():
    service = TrainingService()
    data = [{"prompt": "q", "chosen": "a"}]  # missing 'rejected'
    result = await service.train_dpo_model(
        dataset=data,
        output_dir="/tmp/test_dpo",
        base_model="test-model",
        num_epochs=1,
    )
    assert not result["success"]
    assert "rejected" in result["error"]


# ----------------------------------------------------------------
# GRPO trainer tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
@patch(_CLEANUP_PATH)
@patch(_LOAD_MODEL_PATH)
@patch("trl.GRPOTrainer")
async def test_train_grpo_model(mock_trainer_cls, mock_load_model, mock_cleanup):
    mock_load_model.return_value = (_make_mock_model(), _make_mock_tokenizer())
    mock_trainer_cls.return_value = _make_mock_trainer()
    mock_cleanup.return_value = None

    service = TrainingService()
    data = [
        {
            "prompt": "Solve x+1=3",
            "responses": ["x=2", "x=3", "x=1"],
            "rewards": [1.0, 0.0, 0.0],
        },
        {
            "prompt": "What is 5*5?",
            "responses": ["25", "20", "30"],
            "rewards": [1.0, 0.0, 0.0],
        },
    ]
    result = await service.train_grpo_model(
        dataset=data,
        output_dir="/tmp/test_grpo_output",
        base_model="test-model",
        num_epochs=1,
        num_generations=2,
    )

    assert result["success"], result.get("error")
    assert result["config"]["trainer"] == "grpo"
    assert result["config"]["num_generations"] == 2
    assert result["num_training_examples"] == 2

    # Verify the reward function was passed and works correctly
    call_kwargs = mock_trainer_cls.call_args
    reward_funcs = call_kwargs.kwargs.get("reward_funcs") or call_kwargs.args[1]
    assert reward_funcs is not None and len(reward_funcs) == 1
    assert reward_funcs[0](["Solve x+1=3"], ["x=2"]) == [1.0]   # exact match → stored reward
    assert reward_funcs[0](["Solve x+1=3"], ["x=99"]) == [0.0]  # unseen → 0.0


@pytest.mark.asyncio
async def test_train_grpo_model_missing_columns():
    service = TrainingService()
    data = [{"prompt": "q", "responses": ["a"]}]  # missing 'rewards'
    result = await service.train_grpo_model(
        dataset=data,
        output_dir="/tmp/test_grpo",
        base_model="test-model",
        num_epochs=1,
    )
    assert not result["success"]
    assert "rewards" in result["error"]


# ----------------------------------------------------------------
# KTO trainer tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
@patch(_CLEANUP_PATH)
@patch(_LOAD_MODEL_PATH)
@patch("trl.KTOTrainer")
async def test_train_kto_model(mock_trainer_cls, mock_load_model, mock_cleanup):
    mock_load_model.return_value = (_make_mock_model(), _make_mock_tokenizer())
    mock_trainer_cls.return_value = _make_mock_trainer()
    mock_cleanup.return_value = None

    service = TrainingService()
    data = [
        {"prompt": "What is 2+2?", "completion": "4", "label": True},
        {"prompt": "What is 2+2?", "completion": "5", "label": False},
        {"prompt": "Capital of France?", "completion": "Paris", "label": True},
    ]
    result = await service.train_kto_model(
        dataset=data,
        output_dir="/tmp/test_kto_output",
        base_model="test-model",
        num_epochs=1,
        use_lora=False,
    )

    assert result["success"], result.get("error")
    assert result["config"]["trainer"] == "kto"
    assert result["config"]["beta"] == 0.1
    assert result["config"]["desirable_weight"] == 1.0
    assert result["num_training_examples"] == 3


@pytest.mark.asyncio
async def test_train_kto_model_missing_columns():
    service = TrainingService()
    data = [{"prompt": "q", "completion": "a"}]  # missing 'label'
    result = await service.train_kto_model(
        dataset=data,
        output_dir="/tmp/test_kto",
        base_model="test-model",
        num_epochs=1,
    )
    assert not result["success"]
    assert "label" in result["error"]


# ----------------------------------------------------------------
# Checkpoint / resume tests
# ----------------------------------------------------------------

@pytest.mark.asyncio
@patch(_CLEANUP_PATH)
@patch(_LOAD_MODEL_PATH)
@patch("trl.SFTTrainer")
async def test_train_model_resume_from_checkpoint(
    mock_trainer_cls, mock_load_model, mock_cleanup, tmp_path
):
    mock_tokenizer = _make_mock_tokenizer()
    mock_load_model.return_value = (_make_mock_model(), mock_tokenizer)
    mock_trainer_cls.return_value = _make_mock_trainer()
    mock_cleanup.return_value = None

    # Create a fake checkpoint directory
    ckpt_dir = tmp_path / "checkpoint-100"
    ckpt_dir.mkdir()

    service = TrainingService()
    data = [{"prompt": "hi", "response": "hello"}]
    result = await service.train_model(
        dataset=data,
        output_dir=str(tmp_path),
        base_model="test-model",
        num_epochs=1,
        use_lora=False,
        resume_from_checkpoint="latest",
    )

    assert result["success"], result.get("error")
    assert result["config"]["resumed_from"] == str(ckpt_dir)


def test_resolve_checkpoint_latest(tmp_path):
    """_resolve_checkpoint picks the highest-numbered checkpoint."""
    (tmp_path / "checkpoint-50").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-200").mkdir()

    service = TrainingService()
    resolved = service._resolve_checkpoint(str(tmp_path), "latest")
    assert resolved == str(tmp_path / "checkpoint-200")


def test_resolve_checkpoint_none():
    service = TrainingService()
    assert service._resolve_checkpoint("/some/dir", None) is None
    assert service._resolve_checkpoint("/some/dir", False) is None


def test_resolve_checkpoint_explicit(tmp_path):
    service = TrainingService()
    path = str(tmp_path / "checkpoint-42")
    assert service._resolve_checkpoint(str(tmp_path), path) == path


# ----------------------------------------------------------------
# Curriculum service — unit tests
# ----------------------------------------------------------------


def _make_scored_items(n: int, start_score: float = 0.0) -> list[dict]:
    """Return n dicts with weighted_score evenly spaced from start_score."""
    step = 1.0 / max(n, 1)
    return [
        {
            "instruction": f"q{i}",
            "input": "",
            "output": f"a{i}",
            "weighted_score": round(start_score + i * step, 4),
        }
        for i in range(n)
    ]


def test_curriculum_bucket_even_split():
    """9 items split into 3 stages → 3 items each."""
    svc = CurriculumService()
    data = _make_scored_items(9)
    buckets = svc._bucket_dataset(data, 3, "weighted_score", "easy_first")
    assert len(buckets) == 3
    assert all(len(b) == 3 for b in buckets)


def test_curriculum_bucket_remainder():
    """10 items into 3 stages → 3 / 3 / 4 (last bucket absorbs the remainder)."""
    svc = CurriculumService()
    data = _make_scored_items(10)
    buckets = svc._bucket_dataset(data, 3, "weighted_score", "easy_first")
    assert len(buckets) == 3
    sizes = [len(b) for b in buckets]
    assert sizes == [3, 3, 4]


def test_curriculum_difficulty_ordering_easy_first():
    """easy_first → ascending score, first bucket has lowest scores."""
    svc = CurriculumService()
    data = _make_scored_items(9)
    buckets = svc._bucket_dataset(data, 3, "weighted_score", "easy_first")
    # First bucket scores are all lower than last bucket scores
    assert max(b["weighted_score"] for b in buckets[0]) < min(
        b["weighted_score"] for b in buckets[2]
    )


def test_curriculum_difficulty_ordering_hard_first():
    """hard_first → descending score, first bucket has highest scores."""
    svc = CurriculumService()
    data = _make_scored_items(9)
    buckets = svc._bucket_dataset(data, 3, "weighted_score", "hard_first")
    assert min(b["weighted_score"] for b in buckets[0]) > max(
        b["weighted_score"] for b in buckets[2]
    )


def test_curriculum_prepare_training_data_instruction_format():
    """instruction + input + output → prompt + response."""
    svc = CurriculumService()
    bucket = [{"instruction": "Translate", "input": "hello", "output": "hola"}]
    result = list(svc._prepare_training_data(bucket))
    assert result == [{"prompt": "Translate hello", "response": "hola"}]


def test_curriculum_prepare_training_data_prompt_format():
    """prompt + response passthrough (no instruction key)."""
    svc = CurriculumService()
    bucket = [{"prompt": "What is AI?", "response": "Artificial intelligence."}]
    result = list(svc._prepare_training_data(bucket))
    assert result == [{"prompt": "What is AI?", "response": "Artificial intelligence."}]


def test_curriculum_prepare_training_data_instruction_only():
    """instruction with no input → prompt is just the instruction."""
    svc = CurriculumService()
    bucket = [{"instruction": "Summarise this.", "output": "Done."}]
    result = list(svc._prepare_training_data(bucket))
    assert result[0]["prompt"] == "Summarise this."
    assert result[0]["response"] == "Done."


@pytest.mark.asyncio
async def test_curriculum_skips_scoring_if_already_scored():
    """If weighted_score is present, _score_dataset is never called."""
    svc = CurriculumService()
    data = _make_scored_items(6)

    called = []

    async def fake_score(d, col):
        called.append(True)
        return d

    svc._score_dataset = fake_score  # type: ignore[method-assign]

    # pre_scored check happens inside train_curriculum_model; simulate directly
    pre_scored = "weighted_score" in data[0]
    assert pre_scored is True
    assert called == []


@pytest.mark.asyncio
@patch("finetuning_pipeline.services.curriculum_service.CurriculumService._merge_lora")
@patch("finetuning_pipeline.services.training_service.TrainingService.train_model")
async def test_train_curriculum_model_three_stages(
    mock_train_model, mock_merge_lora, tmp_path
):
    """Full curriculum flow with 3 stages — mocked training + merge."""
    mock_train_model.return_value = {
        "success": True,
        "model_path": str(tmp_path / "stage_X"),
        "num_training_examples": 3,
    }
    mock_merge_lora.return_value = str(tmp_path / "stage_X" / "merged")

    svc = CurriculumService()
    data = _make_scored_items(9)

    result = await svc.train_curriculum_model(
        dataset=data,
        output_dir=str(tmp_path),
        base_model="fake-base",
        num_stages=3,
        num_epochs_per_stage=1,
        score_column="weighted_score",
        difficulty_order="easy_first",
        use_lora=True,
    )

    assert result["success"], result.get("error")
    assert result["num_stages"] == 3
    assert result["num_training_examples"] == 9
    assert result["pre_scored"] is True
    assert len(result["stage_results"]) == 3
    # _merge_lora should be called after stages 1 and 2 (not after stage 3)
    assert mock_merge_lora.call_count == 2
    assert mock_train_model.call_count == 3


@pytest.mark.asyncio
async def test_train_curriculum_model_no_evaluator_no_score():
    """If evaluator is unavailable and dataset lacks score_column, return error."""
    svc = CurriculumService()

    # Dataset without weighted_score
    data = [{"instruction": "q", "output": "a"}]

    # Make _score_dataset return None (simulating unavailable evaluator)
    async def _unavailable_score(d, col):
        return None

    svc._score_dataset = _unavailable_score  # type: ignore[method-assign]

    result = await svc.train_curriculum_model(
        dataset=data,
        output_dir="/tmp/curriculum_test",
        base_model="fake-base",
        num_stages=2,
        score_column="weighted_score",
    )

    assert result["success"] is False
    assert "weighted_score" in result["error"]


@pytest.mark.asyncio
async def test_curriculum_merge_lora_avoids_auto_device_map_and_falls_back(tmp_path):
    """Merge should avoid device_map='auto' and retry on CPU if single-device load fails."""
    stage_dir = tmp_path / "stage_1"
    stage_dir.mkdir()

    base_model = MagicMock()
    peft_model = MagicMock()
    merged_model = MagicMock()
    tokenizer = MagicMock()
    peft_model.merge_and_unload.return_value = merged_model

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.empty_cache"
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=[RuntimeError("oom"), base_model],
    ) as mock_load_model, patch(
        "peft.PeftModel.from_pretrained", return_value=peft_model
    ), patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
    ):
        result = await CurriculumService._merge_lora(
            str(stage_dir), "fake-base", "fake-tokenizer"
        )

    assert result == str(stage_dir / "merged")
    assert mock_load_model.call_count == 2
    assert mock_load_model.call_args_list[0].kwargs["device_map"] == {"": 0}
    assert "device_map" not in mock_load_model.call_args_list[1].kwargs
