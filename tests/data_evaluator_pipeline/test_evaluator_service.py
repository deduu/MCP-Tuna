from data_evaluator_pipeline.services.pipeline_service import EvaluatorService


def test_list_metrics_does_not_initialize_evaluator():
    service = EvaluatorService()

    result = service.list_metrics()

    assert result["success"] is True
    assert result["metrics"] == ["complexity", "ifd", "quality"]
    assert service._evaluator is None


def test_update_config_merges_weights_and_resets_cached_evaluator():
    service = EvaluatorService()
    service._evaluator = object()

    result = service._update_config_sync(
        weights={"quality": 1.1, "custom_metric": 0.25},
        threshold=0.85,
        language="id",
    )

    assert result["success"] is True
    assert result["config"]["threshold"] == 0.85
    assert result["config"]["language"] == "id"
    assert result["config"]["weights"]["complexity"] == 0.3
    assert result["config"]["weights"]["quality"] == 1.1
    assert result["config"]["weights"]["custom_metric"] == 0.25
    assert service._evaluator is None
