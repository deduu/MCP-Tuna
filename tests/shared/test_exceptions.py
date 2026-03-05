"""Tests for the typed exception hierarchy."""
from __future__ import annotations

from shared.exceptions import (
    DiskSpaceError,
    EvaluationError,
    GenerationError,
    HostingError,
    OOMError,
    PipelineError,
    ProviderError,
    ProviderTimeoutError,
    ResourceError,
    TrainingError,
    TunaError,
    ValidationError,
)


class TestExceptionHierarchy:
    def test_oom_is_resource_error(self):
        assert issubclass(OOMError, ResourceError)

    def test_resource_error_is_tuna_error(self):
        assert issubclass(ResourceError, TunaError)

    def test_disk_space_is_resource_error(self):
        assert issubclass(DiskSpaceError, ResourceError)

    def test_pipeline_errors_are_tuna_errors(self):
        for cls in (GenerationError, EvaluationError, TrainingError, HostingError):
            assert issubclass(cls, PipelineError)
            assert issubclass(cls, TunaError)

    def test_provider_timeout_is_provider_error(self):
        assert issubclass(ProviderTimeoutError, ProviderError)
        assert issubclass(ProviderError, TunaError)

    def test_validation_is_tuna_error(self):
        assert issubclass(ValidationError, TunaError)

    def test_oom_catch_as_tuna_error(self):
        try:
            raise OOMError("GPU ran out of memory")
        except TunaError as e:
            assert "GPU" in str(e)

    def test_oom_preserves_message(self):
        err = OOMError("CUDA out of memory. Tried to allocate 2 GiB")
        assert "2 GiB" in str(err)
