import time


class TimingContext:
    """Tracks timing metrics for requests."""

    def __init__(self):
        self.request_start = time.perf_counter()
        self.inference_start = None
        self.inference_end = None
        self.first_token_time = None

    def start_inference(self):
        self.inference_start = time.perf_counter()

    def record_first_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def end_inference(self):
        self.inference_end = time.perf_counter()

    @property
    def total_time(self) -> float:
        return time.perf_counter() - self.request_start

    @property
    def inference_time(self) -> float:
        if self.inference_start and self.inference_end:
            return self.inference_end - self.inference_start
        return 0.0

    @property
    def first_token_latency(self) -> float:
        if self.inference_start and self.first_token_time:
            return self.first_token_time - self.inference_start
        return 0.0
