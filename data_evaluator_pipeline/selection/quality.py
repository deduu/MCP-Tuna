from .base import BaseSelector
# selection/quality.py


class QualityThresholdSelector(BaseSelector):
    def __init__(self, evaluator, metric="combined_score", threshold=0.7):
        self.evaluator = evaluator
        self.metric = metric
        self.threshold = threshold

    def select(self, dataset):
        selected = []
        for dp in dataset:
            scores = self.evaluator.evaluate(dp)
            if scores[self.metric] >= self.threshold:
                selected.append(dp)
        return selected
