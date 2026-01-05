# pipeline.py
class InstructionTuningPipeline:
    def __init__(self, evaluator, selector, analyzer):
        self.evaluator = evaluator
        self.selector = selector
        self.analyzer = analyzer

    def run(self, dataset):
        original_stats = self.analyzer.analyze(dataset)
        subset = self.selector.select(dataset)
        subset_stats = self.analyzer.analyze(subset)

        return {
            "original": original_stats,
            "selected": subset_stats,
            "subset": subset
        }
