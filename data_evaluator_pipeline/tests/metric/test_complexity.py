import math
from dataclasses import dataclass
from data_evaluator_pipeline.core.metrics.complexity import ComplexityMetric


# ------------------------
# Concrete DataPoint
# ------------------------

@dataclass
class DataPoint:
    full_instruction: str
    output: str


def print_score(label: str, score: float):
    print(f"[{label:<30}] complexity = {score:.4f}")


# ------------------------
# Test cases
# ------------------------

def test_simple_answer():
    """
    Very short, low-diversity response.
    Expected: low complexity
    """
    dp = DataPoint(
        full_instruction="Summarize the text.",
        output="This is a short answer."
    )

    metric = ComplexityMetric()
    score = metric.compute(dp)

    print_score("simple_answer", score)

    assert 0.0 <= score <= 1.0


def test_repetitive_answer():
    """
    Long but highly repetitive response.
    Expected: lower than diverse explanations.
    """
    dp = DataPoint(
        full_instruction="Explain neural networks.",
        output=(
            "Neural networks are important. Neural networks are important. "
            "Neural networks are important. Neural networks are important. "
            "Neural networks are important."
        )
    )

    metric = ComplexityMetric()
    score = metric.compute(dp)

    print_score("repetitive_answer", score)

    assert 0.0 <= score <= 1.0


def test_diverse_explanatory_answer():
    """
    Rich vocabulary, causal structure, multiple clauses.
    Expected: higher complexity than repetitive text.
    """
    dp = DataPoint(
        full_instruction="Explain how neural networks learn.",
        output=(
            "Neural networks learn by iteratively adjusting their parameters through "
            "a process known as gradient descent. During training, errors between "
            "predicted outputs and ground truth values are propagated backward through "
            "the network, allowing each layer to refine its internal representations."
        )
    )

    metric = ComplexityMetric()
    score = metric.compute(dp)

    print_score("diverse_explanatory", score)

    assert 0.0 <= score <= 1.0


def test_highly_structured_reasoning():
    """
    Multi-step reasoning with explicit structure.
    Expected: high complexity.
    """
    dp = DataPoint(
        full_instruction=(
            "Explain the CAP theorem and its implications for distributed systems."
        ),
        output=(
            "The CAP theorem states that a distributed system cannot simultaneously "
            "guarantee Consistency, Availability, and Partition tolerance.\n\n"
            "1. Consistency ensures all nodes observe the same data at the same time.\n"
            "2. Availability guarantees that every request receives a response.\n"
            "3. Partition tolerance allows the system to function despite network splits.\n\n"
            "In real-world systems, designers must choose which properties to prioritize "
            "depending on failure modes and latency requirements."
        )
    )

    metric = ComplexityMetric()
    score = metric.compute(dp)

    print_score("structured_reasoning", score)

    assert score > 0.4


def test_very_long_but_shallow_text():
    """
    Long output with weak informational density.
    Expected: length helps, repetition hurts.
    """
    dp = DataPoint(
        full_instruction="Describe the topic.",
        output=" ".join(
            ["This topic is very important and interesting."] * 200
        )
    )

    metric = ComplexityMetric()
    score = metric.compute(dp)

    print_score("long_but_shallow", score)

    assert 0.0 <= score <= 1.0


def test_dense_technical_explanation():
    """
    High information density, technical vocabulary.
    Expected: among the highest scores.
    """
    dp = DataPoint(
        full_instruction=(
            "Explain the difference between strong and eventual consistency."
        ),
        output=(
            "Strong consistency guarantees linearizability, meaning all operations "
            "appear to execute atomically and in a single global order. In contrast, "
            "eventual consistency allows replicas to diverge temporarily, provided "
            "they converge to the same state in the absence of further updates. "
            "Systems such as Amazon Dynamo adopt eventual consistency to reduce latency "
            "and improve availability under network partitions."
        )
    )

    metric = ComplexityMetric()
    score = metric.compute(dp)

    print_score("dense_technical", score)

    assert score > 0.5


def test_empty_output():
    """
    Degenerate case.
    Expected: no crash, very low complexity.
    """
    dp = DataPoint(
        full_instruction="Say something.",
        output=""
    )

    metric = ComplexityMetric()
    score = metric.compute(dp)

    print_score("empty_output", score)

    assert 0.0 <= score <= 1.0


def test_determinism():
    """
    Same input must produce identical score.
    """
    dp = DataPoint(
        full_instruction="Explain entropy.",
        output=(
            "Entropy measures uncertainty in a probability distribution "
            "and is maximized when all outcomes are equally likely."
        )
    )

    metric = ComplexityMetric()
    s1 = metric.compute(dp)
    s2 = metric.compute(dp)

    print_score("determinism_run_1", s1)
    print_score("determinism_run_2", s2)

    assert math.isclose(s1, s2, rel_tol=1e-6)


# ------------------------
# Standalone execution
# ------------------------

if __name__ == "__main__":
    print("\nRunning ComplexityMetric tests manually:\n")

    test_simple_answer()
    test_repetitive_answer()
    test_diverse_explanatory_answer()
    test_highly_structured_reasoning()
    test_very_long_but_shallow_text()
    test_dense_technical_explanation()
    test_empty_output()
    test_determinism()
