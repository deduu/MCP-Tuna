# from .base import BaseMetric


# class InstructionFollowingDifficultyMetric(BaseMetric):
#     name = "ifd"

#     def compute(self, dp):
#         instruction = dp.full_instruction.lower()
#         output = dp.output.lower()

#         specific_terms = [
#             "explain", "analyze", "compare", "evaluate",
#             "step by step", "justify", "describe"
#         ]

#         specificity = sum(1 for t in specific_terms if t in instruction)
#         output_complexity = min(len(output.split()) / 100, 1.0)

#         inst_tokens = set(instruction.split())
#         out_tokens = set(output.split())
#         alignment = len(inst_tokens & out_tokens) / max(len(inst_tokens), 1)

#         return min(
#             1.0,
#             0.4 * min(specificity / 3, 1.0)
#             + 0.3 * output_complexity
#             + 0.3 * (1.0 - alignment)
#         )
from .base import BaseMetric


class InstructionFollowingDifficultyMetric(BaseMetric):
    name = "ifd"

    SPECIFIC_TERMS_ID = [
        "jelaskan", "analisis", "bandingkan", "evaluasi",
        "uraikan", "jabarkan",
        "langkah demi langkah", "secara rinci",
        "mengapa", "bagaimana"
    ]

    STOPWORDS_ID = {
        "yang", "dan", "di", "ke", "dari", "untuk", "pada",
        "adalah", "itu", "ini"
    }

    def compute(self, dp):
        instruction = dp.full_instruction.lower()
        output = dp.output.lower()

        # 1. Instruction specificity
        specificity = sum(
            1 for t in self.SPECIFIC_TERMS_ID if t in instruction
        )
        specificity_score = min(specificity / 3, 1.0)

        # 2. Output effort
        output_complexity = min(len(output.split()) / 100, 1.0)

        # 3. Content alignment (reduced noise)
        inst_tokens = {
            t for t in instruction.split()
            if t not in self.STOPWORDS_ID
        }
        out_tokens = {
            t for t in output.split()
            if t not in self.STOPWORDS_ID
        }

        alignment = len(inst_tokens & out_tokens) / max(len(inst_tokens), 1)

        return min(
            1.0,
            0.4 * specificity_score
            + 0.3 * output_complexity
            + 0.3 * (1.0 - alignment)
        )
