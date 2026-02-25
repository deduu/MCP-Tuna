from .models import DeckRequest


class DeckPromptBuilder:
    def __init__(self, template: str):
        self.template = template

    def build(self, deck: DeckRequest) -> str:
        return self.template.format(
            COMPANY_NAME=deck.company,
            INDUSTRY=deck.industry,
            REGION="[Not specified]",
            TARGET_AUDIENCE=deck.targetAudience,
            MEETING_TYPE="First discovery",

            PAIN_POINTS=deck.painPoints,
            CURRENT_STACK="[Not provided]",
            SOLUTION_DESCRIPTION=deck.solution,
            DIFFERENTIATORS=deck.differentiators,
            PROOF_POINTS="[To be validated]",
            CONSTRAINTS="[None stated]",
            DESIRED_OUTCOMES=deck.desiredOutcomes,
            CALL_TO_ACTION="Agree on next-step workshop"
        )
