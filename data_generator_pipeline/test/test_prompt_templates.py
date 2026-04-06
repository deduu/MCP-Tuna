from __future__ import annotations

from ..prompts.templates import PromptTemplateManager


def test_preference_templates_are_non_empty_and_structured():
    template_manager = PromptTemplateManager()

    dpo_template = template_manager.get_template("dpo")
    grpo_template = template_manager.get_template("grpo")

    assert dpo_template.strip()
    assert '"chosen"' in dpo_template
    assert '"rejected"' in dpo_template
    assert "JSON only" in dpo_template

    assert grpo_template.strip()
    assert '"responses"' in grpo_template
    assert '"rewards"' in grpo_template
    assert "JSON only" in grpo_template
