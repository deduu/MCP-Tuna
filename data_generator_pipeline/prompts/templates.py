
# ============================================================================
# FILE: src/finetuning/prompts/templates.py
# ============================================================================

from pathlib import Path


class PromptTemplateManager:
    """Manages prompt templates for different techniques."""

    def __init__(self, templates_dir: str = None):
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        self.templates_dir = Path(templates_dir)
        self._cache = {}

    def load(self, technique: str) -> str:
        """Load template for a specific technique."""
        if technique in self._cache:
            return self._cache[technique]

        template_file = self.templates_dir / f"{technique}.txt"

        if not template_file.exists():
            raise FileNotFoundError(
                f"Template not found: {template_file}"
            )

        with open(template_file, 'r', encoding='utf-8') as f:
            template = f.read()

        self._cache[technique] = template
        return template

    def get_template(self, technique: str, custom_template: str = None) -> str:
        """Get template, using custom if provided."""
        if custom_template:
            return custom_template
        return self.load(technique)
