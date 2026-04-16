"""Helpers for capability-specific profiled composition templates."""

from __future__ import annotations

from pathlib import Path

from shared.capability_registry import resolve_capability


class ProfilePromptTemplateManager:
    """Load capability templates stored under prompts/profiles/."""

    def __init__(self, profiles_dir: str | None = None):
        if profiles_dir is None:
            profiles_dir = Path(__file__).parent / "profiles"
        self.profiles_dir = Path(profiles_dir)
        self._cache: dict[tuple[str, str, str], str] = {}

    def load(self, mode: str, capability: str, objective: str = "sft") -> str:
        cache_key = (objective, mode, capability)
        if cache_key in self._cache:
            return self._cache[cache_key]

        template_path = self.profiles_dir / objective / mode / f"{capability}.txt"
        if not template_path.exists():
            template_path = self.profiles_dir / mode / f"{capability}.txt"

        if template_path.exists():
            template = template_path.read_text(encoding="utf-8")
        elif objective in {"dpo", "grpo"}:
            template = self._build_preference_template(objective, mode, capability)
        else:
            raise FileNotFoundError(f"Profile template not found: {template_path}")

        self._cache[cache_key] = template
        return template

    def _build_preference_template(
        self,
        objective: str,
        mode: str,
        capability: str,
    ) -> str:
        capability_def = resolve_capability(capability)
        description = (
            capability_def.description
            if capability_def is not None
            else capability.replace("_", " ")
        )
        agent_rule = ""
        if mode == "agent":
            agent_rule = (
                '5. When a tool action is part of the better answer, you may encode it as '
                '`<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>`.\n'
                "6. Return JSON only. No commentary, markdown, or code fences."
            )
        else:
            agent_rule = "5. Return JSON only. No commentary, markdown, or code fences."

        if objective == "grpo":
            schema_lines = (
                "[\n"
                "  {\n"
                '    "prompt": "<a realistic user request that tests this capability>",\n'
                '    "responses": ["<best answer>", "<strong alternative>", "<weaker alternative>", "<bad alternative>"],\n'
                '    "rewards": [1.0, 0.7, 0.2, -0.4]\n'
                "  }\n"
                "]"
            )
            rules = (
                "1. Keep every response grounded in the Source Text.\n"
                "2. Responses for a row must be meaningfully different, not surface paraphrases.\n"
                "3. Rewards must strictly rank the responses from best to worst.\n"
                "4. Make the prompt specific enough that the capability under test is obvious.\n"
            )
        else:
            schema_lines = (
                "[\n"
                "  {\n"
                '    "prompt": "<a realistic user request that tests this capability>",\n'
                '    "chosen": "<the stronger grounded answer or action>",\n'
                '    "rejected": "<a weaker alternative for the same prompt>"\n'
                "  }\n"
                "]"
            )
            rules = (
                "1. Keep both chosen and rejected grounded in the Source Text.\n"
                "2. The chosen answer must be clearly preferable for the target capability, not merely longer.\n"
                "3. The rejected answer should fail through a realistic weakness such as being incomplete, weakly grounded, misprioritized, or slightly incorrect.\n"
                "4. Make the prompt specific enough that the capability under test is obvious.\n"
            )

        return (
            f"You are creating {objective.upper()} rows for the "
            f"'{capability}' capability in {mode} tuning.\n\n"
            f"Capability focus: {description}\n\n"
            "Use the provided Source Text to create 1 to 3 JSON objects. Each object "
            f"must follow this schema exactly:\n{schema_lines}\n\n"
            "Rules:\n"
            f"{rules}"
            f"{agent_rule}\n\n"
            "Source Text:\n\n"
            "{text}"
        )
