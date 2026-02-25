
import re
import json
import logging
from typing import Dict, Any
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


def _attempt_json_repair(raw: str) -> str:
    """
    Best-effort repair for truncated JSON.
    Closes open braces/brackets.
    """
    # Trim trailing junk
    raw = raw.strip()

    # Count braces
    open_curly = raw.count("{")
    close_curly = raw.count("}")
    open_square = raw.count("[")
    close_square = raw.count("]")

    raw += "}" * max(0, open_curly - close_curly)
    raw += "]" * max(0, open_square - close_square)

    return raw


class DeckResponseBuilder:
    REQUIRED_FIELDS = {
        "deckTitle",
        "targetAudience",
        "slides",
        "openQuestions",
        "nextSteps",
    }

    @classmethod
    def build_openai_chat_completion(
        cls,
        raw_content: str,
        model_name: str,
        usage: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parse, validate, and wrap LLM JSON output.
        """
        # Extract JSON from code blocks if present
        if raw_content.strip().startswith("```json"):
            raw_content = raw_content.strip()
            if raw_content.endswith("```"):
                # Remove ```json and ```
                raw_content = raw_content[7:-3].strip()
            else:
                # If no closing ```, try to find the JSON
                lines = raw_content.split('\n')
                if lines[0].startswith("```json"):
                    lines = lines[1:]
                    # Find the closing ```
                    for i, line in enumerate(lines):
                        if line.strip() == "```":
                            raw_content = '\n'.join(lines[:i]).strip()
                            break
                    else:
                        raw_content = '\n'.join(lines).strip()

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            repaired = _attempt_json_repair(raw_content)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                logger.error("LLM returned invalid JSON (unrepairable)")
                logger.error(raw_content)
                raise HTTPException(
                    status_code=500,
                    detail="Model returned invalid JSON"
                )

        # --- schema validation (lightweight but strict) ---
        missing = cls.REQUIRED_FIELDS - parsed.keys()
        if missing:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Missing required fields in model output: {missing}",
            )

        # --- optional: slides sanity check ---
        if not isinstance(parsed["slides"], list) or not parsed["slides"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Slides must be a non-empty list",
            )

        # --- final API payload ---
        return {
            "model": model_name,
            "usage": usage,
            "data": parsed,
        }
