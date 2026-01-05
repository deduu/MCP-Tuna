import json
import re


class JsonExtractor:
    @staticmethod
    def extract(content: str) -> list:
        # remove thinking tags
        if "</think>" in content:
            content = content.split("</think>")[-1]

        # extract fenced json if exists
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)

        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON returned:\n{content}") from e
