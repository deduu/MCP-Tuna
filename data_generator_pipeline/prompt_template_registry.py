from langchain_core.prompts import ChatPromptTemplate


class PromptTemplateRegistry:
    def __init__(self, template: str):
        self.template = template
        self.prompt = ChatPromptTemplate.from_template(template)

    def render(self, text: str) -> str:
        text = text.replace("\n\n", "\n")
        return self.prompt.format_messages(text=text)[0].content
