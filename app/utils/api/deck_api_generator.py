import logging
from fastapi.responses import JSONResponse
from fastapi import HTTPException, status, Request
from .deck.input_parser import InputParser
from .deck.models import DeckOutput, UserRequest
from .deck.agent_executor import AgentExecutor
from .deck.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .deck.prompt_builder import DeckPromptBuilder
from .deck.response_builder import DeckResponseBuilder
from ...utils.timing import TimingContext

logger = logging.getLogger(__name__)


class DeckApiGenerator:
    def __init__(self):
        self.parser = InputParser()
        self.agent_executor = AgentExecutor()
        self.system_prompt = SYSTEM_PROMPT
        self.prompt_builder = DeckPromptBuilder(USER_PROMPT_TEMPLATE)
        self.response_builder = DeckResponseBuilder()

    async def prepare_context(self, request: Request) -> UserRequest:
        deck_request = await self.parser.parse_user_request(request)
        return deck_request

    async def generate_deck(self, request: Request) -> DeckOutput:
        timing = TimingContext()
        try:
            # Prepare context
            context = await self.prepare_context(request)

            user_prompt = self.prompt_builder.build(
                deck=context.user_prompt,
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Create agent
            agent = await self.agent_executor.create_agent(
                model_name=context.model_name,
                query_mode=context.query_mode,
                do_rerank=context.do_rerank,
                selected_tools=context.selected_tools,
                api_key=context.api_key,
                base_url=context.base_url,
                system_prompt=self.system_prompt,
            )

            # Handle non-streaming
            if not context.stream:
                content = await self.agent_executor.run_non_streaming(
                    agent,
                    messages=messages,
                    enable_thinking=False,
                    timing=timing,
                )

                response = self.response_builder.build_openai_chat_completion(
                    raw_content=content["content"],   # ← JSON STRING
                    model_name=context.model_name,
                    usage=content.get("usage", {}),
                )

                return JSONResponse(content=response)

        except Exception as e:
            logging.error(f"Error generating deck: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error generating deck",
            )
