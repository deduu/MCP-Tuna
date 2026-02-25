import logging
from fastapi import HTTPException, status, Request
from .models import DeckRequest, UserRequest


class InputParser:
    @staticmethod
    async def parse_user_request(request: Request) -> UserRequest:
        try:
            body = await request.json()

            deck = DeckRequest(**body["user_prompt"])

            return UserRequest(
                user_prompt=deck,

                model_name=body["model_name"],
                stream=body["stream"],
                query_mode=body["query_mode"],
                selected_tools=body.get("selected_tools", []),
                do_rerank=body.get("do_rerank", False),
                api_key=body.get("api_key"),
                base_url=body.get("base_url"),

                temperature=body["temperature"],
                top_p=body["top_p"],
                top_k=body.get("top_k"),
                max_tokens=body.get("max_tokens"),
                presence_penalty=body.get("presence_penalty"),
                frequency_penalty=body.get("frequency_penalty"),
                repetition_penalty=body.get("repetition_penalty"),
            )

        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing field: {e}",
            )
        except Exception as e:
            logging.error(f"Invalid request: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request body",
            )
