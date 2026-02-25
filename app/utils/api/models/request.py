from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatRequest:

    """Parsed request data fro any OpenAI API request."""
    # Conversation & Model Execution
    messages: List[Dict[str, Any]]
    truncated_messages: List[Dict[str, Any]]
    user_prompt: str
    request_type: str        # "chat_completion" | "response_api" | "voice" | "whatsapp"

    # Core routing/business logic
    model_name: str
    stream: bool
    query_mode: str
    selected_tools: List[str]
    do_rerank: bool
    api_key: Optional[str]
    base_url: Optional[str]

    # ---- Generation Control Parameters ----
    temperature: float
    top_p: float
    top_k: Optional[int]
    max_tokens: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    repetition_penalty: Optional[float]


@dataclass
class AgentContext:
    """Context for the agent execution."""
    chat_request: ChatRequest
    model_name: str
    model_id: str
    enable_thinking: bool
    selected_tools: Optional[List[str]]
    mcp_servers: Optional[List[Dict[str, Any]]]
    route: str
    enable_logging: bool
