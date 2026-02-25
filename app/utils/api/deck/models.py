# UX/src/types.js
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DeckRequest():

    company: str
    industry: str
    targetAudience: str
    painPoints: str
    solution: str
    differentiators: str
    desiredOutcomes: str
    draftStyle: str


@dataclass
class UserRequest:
    user_prompt: DeckRequest

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
class Slide:
    slideNumber: int
    title: str
    bullets: List[str]
    visualSuggestion: str
    speakerNotes: str


@dataclass
class DeckOutput:
    deckTitle: str
    targetAudience: str
    slides: List[Slide]
    openQuestions: List[str]
    nextSteps: List[str]
