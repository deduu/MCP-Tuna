from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import spacy
from transformers import AutoTokenizer

from .base import BaseMetric

log = logging.getLogger(__name__)

# Module-level caches to avoid reloading heavy models on every metric instance
_SPACY_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_spacy(model_name: str) -> Any:
    if model_name not in _SPACY_CACHE:
        log.info("Loading spaCy model '%s' (cached for reuse)", model_name)
        _SPACY_CACHE[model_name] = spacy.load(model_name)
    return _SPACY_CACHE[model_name]


def _get_tokenizer(name: str) -> Any:
    if name not in _TOKENIZER_CACHE:
        log.info("Loading tokenizer '%s' (cached for reuse)", name)
        _TOKENIZER_CACHE[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER_CACHE[name]

# --- Data Structure for Language Config ---


@dataclass
class LanguageConfig:
    simple_words: set[str]
    clause_markers: List[str]
    reasoning_connectors: List[str]
    explanatory_phrases: List[str]
    spacy_model: str

# --- Configuration Loader ---


def load_language_config(language_code: str) -> LanguageConfig:
    """
    Loads language-specific configuration from a JSON file.
    Falls back to English ('en') if the specified language is not found.
    """
    config_dir = Path(__file__).parent / "configs"
    config_path = config_dir / f"{language_code}.json"

    if not config_path.exists():
        print(
            f"Warning: Config for '{language_code}' not found. Falling back to 'en'.")
        config_path = config_dir / "en.json"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return LanguageConfig(
            simple_words=set(data.get("simple_words", [])),
            clause_markers=data.get("clause_markers", []),
            reasoning_connectors=data.get("reasoning_connectors", []),
            explanatory_phrases=data.get("explanatory_phrases", []),
            spacy_model=data.get("spacy_model", "xx_ent_wiki_sm")
        )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"Could not load language config for '{language_code}': {e}")

# --- Helper Functions (Refactored) ---


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _log_norm(n: int, cap: int) -> float:
    if n <= 0:
        return 0.0
    return _clamp01(math.log1p(n) / math.log1p(cap))

# _word_tokens and _sentence_tokens are now replaced by spaCy


def _word_entropy_norm(words: List[str]) -> float:
    if len(words) <= 1:
        return 0.0
    c = Counter(words)
    n = sum(c.values())
    ps = [v / n for v in c.values()]
    H = -sum(p * math.log(p + 1e-12) for p in ps)
    Hmax = math.log(len(c))
    return H / (Hmax + 1e-12)


def _repetition_penalty(doc: spacy.tokens.Doc) -> float:
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if len(sentences) <= 1:
        return 1.0

    normalized = [' '.join(s.lower().split()) for s in sentences]
    unique_ratio = len(set(normalized)) / len(normalized)

    if len(normalized) >= 3:
        similar_count = 0
        total_pairs = 0
        for i in range(len(normalized)):
            for j in range(i + 1, min(i + 5, len(normalized))):
                total_pairs += 1
                s1, s2 = set(normalized[i].split()), set(normalized[j].split())
                if s1 and s2:
                    similarity = len(s1 & s2) / max(len(s1), len(s2))
                    if similarity > 0.7:
                        similar_count += 1

        if total_pairs > 0:
            similarity_penalty = 1.0 - (similar_count / total_pairs)
            unique_ratio = min(unique_ratio, similarity_penalty)

    return unique_ratio


def _vocabulary_richness(words: List[str]) -> float:
    if len(words) == 0:
        return 0.0
    unique_words = len(set(words))
    total_words = len(words)
    ttr = unique_words / total_words
    if total_words < 10:
        length_penalty = total_words / 10
        ttr = ttr * length_penalty
    length_bonus = _log_norm(total_words, cap=200)
    return _clamp01(ttr * 0.5 + length_bonus * 0.5)


def _technical_vocabulary_score(words: List[str], config: LanguageConfig) -> float:
    if len(words) == 0:
        return 0.0

    technical_count = 0
    very_technical_count = 0

    for word in words:
        if word not in config.simple_words:
            if len(word) >= 10:
                very_technical_count += 1
                technical_count += 1
            elif len(word) >= 6:
                technical_count += 1

    tech_ratio = technical_count / len(words)
    very_tech_ratio = very_technical_count / len(words)
    avg_word_len = sum(len(w) for w in words) / len(words)
    length_score = _clamp01((avg_word_len - 3) / 5)

    return _clamp01(tech_ratio * 0.5 + very_tech_ratio * 0.3 + length_score * 0.2)


def _semantic_density_score(doc: spacy.tokens.Doc, config: LanguageConfig) -> float:
    text = doc.text
    words = [token.text for token in doc]
    if len(words) == 0:
        return 0.0

    score = 0.0
    sentences = list(doc.sents)
    if len(sentences) > 0:
        # Build regex from config
        markers_pattern = r"\b(" + "|".join(config.clause_markers) + r")\b"
        clause_markers = len(re.findall(markers_pattern, text.lower()))
        clause_density = min(1.0, clause_markers / (len(sentences) * 2))
        score += clause_density * 0.4

        comma_count = text.count(',')
        commas_per_sentence = comma_count / len(sentences)
        comma_score = min(1.0, commas_per_sentence / 3)
        score += comma_score * 0.3

        concepts_per_sentence = len(words) / len(sentences)
        concept_density = _clamp01((concepts_per_sentence - 5) / 15)
        score += concept_density * 0.3

    return _clamp01(score)


def _structure_score(text: str, config: LanguageConfig) -> float:
    score = 0.0
    t = text.lower()

    numbered_steps = len(re.findall(r"\n\s*\d+\.", t))
    score += min(0.30, numbered_steps * 0.10)
    bullets = len(re.findall(r"\n\s*[-*•]", t))
    score += min(0.20, bullets * 0.05)

    # Build regexes from config
    connectors_pattern = r"\b(" + \
        "|".join(config.reasoning_connectors) + r")\b"
    connectors = len(re.findall(connectors_pattern, t))
    score += min(0.25, connectors * 0.06)

    explanatory_pattern = r"\b(" + \
        "|".join(config.explanatory_phrases) + r")\b"
    explanatory = len(re.findall(explanatory_pattern, t))
    score += min(0.15, explanatory * 0.04)

    paragraphs = len(re.split(r'\n\s*\n', text.strip()))
    if paragraphs >= 2:
        score += min(0.15, (paragraphs - 1) * 0.08)

    return _clamp01(score)


def _minimum_length_gate(output_tokens: int) -> float:
    if output_tokens < 10:
        return output_tokens / 10
    elif output_tokens < 30:
        return 0.5 + (output_tokens - 10) / 40
    else:
        return 1.0

# --- Main Metric Class (Refactored) ---


class ComplexityMetric(BaseMetric):
    """
    Measures instructional / reasoning complexity in a modular, multi-lingual way.

    Key improvements:
    - Multi-lingual support via spaCy and JSON configs.
    - Language-specific vocabulary, clause markers, and structural cues.
    - Robust tokenization and sentence splitting.
    """
    name = "complexity"

    def __init__(self, language_code: str = "en", tokenizer_name: str = "bert-base-multilingual-cased"):
        """
        Initialize the metric for a specific language.

        Args:
            language_code: The language code (e.g., 'en', 'id'). 
                           Must have a corresponding JSON in the /configs directory.
            tokenizer_name: The HuggingFace tokenizer for length normalization.
        """
        self.language_config = load_language_config(language_code)
        try:
            self.nlp = _get_spacy(self.language_config.spacy_model)
        except OSError:
            raise OSError(
                f"Spacy model '{self.language_config.spacy_model}' not found. "
                f"Please install it using: python -m spacy download {self.language_config.spacy_model}"
            )
        self.tokenizer = _get_tokenizer(tokenizer_name)

    def compute(self, dp) -> float:
        # Token-level length (using multilingual tokenizer)
        inst_ids = self.tokenizer.encode(
            dp.full_instruction, add_special_tokens=False)
        out_ids = self.tokenizer.encode(dp.output, add_special_tokens=False)

        instr_norm = _log_norm(len(inst_ids), cap=150)
        out_norm = _log_norm(len(out_ids), cap=500)
        length_gate = _minimum_length_gate(len(out_ids))

        # Language-aware processing using spaCy
        doc = self.nlp(dp.output)
        words = [token.text for token in doc]

        # Lexical diversity measures
        word_entropy = _word_entropy_norm(words)
        vocab_richness = _vocabulary_richness(words)
        tech_vocab = _technical_vocabulary_score(words, self.language_config)

        # Repetition and structure analysis
        repetition_factor = _repetition_penalty(doc)
        structure = _structure_score(dp.output, self.language_config)
        semantic_density = _semantic_density_score(doc, self.language_config)

        # Final weighted score (updated to include semantic density)
        base_score = (
            0.10 * instr_norm +
            0.10 * out_norm +          # Slightly reduced weight for raw length
            0.15 * word_entropy +
            0.15 * vocab_richness +
            0.15 * tech_vocab +
            0.20 * structure +         # Increased weight for structure
            0.15 * semantic_density    # New component
        )

        final_score = base_score * length_gate * \
            (0.2 + 0.8 * repetition_factor)

        return _clamp01(final_score)
