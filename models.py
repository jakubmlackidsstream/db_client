"""
Model factory for the DB agent.

Usage:
    from models import get_llm, ModelTier, Provider

    llm_small  = get_llm(Provider.ANTHROPIC, ModelTier.SMALL)
    llm_large  = get_llm(Provider.OPENAI, ModelTier.LARGE)

Tiers:
    SMALL  – fast, cheap; good for classification, binary decisions
    MEDIUM – balanced; good for summarisation, clarification phrasing
    LARGE  – most capable; use for SQL planning and fixing

LangChain's init_chat_model is used under the hood, so the same
BaseChatModel interface works everywhere regardless of provider.
"""

from enum import Enum
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


class Provider(str, Enum):
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE    = "google_genai"


class ModelTier(str, Enum):
    SMALL  = "small"   # fast / cheap
    MEDIUM = "medium"  # balanced
    LARGE  = "large"   # most capable


# Map (provider, tier) → model name understood by init_chat_model
_MODEL_MAP: dict[tuple[Provider, ModelTier], str] = {
    # OpenAI
    (Provider.OPENAI, ModelTier.SMALL):  "gpt-4o-mini",
    (Provider.OPENAI, ModelTier.MEDIUM): "gpt-4o",
    (Provider.OPENAI, ModelTier.LARGE):  "gpt-4o",

    # Anthropic
    (Provider.ANTHROPIC, ModelTier.SMALL):  "claude-haiku-4-5-20251001",
    (Provider.ANTHROPIC, ModelTier.MEDIUM): "claude-sonnet-4-6",
    (Provider.ANTHROPIC, ModelTier.LARGE):  "claude-opus-4-6",

    # Google Gemini
    (Provider.GOOGLE, ModelTier.SMALL):  "gemini-2.0-flash",
    (Provider.GOOGLE, ModelTier.MEDIUM): "gemini-2.5-flash-preview-04-17",
    (Provider.GOOGLE, ModelTier.LARGE):  "gemini-2.5-pro-preview-03-25",
}


def get_llm(
    provider: Provider = Provider.OPENAI,
    tier: ModelTier = ModelTier.SMALL,
    **kwargs,
) -> BaseChatModel:
    """
    Return an initialised LangChain chat model for the given provider and tier.

    Extra kwargs (e.g. temperature=0) are forwarded to init_chat_model.
    """
    model_name = _MODEL_MAP[(provider, tier)]
    return init_chat_model(model_name, model_provider=provider.value, **kwargs)
