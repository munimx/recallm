"""extract_prompt_text — canonical prompt extraction from chat messages."""
from __future__ import annotations

from llm_semantic_cache.models import ChatMessage


def extract_prompt_text(messages: list[ChatMessage]) -> str | None:
    """Extract the canonical prompt text from a list of chat messages.

    Returns the content of the last message with role='user'. Returns None if:
    - No message has role='user'
    - The last user message has empty or non-string content

    System prompts and assistant messages are intentionally excluded from
    the embedding. They are part of the context (passed via cache_context),
    not the prompt.
    """
    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        return None
    last_user = user_messages[-1]
    if not last_user.content or not last_user.content.strip():
        return None
    return last_user.content.strip()
