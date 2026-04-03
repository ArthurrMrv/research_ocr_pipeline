"""Token counting and context window management utilities."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Context window limits (input tokens) per model family.
# Conservative estimates leaving room for output tokens.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-5.4": 120_000,
    "gpt-5.4-mini": 120_000,
    "gpt-5.4-nano": 120_000,
    "gpt-4o": 120_000,
    "gpt-4o-mini": 120_000,
}

# Threshold: warn and trim when exceeding this fraction of context limit.
CONTEXT_THRESHOLD = 0.80

# Rough chars-per-token ratio for GPT models (average ~4 chars/token for English).
_CHARS_PER_TOKEN = 4

try:
    import tiktoken

    _encoder = tiktoken.encoding_for_model("gpt-4o")

    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken."""
        return len(_encoder.encode(text))

except ImportError:
    _encoder = None  # type: ignore[assignment]

    def count_tokens(text: str) -> int:  # type: ignore[misc]
        """Estimate token count from character count (tiktoken not installed)."""
        return len(text) // _CHARS_PER_TOKEN


def get_context_limit(model: str) -> int:
    """Return the context window limit for a model, with a safe default."""
    # Try exact match, then prefix match
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if model.startswith(prefix):
            return limit
    return 120_000  # safe default


def check_context_budget(
    prompt: str,
    ocr_text: str,
    model: str,
    *,
    step_name: str = "",
) -> str:
    """Check if prompt + ocr_text fits within context budget.

    Returns the (possibly truncated) ocr_text. If the combined token count
    exceeds the threshold, logs a warning with token counts.
    """
    limit = get_context_limit(model)
    threshold = int(limit * CONTEXT_THRESHOLD)

    prompt_tokens = count_tokens(prompt)
    ocr_tokens = count_tokens(ocr_text)
    total = prompt_tokens + ocr_tokens

    logger.debug(
        "Token budget [%s]: prompt=%d, ocr=%d, total=%d, limit=%d (threshold=%d)",
        step_name or model,
        prompt_tokens,
        ocr_tokens,
        total,
        limit,
        threshold,
    )

    if total > threshold:
        logger.warning(
            "Context budget exceeded for %s: %d tokens > %d threshold "
            "(prompt=%d, ocr=%d). Consider reducing page count.",
            step_name or model,
            total,
            threshold,
            prompt_tokens,
            ocr_tokens,
        )

    return ocr_text
