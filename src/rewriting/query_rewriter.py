import logging

import anthropic

logger = logging.getLogger(__name__)

MAX_REWRITE_CHARS = 300
QUOTE_CHARS = "\"'“”‘’"


class QueryRewriteError(Exception):
    """Raised when a query cannot be rewritten into usable text."""


SYSTEM_PROMPT = """
You are a query rewriting component in a retrieval-augmented
generation (RAG) system.

Your task is to rewrite a user's question to improve semantic
retrieval from a knowledge base.

Rules:
1. Preserve the original question's meaning and intent.
2. Do not answer the question.
3. Do not introduce facts or assumptions not present in the question.
4. Use clear terminology that is likely to match relevant
   knowledge-base content.
5. Keep the rewritten query concise.
6. Return only the rewritten question, without explanations,
   quotation marks, or additional commentary.
"""


def rewrite_query(
    question: str,
    client,
    model: str,
    max_tokens: int = 200,
) -> str:
    """
    Rewrite a user question to improve retrieval.

    Args:
        question: Original user question.
        client: Configured Anthropic client.
        model: Anthropic model identifier.
        max_tokens: Maximum response tokens.

    Returns:
        Rewritten question as a string.

    Raises:
        ValueError: If the input or model configuration is invalid.
        QueryRewriteError: If the request fails or the model returns
            unusable output (empty, cut off, too long or several lines).
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(model, str) or not model.strip():
        raise ValueError("A valid model identifier is required.")

    if max_tokens < 1:
        raise ValueError("max_tokens must be greater than zero.")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": question.strip(),
                }
            ],
        )
    except anthropic.APIError as e:
        raise QueryRewriteError("The rewrite request failed.") from e

    if response.stop_reason == "max_tokens":
        raise QueryRewriteError("The rewritten query was cut off by the token limit.")

    rewritten = (
        "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        .strip()
        .strip(QUOTE_CHARS)
        .strip()
    )

    if not rewritten:
        raise QueryRewriteError("The model returned an empty rewritten query.")

    if "\n" in rewritten:
        raise QueryRewriteError("The model returned more than one line.")

    if len(rewritten) > MAX_REWRITE_CHARS:
        raise QueryRewriteError("The rewritten query is longer than expected.")

    return rewritten


def rewrite_or_original(
    question: str,
    client,
    model: str,
    max_tokens: int = 200,
) -> str:
    """
    Rewrite the question, or return it unchanged if rewriting fails.

    A failed rewrite should never stop a search, because the original
    question is still a valid query.
    """
    try:
        return rewrite_query(question, client, model, max_tokens)
    except QueryRewriteError:
        logger.warning(
            "Query rewrite failed, using the original question",
            exc_info=True,
        )
        return question.strip()
