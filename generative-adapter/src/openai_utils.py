import asyncio
import logging
import os
from typing import Any

import aiolimiter

import openai
from openai import AsyncAzureOpenAI
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio

ERROR_ERRORS_TO_MESSAGES = {
    openai.UnprocessableEntityError: "OpenAI API Invalid Request: Prompt was filtered",
    openai.RateLimitError: "OpenAI API rate limit exceeded. Sleeping for 10 seconds.",
    openai.APIStatusError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    openai.APITimeoutError: "OpenAI APITimeout Error: OpenAI Timeout",
    openai.InternalServerError: "OpenAI service unavailable error: {e}",
    openai.APIError: "OpenAI API error: {e}",
    openai.BadRequestError: "OpenAI API Bad Request error: {e}",
}

async def _throttled_openai_chat_completion_acreate(
    client: AsyncAzureOpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(30):
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                )
            except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                if isinstance(e, (openai.InternalServerError)):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                elif isinstance(e, openai.UnprocessableEntityError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Invalid Request: Prompt was filtered"
                                }
                            }
                        ]
                    }
                elif isinstance(e, openai.BadRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Bad Request"
                                }
                            }
                        ]
                    }
                else:
                    a = 1
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    azure_openai_api_key: str,
    azure_openai_api_endpoint: str,
    full_contexts: list,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int,
    requests_per_minute: int = 1000,
) -> list[list[str]]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        model_name: Model name.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """

    client = AsyncAzureOpenAI(
        api_key = azure_openai_api_key,
        api_version = "2024-02-15-preview",
        azure_endpoint=azure_openai_api_endpoint,
    )
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client=client,
            model=model_name,
            messages=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # # Note: will never be none because it's set, but mypy doesn't know that.
    # all_responses = []
    # for i, x in enumerate(responses):
    #     try:
    #         all_responses.append([x.choices[i].message.content for i in range(n)])
    #     except Exception as e:
    #         print(f"An error occurred: {e} for {x} at index {i}")
    #         all_responses.append([""])
    #     # all_responses.append(x)
    return responses
