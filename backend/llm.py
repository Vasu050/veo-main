import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def response_text(response_content: Any) -> str:
    if isinstance(response_content, str):
        return response_content
    if isinstance(response_content, list):
        return "\n".join(
            block["text"]
            for block in response_content
            if isinstance(block, dict)
            and isinstance(block.get("text"), str)
        )
    return ""


def create_gemini_model(
    model_name: str,
    temperature: float,
    max_tokens: int,
    response_mime_type: str | None = None,
) -> ChatGoogleGenerativeAI:
    """Create a Gemini client using the configured Google API key."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing from your .env file."
        )

    model_options: dict[str, str] = {}
    if response_mime_type is not None:
        model_options["response_mime_type"] = response_mime_type
   

    return ChatGoogleGenerativeAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,   
        **model_options,
    )
