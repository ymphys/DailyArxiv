import json
import logging
import os
from dataclasses import dataclass
from typing import List, Sequence

logger = logging.getLogger("dailyarxiv.summarize")


@dataclass(frozen=True)
class SummarizerConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int = 300


def _build_prompt(representatives: Sequence[dict]) -> str:
    fragments = []
    for rep in representatives:
        fragments.append(
            f"Title: {rep['title']}\nAbstract: {rep['abstract']}"
        )
    details = "\n\n".join(fragments)
    return (
        "Generate a concise topic label (6-12 words), 3-5 technical keywords, and a single sentence description (<=20 words) "
        "for the following group of papers.\n\n"
        f"{details}\n\n"
        "Respond in JSON with keys: topic, keywords (list), description."
    )


def summarize_cluster(representatives: Sequence[dict], config: SummarizerConfig) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required for summarization.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to summarize clusters.")

    prompt = _build_prompt(representatives)
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a high-level research assistant."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    text = response.choices[0].message.content
    try:
        summary = json.loads(text.strip())
        return summary
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM summary; returning defaults.")
        return {
            "topic": "",
            "keywords": [],
            "description": "",
        }
