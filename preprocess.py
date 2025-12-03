import re
import unicodedata
from typing import Iterable, Sequence

COMMON_BOILERPLATE = [
    "we propose",
    "in this paper",
    "this paper proposes",
    "this work presents",
    "we introduce",
    "our approach",
    "this study",
    "in recent years",
]

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "for",
    "with",
    "to",
    "is",
    "are",
    "be",
    "by",
    "from",
    "this",
    "that",
    "on",
    "in",
    "as",
    "we",
    "will",
}


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def remove_latex(text: str) -> str:
    text = re.sub(r"\$[^$]+\$", " ", text)
    text = re.sub(r"\\\([^\)]+\\\)", " ", text)
    text = re.sub(r"\\\[[^\]]+\\\]", " ", text)
    return text


def remove_citations(text: str) -> str:
    return re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", " ", text)


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_boilerplate(text: str, blacklist: Iterable[str]) -> str:
    lowered = text.lower()
    for phrase in blacklist:
        lowered = lowered.replace(phrase.lower(), " ")
    return lowered


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str, remove_stopwords: bool = False, boilerplate: Sequence[str] = COMMON_BOILERPLATE) -> str:
    if not text:
        return ""
    text = text.lower()
    text = remove_latex(text)
    text = remove_citations(text)
    text = remove_urls(text)
    text = strip_accents(text)
    text = remove_boilerplate(text, boilerplate)
    tokens = normalize_whitespace(text).split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return " ".join(tokens)
