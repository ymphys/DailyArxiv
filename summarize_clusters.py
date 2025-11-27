import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

DEFAULT_DATE = "today"
DEFAULT_INPUT_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_PAPERS = 50
DEFAULT_CHUNK_SIZE = 30
DEFAULT_MAX_WORKERS = 16
DEFAULT_TEMPERATURE = 0.3
LLM_RETRIES = 3
BACKOFF_SECONDS = 2.0

SYSTEM_PROMPT = """你是资深科学编辑，需要对论文主题簇进行解读。务必使用简体中文，并输出有效的 JSON。"""

CHUNK_PROMPT_TEMPLATE = """你将收到一组论文的标题和摘要，它们属于同一个主题簇。
请输出 JSON，字段为：
{{
  "research_question": "该主题簇的主要科学问题",
  "methods": "常用的方法或技术路径",
  "trends": "今天观察到的新趋势",
  "summary": "100-200 字的主题摘要，聚焦于科研意义与发现",
  "keywords": ["关键词1", "关键词2", "... 最多 6 个"]
}}
论文数据：
{papers}
"""

AGGREGATE_PROMPT_TEMPLATE = """以下是同一主题簇的分块总结，请综合它们，生成最终 JSON：
{{
  "research_question": "...",
  "methods": "...",
  "trends": "...",
  "summary": "100-200 字的主题摘要",
  "keywords": ["..."]
}}
分块总结：
{chunks}
"""


@dataclass(frozen=True)
class SummaryConfig:
    date: str
    papers_path: Path
    clusters_path: Path
    output_path: Path
    model: str
    max_papers: int
    chunk_size: int
    max_workers: int
    temperature: float


class LLMClient:
    def __init__(self, model: str, temperature: float):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; required for summarization.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def chat_completion(self, prompt: str) -> str:
        last_err = None
        for attempt in range(1, LLM_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content
            except Exception as err:  # pylint: disable=broad-except
                last_err = err
                if attempt == LLM_RETRIES:
                    raise
                time.sleep(BACKOFF_SECONDS * attempt)
        raise last_err  # type: ignore[misc]


def sanitize_suffix(raw_suffix: Optional[str]) -> str:
    if not raw_suffix:
        return ""
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw_suffix.strip())
    return sanitized.strip("-")


def default_basename(prefix: str, date_str: str, suffix: str) -> str:
    if suffix:
        return f"{prefix}_{suffix}_{date_str}.json"
    return f"{prefix}_{date_str}.json"


def resolve_path(raw: Optional[str], default_dir: Path, prefix: str, date_str: str, suffix: str) -> Path:
    if raw:
        path = Path(raw)
        if path.is_dir():
            return path / default_basename(prefix, date_str, suffix)
        return path
    return default_dir / default_basename(prefix, date_str, suffix)


def build_config(args: argparse.Namespace) -> SummaryConfig:
    date = args.date
    suffix = sanitize_suffix(args.suffix)
    papers_path = resolve_path(args.papers_path, DEFAULT_INPUT_DIR, "arxiv", date, suffix)
    clusters_path = resolve_path(args.clusters_path, DEFAULT_INPUT_DIR, "clusters", date, suffix)
    output_path = resolve_path(args.output_path, DEFAULT_OUTPUT_DIR, "cluster_summaries", date, suffix)

    return SummaryConfig(
        date=date,
        papers_path=papers_path,
        clusters_path=clusters_path,
        output_path=output_path,
        model=args.model or DEFAULT_MODEL,
        max_papers=args.max_papers,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
        temperature=args.temperature,
    )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def chunk_list(items: Sequence, chunk_size: int) -> List[Sequence]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def prepare_paper_text(paper: Dict) -> str:
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    return f"- 标题: {title}\n  摘要: {abstract}"


def summarize_chunk(papers_text: str, llm: LLMClient) -> Dict:
    prompt = CHUNK_PROMPT_TEMPLATE.format(papers=papers_text)
    response_text = llm.chat_completion(prompt)
    return parse_json_response(response_text)


def aggregate_chunks(chunk_summaries: List[Dict], llm: LLMClient) -> Dict:
    chunks_text = json.dumps(chunk_summaries, ensure_ascii=False, indent=2)
    prompt = AGGREGATE_PROMPT_TEMPLATE.format(chunks=chunks_text)
    response_text = llm.chat_completion(prompt)
    return parse_json_response(response_text)


def parse_json_response(text: str) -> Dict:
    text = text.strip()
    # Trim markdown fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip(":").strip()
    # Escape lone backslashes that break JSON (e.g., LaTeX fragments like \ell)
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {text}") from exc


def summarize_cluster(
    cluster_name: str,
    cluster_info: Dict,
    paper_lookup: Dict[str, Dict],
    config: SummaryConfig,
    llm: LLMClient,
) -> Tuple[str, Dict]:
    paper_ids = cluster_info.get("paper_ids", [])
    selected_ids = paper_ids[: config.max_papers]
    papers = [paper_lookup[pid] for pid in selected_ids if pid in paper_lookup]

    if not papers:
        return cluster_name, {
            "size": 0,
            "summary": "未找到对应论文数据，无法生成主题摘要。",
            "keywords": [],
        }

    paper_texts = [prepare_paper_text(paper) for paper in papers]
    chunks = chunk_list(paper_texts, config.chunk_size) or [paper_texts]

    chunk_summaries = []
    for chunk in chunks:
        text_block = "\n\n".join(chunk)
        chunk_summary = summarize_chunk(text_block, llm)
        chunk_summaries.append(chunk_summary)

    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        final_summary = aggregate_chunks(chunk_summaries, llm)

    keywords = final_summary.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()]

    result = {
        "size": len(papers),
        "research_question": final_summary.get("research_question", ""),
        "methods": final_summary.get("methods", ""),
        "trends": final_summary.get("trends", ""),
        "summary": final_summary.get("summary", ""),
        "keywords": keywords,
    }
    return cluster_name, result


def process_clusters(config: SummaryConfig) -> Dict[str, Dict]:
    papers = load_json(config.papers_path)
    clusters_payload = load_json(config.clusters_path)
    clusters = clusters_payload.get("clusters", {})

    paper_lookup = {paper["id"]: paper for paper in papers}
    llm = LLMClient(config.model, config.temperature)

    summaries: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(summarize_cluster, name, info, paper_lookup, config, llm)
            for name, info in clusters.items()
        ]
        for future in as_completed(futures):
            name, summary = future.result()
            summaries[name] = summary
    return summaries


def save_summaries(summaries: Dict[str, Dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, ensure_ascii=False, indent=2)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate LLM summaries for topic clusters.")
    parser.add_argument("--date", required=True, help="Date string used to locate input/output files.")
    parser.add_argument("--papers-path", help="Path or directory of Phase 1 JSON data.")
    parser.add_argument("--clusters-path", help="Path or directory of cluster JSON data.")
    parser.add_argument("--output-path", help="Output directory or file for summaries.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI chat model to use.")
    parser.add_argument("--max-papers", type=int, default=DEFAULT_MAX_PAPERS, help="Max papers included per cluster.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Papers per LLM chunk.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers for API calls.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--suffix", help="Optional suffix for locating default files (e.g., category tag).")
    return parser


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    config = build_config(args)

    print(f"Loading papers from {config.papers_path}")
    print(f"Reading clusters from {config.clusters_path}")
    print(f"Using model {config.model} with up to {config.max_workers} parallel workers.")

    summaries = process_clusters(config)
    output_path = save_summaries(summaries, config.output_path)

    print(f"Saved cluster summaries to {output_path}")


if __name__ == "__main__":
    main()
