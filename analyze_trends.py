import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_INPUT_DIR = Path("data")
DOMAIN_MAPPING = {
    "physics": ("astro-ph", "cond-mat", "gr-qc", "hep", "nucl", "quant-ph", "physics"),
    "cs": ("cs", "eess"),
    "econ_fin": ("econ", "q-fin"),
    "math": ("math",),
    "biology": ("q-bio",),
}
DOMAIN_DISPLAY = {
    "physics": "Physics & High-Energy",
    "cs": "Computer Science & AI",
    "econ_fin": "Economics & Quant Finance",
    "math": "Mathematics",
    "biology": "Quantitative Biology",
}


@dataclass(frozen=True)
class TrendConfig:
    date: str
    today_clusters: Path
    today_summaries: Path
    today_papers: Path
    yesterday_date: Optional[str]
    yesterday_clusters: Optional[Path]
    yesterday_summaries: Optional[Path]
    yesterday_papers: Optional[Path]
    output_path: Path
    today_suffix: str
    yesterday_suffix: str


def sanitize_suffix(raw_suffix: Optional[str]) -> str:
    if not raw_suffix:
        return ""
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw_suffix.strip())
    return sanitized.strip("-")


def build_filename(prefix: str, date: str, suffix: str) -> str:
    if suffix:
        return f"{prefix}_{suffix}_{date}.json"
    return f"{prefix}_{date}.json"


def resolve_path(raw: Optional[str], date: str, prefix: str, suffix: str) -> Path:
    if raw:
        p = Path(raw)
        if p.is_dir():
            return p / build_filename(prefix, date, suffix)
        return p
    return DEFAULT_INPUT_DIR / build_filename(prefix, date, suffix)


def load_json(path: Optional[Path]) -> Optional[dict]:
    if not path:
        return None
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_summary_theme(summary: str) -> str:
    if not summary:
        return ""
    sentences = summary.split("。")
    first_sentence = sentences[0].strip()
    if not first_sentence:
        return summary[:80]
    return first_sentence + "。"


def normalize_keywords(keywords) -> List[str]:
    if keywords is None:
        return []
    if isinstance(keywords, str):
        return [kw.strip() for kw in keywords.split(",") if kw.strip()]
    if isinstance(keywords, list):
        clean = []
        for kw in keywords:
            if kw:
                clean.append(str(kw).strip())
        return clean
    return []


def build_hot_topics(clusters: Dict[str, dict], summaries: Dict[str, dict]) -> List[dict]:
    entries: List[Tuple[str, int]] = []
    for name, info in clusters.items():
        size = len(info.get("paper_ids", []))
        entries.append((name, size))
    entries.sort(key=lambda x: x[1], reverse=True)
    top_entries = entries[:10]

    hot_topics = []
    for name, size in top_entries:
        summary_info = summaries.get(name, {})
        summary_text = summary_info.get("summary", "")
        keywords = normalize_keywords(summary_info.get("keywords"))
        hot_topics.append(
            {
                "cluster": name,
                "size": size,
                "theme": extract_summary_theme(summary_text),
                "keywords": keywords,
            }
        )
    return hot_topics


def build_emerging_topics(
    today_clusters: Dict[str, dict],
    yesterday_clusters: Optional[Dict[str, dict]],
    summaries: Dict[str, dict],
) -> List[dict]:
    if not yesterday_clusters:
        return []

    growth_entries = []
    for name, info in today_clusters.items():
        today_size = len(info.get("paper_ids", []))
        yesterday_size = len(yesterday_clusters.get(name, {}).get("paper_ids", [])) if yesterday_clusters else 0
        growth_rate = (today_size - yesterday_size) / (yesterday_size + 1)
        growth_entries.append((name, today_size, yesterday_size, growth_rate, summaries.get(name, {})))

    growth_entries.sort(key=lambda x: x[3], reverse=True)
    return [
        {
            "cluster": name,
            "size_today": today_size,
            "size_yesterday": yesterday_size,
            "growth_rate": growth_rate,
            "theme": extract_summary_theme(summary_info.get("summary", "")),
        }
        for name, today_size, yesterday_size, growth_rate, summary_info in growth_entries[:10]
    ]


def categorize_primary(primary: str) -> str:
    if not primary:
        return "unknown"
    for domain, prefixes in DOMAIN_MAPPING.items():
        if any(primary.startswith(prefix) for prefix in prefixes):
            return domain
    return "other"


def build_domain_insights(
    today_clusters: Dict[str, dict],
    summaries: Dict[str, dict],
    paper_lookup: Dict[str, dict],
    yesterday_clusters: Optional[Dict[str, dict]] = None,
    yesterday_lookup: Optional[Dict[str, dict]] = None,
) -> Dict[str, dict]:
    def cluster_domain(info: dict, lookup: Dict[str, dict]) -> str:
        paper_ids = info.get("paper_ids", [])
        categories = [lookup.get(pid, {}).get("primary_category", "") for pid in paper_ids]
        mapped = [categorize_primary(cat) for cat in categories if cat]
        if not mapped:
            return "unknown"
        domain_counter = Counter(mapped)
        return domain_counter.most_common(1)[0][0]

    domain_clusters: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    total_papers = 0
    for name, info in today_clusters.items():
        size = len(info.get("paper_ids", []))
        total_papers += size
        domain = cluster_domain(info, paper_lookup)
        domain_clusters[domain].append((name, size))

    yesterday_domain_share: Dict[str, float] = {}
    if yesterday_clusters and yesterday_lookup:
        yesterday_total = 0
        domain_counts = defaultdict(int)
        for info in yesterday_clusters.values():
            size = len(info.get("paper_ids", []))
            yesterday_total += size
            domain = cluster_domain(info, yesterday_lookup)
            domain_counts[domain] += size
        if yesterday_total:
            for domain, count in domain_counts.items():
                yesterday_domain_share[domain] = count / yesterday_total

    domain_insights = {}
    for domain, clusters_list in domain_clusters.items():
        clusters_list.sort(key=lambda x: x[1], reverse=True)
        top_cluster_name = clusters_list[0][0]
        summary_info = summaries.get(top_cluster_name, {})
        total_size = sum(size for _, size in clusters_list)
        share = (total_size / total_papers) if total_papers else 0
        prev_share = yesterday_domain_share.get(domain)
        if prev_share is None:
            trend_note = "缺少历史对比数据，暂无法评估环比趋势。"
        else:
            delta = share - prev_share
            trend_note = f"领域占比{'上升' if delta >= 0 else '下降'} {abs(delta)*100:.1f} 个百分点（由 {prev_share*100:.1f}% 变为 {share*100:.1f}%）。"
        domain_insights[domain] = {
            "display_name": DOMAIN_DISPLAY.get(domain, domain),
            "top_clusters": [
                {"cluster": name, "size": size, "theme": extract_summary_theme(summaries.get(name, {}).get("summary", ""))}
                for name, size in clusters_list[:3]
            ],
            "representative_cluster": top_cluster_name,
            "representative_theme": extract_summary_theme(summary_info.get("summary", "")),
            "hot_focus": extract_summary_theme(summary_info.get("summary", "")),
            "keywords": normalize_keywords(summary_info.get("keywords")),
            "total_papers": total_size,
            "share": round(share, 4),
            "trend_note": trend_note,
        }
    return domain_insights


def build_report(config: TrendConfig) -> dict:
    today_clusters_payload = load_json(config.today_clusters)
    cluster_summaries = load_json(config.today_summaries)
    today_papers = load_json(config.today_papers)
    yesterday_clusters_payload = load_json(config.yesterday_clusters) if config.yesterday_clusters else None
    yesterday_papers = load_json(config.yesterday_papers) if config.yesterday_papers else None

    if not today_clusters_payload or not cluster_summaries or not today_papers:
        raise ValueError("Missing required data for trend analysis.")

    today_clusters = today_clusters_payload.get("clusters", {})
    yesterday_clusters = yesterday_clusters_payload.get("clusters", {}) if yesterday_clusters_payload else {}

    hot_topics = build_hot_topics(today_clusters, cluster_summaries)
    emerging = build_emerging_topics(today_clusters, yesterday_clusters, cluster_summaries)

    paper_lookup = {paper["id"]: paper for paper in today_papers}
    yesterday_lookup = {paper["id"]: paper for paper in yesterday_papers} if isinstance(yesterday_papers, list) else None
    domain_insights = build_domain_insights(
        today_clusters,
        cluster_summaries,
        paper_lookup,
        yesterday_clusters,
        yesterday_lookup,
    )

    return {
        "date": config.date,
        "hot_topics": hot_topics,
        "emerging_topics": emerging,
        "domain_insights": domain_insights,
    }


def save_report(report: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze topic clusters to produce daily trend reports.")
    parser.add_argument("--date", required=True, help="Target date (YYYY-MM-DD).")
    parser.add_argument("--today-clusters", help="Path to today's clusters JSON.")
    parser.add_argument("--today-summaries", help="Path to today's cluster summaries JSON.")
    parser.add_argument("--today-papers", help="Path to today's papers JSON.")
    parser.add_argument("--yesterday-date", help="Date string for comparison (defaults to date-1).")
    parser.add_argument("--yesterday-clusters", help="Path to yesterday's clusters JSON.")
    parser.add_argument("--yesterday-summaries", help="Path to yesterday's cluster summaries JSON.")
    parser.add_argument("--yesterday-papers", help="Path to yesterday's papers JSON.")
    parser.add_argument("--suffix", help="Suffix used for today's default filenames.")
    parser.add_argument("--yesterday-suffix", help="Suffix used for yesterday's default filenames (defaults to today's suffix).")
    parser.add_argument("--output-path", help="Output file path for trend report.")
    return parser


def build_config(args: argparse.Namespace) -> TrendConfig:
    date = args.date
    today_suffix = sanitize_suffix(args.suffix)
    yesterday_suffix = sanitize_suffix(args.yesterday_suffix) if args.yesterday_suffix else today_suffix

    today_clusters = resolve_path(args.today_clusters, date, "clusters", today_suffix)
    today_summaries = resolve_path(args.today_summaries, date, "cluster_summaries", today_suffix)
    today_papers = resolve_path(args.today_papers, date, "arxiv", today_suffix)

    if args.yesterday_date:
        yesterday_date = args.yesterday_date
    else:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        yesterday_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")

    yesterday_clusters = resolve_path(args.yesterday_clusters, yesterday_date, "clusters", yesterday_suffix)
    yesterday_summaries = resolve_path(args.yesterday_summaries, yesterday_date, "cluster_summaries", yesterday_suffix)
    yesterday_papers = resolve_path(args.yesterday_papers, yesterday_date, "arxiv", yesterday_suffix)

    output_path = resolve_path(args.output_path, date, "trend_report", today_suffix)
    return TrendConfig(
        date=date,
        today_clusters=today_clusters,
        today_summaries=today_summaries,
        today_papers=today_papers,
        yesterday_date=yesterday_date,
        yesterday_clusters=yesterday_clusters,
        yesterday_summaries=yesterday_summaries,
        yesterday_papers=yesterday_papers,
        output_path=output_path,
        today_suffix=today_suffix,
        yesterday_suffix=yesterday_suffix,
    )


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    config = build_config(args)
    report = build_report(config)
    output_path = save_report(report, config.output_path)
    print(f"Saved trend report to {output_path}")


if __name__ == "__main__":
    main()
