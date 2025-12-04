import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_INPUT_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("reports")


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


def resolve_cluster_summaries_path(date: str, suffix: str, override: Optional[str]) -> Path:
    if override:
        return Path(override)
    candidates = [
        DEFAULT_INPUT_DIR / build_filename("cluster_summaries", date, suffix),
        DEFAULT_INPUT_DIR / build_filename("clusters_summaries", date, suffix),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output(raw: Optional[str], date: str, suffix: str, format_suffix: str) -> Path:
    name_suffix = f"_{suffix}" if suffix else ""
    if raw:
        p = Path(raw)
        if p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
            return p / f"report{name_suffix}_{date}.{format_suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"report{name_suffix}_{date}.{format_suffix}"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _infer_clusters_metadata(summary_path: Path) -> Dict:
    name = summary_path.name
    if name.startswith("cluster_summaries"):
        try:
            alternate = "clusters" + name[len("cluster_summaries") :]
            candidate = summary_path.parent / alternate
            if candidate.exists():
                clusters_data = load_json(candidate)
                return clusters_data.get("metadata", {})
        except Exception:
            pass
    return {}


def _escape_markdown(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "`": "\\`",
        "*": "\\*",
        "_": "\\_",
        "[": "\\[",
        "]": "\\]",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _ensure_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _format_cluster_title(cluster: Dict) -> str:
    title = (cluster.get("title") or "").strip()
    if title:
        return title
    keywords = cluster.get("keywords") or []
    filtered = [kw for kw in keywords if kw]
    if filtered:
        return " / ".join(filtered[:5])
    return f"Cluster {cluster.get('cluster_id', '')}"


def _safe_html(text: str) -> str:
    return html.escape(text or "")


def _render_keywords_markdown(keywords: Sequence[str]) -> Optional[str]:
    keywords = [kw for kw in keywords if kw]
    if not keywords:
        return None
    escaped = [f"`{_escape_markdown(kw)}`" for kw in keywords]
    return ", ".join(escaped)


def _render_cluster_markdown(cluster: Dict) -> List[str]:
    lines: List[str] = []
    cid = cluster.get("cluster_id", "")
    title = _format_cluster_title(cluster)
    lines.append(f"## 🪐 Cluster {cid}: {_escape_markdown(title)}")

    summary = cluster.get("summary", "") or ""
    if summary.strip():
        lines.append("")
        lines.append("### 🔍 Summary")
        lines.append(_escape_markdown(summary.strip()))

    research_question = cluster.get("research_question", "") or ""
    if research_question.strip():
        lines.append("")
        lines.append("### 🎯 Research Question")
        lines.append(_escape_markdown(research_question.strip()))

    methods = _ensure_list(cluster.get("methods"))
    if methods:
        lines.append("")
        lines.append("### 🧪 Methods")
        for method in methods:
            lines.append(f"- {_escape_markdown(method)}")

    trends = _ensure_list(cluster.get("trends"))
    if trends:
        lines.append("")
        lines.append("### 📈 Trends")
        for trend in trends:
            lines.append(f"- {_escape_markdown(trend)}")

    keywords = cluster.get("keywords") or []
    kw_line = _render_keywords_markdown(keywords)
    if kw_line:
        lines.append("")
        lines.append("### 🏷 Keywords")
        lines.append(kw_line)

    subclusters = cluster.get("subclusters") or []
    if subclusters:
        lines.append("")
        lines.extend(_render_subclusters_markdown(subclusters))

    return lines


def _render_report_markdown(date: str, metadata: Dict, cluster_summaries: List[Dict], total_papers: int) -> str:
    formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y年%m月%d日")
    lines = [
        f"# Daily ArXiv Research Digest — {formatted_date}",
        "",
        "## Overview",
        f"- Total papers: {metadata.get('paper_count_input', total_papers)}",
        f"- Clusters: {metadata.get('cluster_count', len(cluster_summaries))}",
        f"- Noise before filtering: {metadata.get('noise_pre_filtered', 'N/A')}",
        f"- Noise after filtering: {metadata.get('noise_hdbscan', 'N/A')}",
        "",
        "---",
        "",
        "## Clustered Research Topics",
        "",
    ]

    for cluster in cluster_summaries:
        lines.extend(_render_cluster_markdown(cluster))
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Footer")
    lines.append("Generated automatically from the DailyArxiv pipeline.")
    return "\n".join(lines)


def _render_report_html(date: str, metadata: Dict, cluster_summaries: List[Dict], total_papers: int) -> str:
    formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y年%m月%d日")
    lines = [
        f"<h1>Daily ArXiv Research Digest — {_safe_html(formatted_date)}</h1>",
        "<h2>Overview</h2>",
        "<ul>",
        f"<li>Total papers: {_safe_html(str(metadata.get('paper_count_input', total_papers)))}</li>",
        f"<li>Clusters: {_safe_html(str(metadata.get('cluster_count', len(cluster_summaries))))}</li>",
        f"<li>Noise before filtering: {_safe_html(str(metadata.get('noise_pre_filtered', 'N/A')))}</li>",
        f"<li>Noise after filtering: {_safe_html(str(metadata.get('noise_hdbscan', 'N/A')))}</li>",
        "</ul>",
        "<hr>",
        "<h2>Clustered Research Topics</h2>",
    ]

    for cluster in cluster_summaries:
        lines.extend(_render_cluster_html(cluster))
        lines.append("<hr>")

    lines.append("<h2>Footer</h2>")
    lines.append("<p>Generated automatically from the DailyArxiv pipeline.</p>")
    return "\n".join(lines)


def generate_reports(
    date: str,
    cluster_summaries_path: Path,
    trend_report_path: Path,
    papers_path: Path,
    output_dir: Optional[str],
    suffix: str,
    format_choice: str,
) -> Dict[str, Path]:
    cluster_summaries = load_json(cluster_summaries_path)
    trend_report = load_json(trend_report_path) if trend_report_path.exists() else {}
    papers = load_json(papers_path)
    metadata = _infer_clusters_metadata(cluster_summaries_path)

    clusters = list(cluster_summaries.values()) if isinstance(cluster_summaries, dict) else list(cluster_summaries)
    clusters_sorted = sorted(clusters, key=lambda item: item.get("cluster_id", 0))

    if format_choice == "html":
        content = _render_report_html(date, metadata, clusters_sorted, len(papers))
        output_path = resolve_output(output_dir, date, suffix, "html")
    else:
        content = _render_report_markdown(date, metadata, clusters_sorted, len(papers))
        output_path = resolve_output(output_dir, date, suffix, "md")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return {format_choice: output_path}


def _render_subclusters_markdown(subclusters: List[Dict]) -> List[str]:
    lines: List[str] = ["### 🌿 Subclusters", ""]
    for sub in subclusters:
        sub_id = sub.get("subcluster_id", "unknown")
        paper_count = sub.get("paper_count", 0)
        title = sub.get("title", "").strip() or f"Subcluster {sub_id}"
        summary = sub.get("summary", "").strip() or "(no summary available)"
        keywords = sub.get("keywords") or []
        sample_papers = _dedupe_preserve_order(sub.get("sample_papers") or [])

        lines.append(f"#### 🔹 Subcluster {sub_id}  ({paper_count} papers)")
        lines.append("")
        lines.append(f"**Summary:** {_escape_markdown(summary)}")
        if keywords:
            lines.append(f"**Keywords:** {', '.join([_escape_markdown(kw) for kw in keywords])}")
        if sample_papers:
            lines.append("")
            lines.append("**Sample papers:**")
            for title in sample_papers:
                lines.append(f"- {_escape_markdown(title)}")
        lines.append("")
    return lines


def _render_cluster_html(cluster: Dict) -> List[str]:
    lines: List[str] = []
    cid = cluster.get("cluster_id", "")
    title = _format_cluster_title(cluster)
    lines.append(f"<h2>🪐 Cluster {cid}: {_safe_html(title)}</h2>")

    summary = (cluster.get("summary") or "").strip()
    if summary:
        lines.append("<h3>🔍 Summary</h3>")
        lines.append(f"<p>{_safe_html(summary)}</p>")

    research_question = (cluster.get("research_question") or "").strip()
    if research_question:
        lines.append("<h3>🎯 Research Question</h3>")
        lines.append(f"<p>{_safe_html(research_question)}</p>")

    methods = _ensure_list(cluster.get("methods"))
    if methods:
        lines.append("<h3>🧪 Methods</h3>")
        lines.append("<ul>")
        for method in methods:
            lines.append(f"<li>{_safe_html(method)}</li>")
        lines.append("</ul>")

    trends = _ensure_list(cluster.get("trends"))
    if trends:
        lines.append("<h3>📈 Trends</h3>")
        lines.append("<ul>")
        for trend in trends:
            lines.append(f"<li>{_safe_html(trend)}</li>")
        lines.append("</ul>")

    keywords = cluster.get("keywords") or []
    keywords = [kw for kw in keywords if kw]
    if keywords:
        escaped = ", ".join(_safe_html(kw) for kw in keywords)
        lines.append("<h3>🏷 Keywords</h3>")
        lines.append(f"<p>{escaped}</p>")

    subclusters = cluster.get("subclusters") or []
    if subclusters:
        lines.extend(_render_subclusters_html(subclusters))

    return lines


def _render_subclusters_html(subclusters: List[Dict]) -> List[str]:
    lines: List[str] = ["<h3>🌿 Subclusters</h3>"]
    for sub in subclusters:
        sub_id = sub.get("subcluster_id", "unknown")
        paper_count = sub.get("paper_count", 0)
        title = sub.get("title", "").strip() or f"Subcluster {sub_id}"
        summary = sub.get("summary", "").strip() or "(no summary available)"
        keywords = [kw for kw in (sub.get("keywords") or []) if kw]
        sample_papers = _dedupe_preserve_order(sub.get("sample_papers") or [])

        lines.append(f"<h4>🔹 Subcluster {sub_id}  ({paper_count} papers)</h4>")
        lines.append(f"<p><strong>Summary:</strong> {_safe_html(summary)}</p>")
        if keywords:
            lines.append(f"<p><strong>Keywords:</strong> {', '.join(_safe_html(kw) for kw in keywords)}</p>")
        if sample_papers:
            lines.append("<p><strong>Sample papers:</strong></p>")
            lines.append("<ul>")
            for title in sample_papers:
                lines.append(f"<li>{_safe_html(title)}</li>")
            lines.append("</ul>")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate daily science trend reports in Markdown/HTML.")
    parser.add_argument("--date", required=True, help="Date to render (YYYY-MM-DD).")
    parser.add_argument("--cluster-summaries", help="Path to cluster_summaries_<date>.json.")
    parser.add_argument("--trend-report", help="Path to trend_report_<date>.json.")
    parser.add_argument("--papers", help="Path to arxiv_<date>.json.")
    parser.add_argument("--output", help="Output directory or exact file path (without extension).")
    parser.add_argument("--suffix", help="Optional suffix used for locating input/output files.")
    parser.add_argument("--format", choices=["markdown", "html"], default="markdown", help="Output format")
    return parser


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)

    suffix = sanitize_suffix(args.suffix)
    cluster_summaries_path = resolve_cluster_summaries_path(args.date, suffix, args.cluster_summaries)
    trend_report_path = resolve_path(args.trend_report, args.date, "trend_report", suffix)
    papers_path = resolve_path(args.papers, args.date, "arxiv", suffix)

    outputs = generate_reports(
        date=args.date,
        cluster_summaries_path=cluster_summaries_path,
        trend_report_path=trend_report_path,
        papers_path=papers_path,
        output_dir=args.output,
        suffix=suffix,
        format_choice=args.format,
    )

    for fmt, path in outputs.items():
        print(f"Generated {fmt} report at {path}")


if __name__ == "__main__":
    main()
