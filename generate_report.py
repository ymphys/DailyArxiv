import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from jinja2 import Environment, FileSystemLoader, Template

DEFAULT_INPUT_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_TEMPLATE_DIR = Path("templates")
MARKDOWN_TEMPLATE = "daily_report.md.j2"
HTML_TEMPLATE = "daily_report.html.j2"


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


def ensure_templates(template_dir: Path) -> Environment:
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    return env


def prepare_overview(papers: List[Dict], trend_report: Dict) -> Dict:
    total_papers = len(papers)
    hot_topics = trend_report.get("hot_topics", [])
    emerging = trend_report.get("emerging_topics", [])
    domain_insights = trend_report.get("domain_insights", {})

    growth_fields = [topic.get("theme") for topic in emerging[:3] if topic.get("theme")]
    new_clusters = [topic.get("cluster") for topic in emerging[:5]]

    return {
        "total_papers": total_papers,
        "major_growth_fields": growth_fields,
        "new_clusters": new_clusters,
        "hot_topics": hot_topics,
        "emerging_topics": emerging,
        "domain_insights": domain_insights,
    }


def sorted_clusters(cluster_summaries: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    sorted_items = sorted(
        cluster_summaries.items(),
        key=lambda item: item[1].get("size", 0),
        reverse=True,
    )
    return sorted_items


def render_template(
    template: Template,
    context: Dict,
    output_path: Path,
) -> Path:
    rendered = template.render(**context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def prepare_context(date: str, papers: List[Dict], cluster_summaries: Dict[str, Dict], trend_report: Dict) -> Dict:
    overview = prepare_overview(papers, trend_report)
    clusters_sorted = sorted_clusters(cluster_summaries)
    formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y年%m月%d日")
    return {
        "date": date,
        "date_display": formatted_date,
        "overview": overview,
        "clusters_sorted": clusters_sorted,
    }


def generate_reports(
    date: str,
    cluster_summaries_path: Path,
    trend_report_path: Path,
    papers_path: Path,
    template_dir: Path,
    output_dir: Optional[str],
    suffix: str,
) -> Dict[str, Path]:
    cluster_summaries = load_json(cluster_summaries_path)
    trend_report = load_json(trend_report_path)
    papers = load_json(papers_path)

    context = prepare_context(date, papers, cluster_summaries, trend_report)
    env = ensure_templates(template_dir)

    markdown_template = env.get_template(MARKDOWN_TEMPLATE)
    html_template = env.get_template(HTML_TEMPLATE)

    outputs = {}
    outputs["markdown"] = render_template(markdown_template, context, resolve_output(output_dir, date, suffix, "md"))
    outputs["html"] = render_template(html_template, context, resolve_output(output_dir, date, suffix, "html"))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate daily science trend reports in Markdown/HTML.")
    parser.add_argument("--date", required=True, help="Date to render (YYYY-MM-DD).")
    parser.add_argument("--cluster-summaries", help="Path to cluster_summaries_<date>.json.")
    parser.add_argument("--trend-report", help="Path to trend_report_<date>.json.")
    parser.add_argument("--papers", help="Path to arxiv_<date>.json.")
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR), help="Directory containing Jinja2 templates.")
    parser.add_argument("--output", help="Output directory or exact file path (without extension).")
    parser.add_argument("--suffix", help="Optional suffix used for locating input/output files.")
    return parser


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)

    suffix = sanitize_suffix(args.suffix)
    cluster_summaries_path = resolve_path(args.cluster_summaries, args.date, "cluster_summaries", suffix)
    trend_report_path = resolve_path(args.trend_report, args.date, "trend_report", suffix)
    papers_path = resolve_path(args.papers, args.date, "arxiv", suffix)
    template_dir = Path(args.template_dir)

    outputs = generate_reports(
        date=args.date,
        cluster_summaries_path=cluster_summaries_path,
        trend_report_path=trend_report_path,
        papers_path=papers_path,
        template_dir=template_dir,
        output_dir=args.output,
        suffix=suffix,
    )

    for fmt, path in outputs.items():
        print(f"Generated {fmt} report at {path}")


if __name__ == "__main__":
    main()
