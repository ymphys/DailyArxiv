import argparse
import re
import sys
from datetime import datetime, timedelta
from typing import List, Optional

import analyze_trends
import cluster_topics
import fetch_arxiv
import generate_report
import summarize_clusters


def sanitize_suffix(raw: Optional[str]) -> str:
    if not raw:
        return ""
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw.strip())
    return sanitized.strip("-")


def categories_to_suffix(categories: Optional[List[str]]) -> str:
    if not categories:
        return ""
    return sanitize_suffix("-".join(categories))


def build_fetch_args(args: argparse.Namespace, suffix: str) -> List[str]:
    cli_args = ["--date", args.date]
    if args.categories:
        cli_args.append("--categories")
        cli_args.extend(args.categories)
    if args.fetch_max_results:
        cli_args.extend(["--max-results", str(args.fetch_max_results)])
    if args.fetch_save_path:
        cli_args.extend(["--save-path", args.fetch_save_path])
    if args.fetch_retries:
        cli_args.extend(["--retries", str(args.fetch_retries)])
    if args.fetch_backoff:
        cli_args.extend(["--backoff", str(args.fetch_backoff)])
    if suffix:
        cli_args.extend(["--suffix", suffix])
    return cli_args


def build_cluster_args(args: argparse.Namespace, suffix: str) -> List[str]:
    cli_args = ["--date", args.date, "--backend", args.embed_backend]
    if args.embed_model:
        cli_args.extend(["--model", args.embed_model])
    if args.embed_batch_size:
        cli_args.extend(["--batch-size", str(args.embed_batch_size)])
    if args.embed_min_cluster_size:
        cli_args.extend(["--min-cluster-size", str(args.embed_min_cluster_size)])
    if args.embed_device:
        cli_args.extend(["--device", args.embed_device])
    if args.embed_force_kmeans:
        cli_args.append("--force-kmeans")
    if suffix:
        cli_args.extend(["--suffix", suffix])
    return cli_args


def build_summary_args(args: argparse.Namespace, suffix: str) -> List[str]:
    cli_args = ["--date", args.date]
    if args.summary_model:
        cli_args.extend(["--model", args.summary_model])
    if args.summary_max_papers:
        cli_args.extend(["--max-papers", str(args.summary_max_papers)])
    if args.summary_chunk_size:
        cli_args.extend(["--chunk-size", str(args.summary_chunk_size)])
    if args.summary_max_workers:
        cli_args.extend(["--max-workers", str(args.summary_max_workers)])
    if args.summary_temperature is not None:
        cli_args.extend(["--temperature", str(args.summary_temperature)])
    if args.summary_papers_path:
        cli_args.extend(["--papers-path", args.summary_papers_path])
    if args.summary_clusters_path:
        cli_args.extend(["--clusters-path", args.summary_clusters_path])
    if args.summary_output_path:
        cli_args.extend(["--output-path", args.summary_output_path])
    if suffix:
        cli_args.extend(["--suffix", suffix])
    return cli_args


def build_trend_args(args: argparse.Namespace, suffix: str, yesterday_suffix: str) -> List[str]:
    yesterday = args.yesterday_date or (datetime.strptime(args.date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    cli_args = ["--date", args.date, "--yesterday-date", yesterday]
    if args.trend_today_clusters:
        cli_args.extend(["--today-clusters", args.trend_today_clusters])
    if args.trend_today_summaries:
        cli_args.extend(["--today-summaries", args.trend_today_summaries])
    if args.trend_today_papers:
        cli_args.extend(["--today-papers", args.trend_today_papers])
    if args.trend_yesterday_clusters:
        cli_args.extend(["--yesterday-clusters", args.trend_yesterday_clusters])
    if args.trend_yesterday_summaries:
        cli_args.extend(["--yesterday-summaries", args.trend_yesterday_summaries])
    if args.trend_yesterday_papers:
        cli_args.extend(["--yesterday-papers", args.trend_yesterday_papers])
    if args.trend_output_path:
        cli_args.extend(["--output-path", args.trend_output_path])
    if suffix:
        cli_args.extend(["--suffix", suffix])
    if yesterday_suffix:
        cli_args.extend(["--yesterday-suffix", yesterday_suffix])
    return cli_args


def build_report_args(args: argparse.Namespace, suffix: str) -> List[str]:
    cli_args = ["--date", args.date]
    if args.report_cluster_summaries:
        cli_args.extend(["--cluster-summaries", args.report_cluster_summaries])
    if args.report_trend_report:
        cli_args.extend(["--trend-report", args.report_trend_report])
    if args.report_papers:
        cli_args.extend(["--papers", args.report_papers])
    if args.report_template_dir:
        cli_args.extend(["--template-dir", args.report_template_dir])
    if args.report_output:
        cli_args.extend(["--output", args.report_output])
    if suffix:
        cli_args.extend(["--suffix", suffix])
    return cli_args


def run_phase(name: str, func, cli_args: List[str]) -> None:
    print(f"=== Running {name} ===")
    func(cli_args)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Daily arXiv pipeline orchestrator: fetch -> cluster -> summarize -> trends -> report."
    )
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Target date for the workflow.")
    parser.add_argument("--yesterday-date", help="Override yesterday's date for comparison.")
    parser.add_argument("--run-suffix", help="Custom suffix inserted into default filenames (e.g., category tag).")
    parser.add_argument("--yesterday-run-suffix", help="Suffix for yesterday's data (defaults to run suffix).")

    # Skip flags
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-trend", action="store_true")
    parser.add_argument("--skip-report", action="store_true")

    # Fetch options
    parser.add_argument("--categories", nargs="+", help="Categories for fetch_arxiv.")
    parser.add_argument("--fetch-max-results", type=int, help="Max results for fetch_arxiv.")
    parser.add_argument("--fetch-save-path", help="Override fetch output path (file or dir).")
    parser.add_argument("--fetch-retries", type=int, help="Retry attempts for fetch_arxiv.")
    parser.add_argument("--fetch-backoff", type=float, help="Backoff multiplier for fetch_arxiv.")

    # Embedding/cluster options
    parser.add_argument("--embed-backend", choices=["openai", "huggingface"], default="openai")
    parser.add_argument("--embed-model", help="Embedding model name.")
    parser.add_argument("--embed-batch-size", type=int, help="Embedding batch size.")
    parser.add_argument("--embed-min-cluster-size", type=int, help="Minimum cluster size.")
    parser.add_argument("--embed-device", help="Device for huggingface embeddings.")
    parser.add_argument("--embed-force-kmeans", action="store_true", help="Force kmeans clustering.")

    # Summaries
    parser.add_argument("--summary-model", help="LLM model for summarization.")
    parser.add_argument("--summary-max-papers", type=int, help="Max papers per cluster for summarization.")
    parser.add_argument("--summary-chunk-size", type=int, help="Chunk size for summarization.")
    parser.add_argument("--summary-max-workers", type=int, help="Parallel workers for summarization.")
    parser.add_argument("--summary-temperature", type=float, help="LLM temperature.")
    parser.add_argument("--summary-papers-path", help="Override papers path for summarization.")
    parser.add_argument("--summary-clusters-path", help="Override clusters path for summarization.")
    parser.add_argument("--summary-output-path", help="Override summary output path.")

    # Trend analysis paths
    parser.add_argument("--trend-today-clusters", help="Override today's clusters path.")
    parser.add_argument("--trend-today-summaries", help="Override today's summaries path.")
    parser.add_argument("--trend-today-papers", help="Override today's papers path.")
    parser.add_argument("--trend-yesterday-clusters", help="Override yesterday's clusters path.")
    parser.add_argument("--trend-yesterday-summaries", help="Override yesterday's summaries path.")
    parser.add_argument("--trend-yesterday-papers", help="Override yesterday's papers path.")
    parser.add_argument("--trend-output-path", help="Override trend report output path.")

    # Report options
    parser.add_argument("--report-cluster-summaries", help="Override path to cluster summaries for report.")
    parser.add_argument("--report-trend-report", help="Override path to trend report for report.")
    parser.add_argument("--report-papers", help="Override path to papers for report.")
    parser.add_argument("--report-template-dir", help="Templates dir for report generation.")
    parser.add_argument("--report-output", help="Output directory/path for final reports.")

    args = parser.parse_args(argv)

    run_suffix = sanitize_suffix(args.run_suffix)
    if not run_suffix and args.categories:
        run_suffix = categories_to_suffix(args.categories)
    yesterday_suffix = sanitize_suffix(args.yesterday_run_suffix) or run_suffix

    try:
        if not args.skip_fetch:
            run_phase("fetch_arxiv", fetch_arxiv.main, build_fetch_args(args, run_suffix))
        if not args.skip_cluster:
            run_phase("cluster_topics", cluster_topics.main, build_cluster_args(args, run_suffix))
        if not args.skip_summary:
            run_phase("summarize_clusters", summarize_clusters.main, build_summary_args(args, run_suffix))
        if not args.skip_trend:
            run_phase("analyze_trends", analyze_trends.main, build_trend_args(args, run_suffix, yesterday_suffix))
        if not args.skip_report:
            run_phase("generate_report", generate_report.main, build_report_args(args, run_suffix))
    except SystemExit as exc:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Workflow failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
