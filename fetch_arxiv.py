import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_CATEGORIES = [
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "nlin",
    "nucl-ex",
    "nucl-th",
    "physics",
    "quant-ph",
    "math",
    "cs",
    "q-bio",
    "q-fin",
    "stat",
    "eess",
    "econ",
]
LISTING_DATE_PATTERN = re.compile(r"Showing new listings for (?P<date>.+)", re.IGNORECASE)


class ArxivFetchError(RuntimeError):
    """Raised when arXiv pages repeatedly fail."""


@dataclass(frozen=True)
class FetchConfig:
    categories: Sequence[str]
    target_date: date
    max_results: int
    output_path: Path
    retries: int
    backoff: float
    suffix: str


def parse_date_arg(date_arg: str) -> date:
    """Resolve a --date argument into a concrete date in UTC+8."""
    normalized = date_arg.strip().lower()
    now_local = datetime.now(ASIA_SHANGHAI).date()

    if normalized in ("today", ""):
        return now_local
    if normalized == "yesterday":
        return now_local - timedelta(days=1)

    try:
        return datetime.strptime(date_arg, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must be 'today', 'yesterday', or formatted as YYYY-MM-DD"
        ) from exc


def parse_listing_date(soup: BeautifulSoup) -> Optional[date]:
    """Extract the listing date from the page heading."""
    heading = soup.select_one("#dlpage > h3")
    if not heading:
        return None
    match = LISTING_DATE_PATTERN.search(heading.get_text(strip=True))
    if not match:
        return None
    try:
        return datetime.strptime(match.group("date"), "%A, %d %B %Y").date()
    except ValueError:
        return None


def format_listing_datetime(listing_date: date) -> str:
    local_dt = datetime.combine(listing_date, dt_time.min, tzinfo=ASIA_SHANGHAI)
    return local_dt.isoformat()


def extract_primary_category(raw_text: str, fallback: str) -> str:
    match = re.search(r"\(([^)]+)\)", raw_text)
    return match.group(1) if match else fallback


def parse_entry(dt_node, dd_node, category: str, listing_date: date) -> Optional[dict]:
    id_link = dt_node.find("a", title="Abstract")
    if not id_link:
        return None
    paper_id = id_link.get_text(strip=True)
    if not paper_id:
        return None

    title_div = dd_node.find("div", class_="list-title")
    if not title_div:
        return None
    title = title_div.get_text(" ", strip=True).replace("Title:", "", 1).strip()

    abstract_p = dd_node.find("p", class_="mathjax")
    abstract = abstract_p.get_text(" ", strip=True) if abstract_p else ""

    authors = [a.get_text(strip=True) for a in dd_node.select("div.list-authors a")]

    primary_subject_span = dd_node.select_one("span.primary-subject")
    if primary_subject_span:
        primary_category = extract_primary_category(primary_subject_span.get_text(" ", strip=True), category)
    else:
        primary_category = category

    iso_ts = format_listing_datetime(listing_date)

    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "primary_category": primary_category,
        "published": iso_ts,
        "updated": iso_ts,
    }


def fetch_category_entries(category: str, session: requests.Session, config: FetchConfig) -> List[dict]:
    url = f"https://arxiv.org/list/{category}/new"
    html = None
    for attempt in range(1, config.retries + 1):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            html = response.text
            break
        except requests.RequestException as exc:
            if attempt == config.retries:
                raise ArxivFetchError(f"Failed to load {url} after {config.retries} attempts") from exc
            time.sleep(config.backoff * attempt)

    soup = BeautifulSoup(html, "html.parser")
    listing_date = parse_listing_date(soup) or config.target_date

    dt_nodes = soup.select("dl#articles dt")
    dd_nodes = soup.select("dl#articles dd")

    entries: List[dict] = []
    for dt_node, dd_node in zip(dt_nodes, dd_nodes):
        parsed = parse_entry(dt_node, dd_node, category, listing_date)
        if parsed:
            entries.append(parsed)

    return entries


def save_results(results: Iterable[dict], save_path: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as fh:
        json.dump(list(results), fh, ensure_ascii=False, indent=2)
    return save_path


def sanitize_suffix(raw_suffix: Optional[str]) -> str:
    if not raw_suffix:
        return ""
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw_suffix.strip())
    return sanitized.strip("-")


def parse_categories(raw: Optional[Sequence[str]]) -> List[str]:
    if not raw or raw == ["all"]:
        return DEFAULT_CATEGORIES
    cleaned = [cat.strip() for cat in raw if cat.strip()]
    return cleaned or DEFAULT_CATEGORIES


def resolve_output_path(raw_path: str, target_date: date, suffix: str) -> Path:
    path_obj = Path(raw_path)
    if path_obj.suffix.lower() == ".json":
        return path_obj
    filename = f"arxiv_{target_date.isoformat()}.json"
    if suffix:
        filename = f"arxiv_{suffix}_{target_date.isoformat()}.json"
    return path_obj / filename


def build_config(args: argparse.Namespace) -> FetchConfig:
    categories = parse_categories(args.categories)
    target_date = parse_date_arg(args.date)
    suffix = sanitize_suffix(args.suffix)
    if not suffix and args.categories:
        suffix = sanitize_suffix("-".join(args.categories))
    output_path = resolve_output_path(args.save_path, target_date, suffix)

    return FetchConfig(
        categories=categories,
        target_date=target_date,
        max_results=args.max_results,
        output_path=output_path,
        retries=args.retries,
        backoff=args.backoff,
        suffix=suffix,
    )


def gather_entries(config: FetchConfig) -> List[dict]:
    session = requests.Session()
    collected: List[dict] = []

    for category in config.categories:
        category_entries = fetch_category_entries(category, session, config)
        collected.extend(category_entries)
        if config.max_results and len(collected) >= config.max_results:
            break

    if config.max_results and len(collected) > config.max_results:
        return collected[: config.max_results]
    return collected


def run_with_config(config: FetchConfig) -> Path:
    entries = gather_entries(config)
    return save_results(entries, config.output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch daily arXiv papers and store them as JSON.")
    parser.add_argument("--date", default="today", help="Date to fetch (today, yesterday, or YYYY-MM-DD, local UTC+8).")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="One or more arXiv categories (e.g., cs.CL stat.ML). Defaults to all categories.",
    )
    parser.add_argument("--max-results", type=int, default=3000, help="Maximum number of papers to fetch.")
    parser.add_argument(
        "--save-path",
        default="data",
        help="Directory or JSON file path for output; directories will create arxiv_<date>.json.",
    )
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts for network errors.")
    parser.add_argument("--backoff", type=float, default=2.0, help="Backoff multiplier between retries in seconds.")
    parser.add_argument("--suffix", help="Optional suffix inserted into generated filenames (e.g., category tag).")
    return parser


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    config = build_config(args)

    print(f"Fetching arXiv papers for {config.target_date.isoformat()} (UTC+8) ...")
    if config.categories:
        print(f"Categories: {', '.join(config.categories)}")
    else:
        print("Categories: all")
    print(f"Max results: {config.max_results}")

    try:
        output_path = run_with_config(config)
    except ArxivFetchError as exc:
        print(f"Failed to fetch arXiv data: {exc}")
        raise SystemExit(1)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
