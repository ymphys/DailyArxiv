#!/usr/bin/env python3
"""Cluster arXiv papers with improved preprocessing, embeddings, and summarization."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from cluster import ClusterConfig, ClusterGroup, ClusterResult, perform_clustering
from embed import DEFAULT_FALLBACK_MODEL, DEFAULT_CACHE, EmbeddingConfig, embed_texts
from logging_setup import configure_logging
from preprocess import preprocess_text
from summarize import SummarizerConfig, summarize_cluster

DEFAULT_INPUT_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_BACKEND = "openai"
DEFAULT_SUMMARY_MODEL = "gpt-4o-mini"

LOGGER = logging.getLogger("dailyarxiv.cluster_topics")


def _compute_umap_projection(embeddings: np.ndarray, n_components: int) -> Optional[np.ndarray]:
    try:
        import umap
    except ImportError as exc:
        LOGGER.warning("umap-learn not installed (%s); skipping %dD projection.", exc, n_components)
        return None

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _assign_colors(labels: np.ndarray, plt_module):
    base = plt_module.cm.get_cmap("tab20")
    mapping = {}
    color_idx = 0
    mapping[-1] = (0.5, 0.5, 0.5, 1.0)
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mapping[label] = base(color_idx % base.N)
        color_idx += 1
    return np.array([mapping[label] for label in labels], dtype=float)


def _plot_umap_projection(
    projection: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    dims: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  pylint: disable=unused-import
    except ImportError as exc:
        LOGGER.warning("matplotlib not available (%s); skipping UMAP plots.", exc)
        return False

    colors = _assign_colors(labels, plt)
    fig = plt.figure(figsize=(8, 6))
    if dims == 2:
        ax = fig.add_subplot(111)
        ax.scatter(projection[:, 0], projection[:, 1], c=colors, s=5, alpha=0.8)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], c=colors, s=5, alpha=0.8)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")

    ax.set_title(f"UMAP {dims}D Clusters")
    fig.tight_layout()
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        LOGGER.info("Saved UMAP %dD plot to %s", dims, save_path)
    except Exception as exc:
        LOGGER.warning("Failed to save UMAP %dD plot (%s); skipping.", dims, exc)
        plt.close(fig)
        return False
    plt.close(fig)
    return True


def _maybe_plot_umap(embeddings: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    if embeddings.size == 0:
        LOGGER.warning("No embeddings available for UMAP plotting.")
        return

    for dims, filename in ((2, "umap_2d.png"), (3, "umap_3d.png")):
        projection = _compute_umap_projection(embeddings, dims)
        if projection is None:
            continue
        _plot_umap_projection(projection, labels, output_dir / filename, dims)


def sanitize_suffix(raw: Optional[str]) -> str:
    if not raw:
        return ""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", raw.strip()).strip("-")


def resolve_path(raw: Optional[str], date: str, prefix: str, default_dir: Path) -> Path:
    if raw:
        candidate = Path(raw)
        if candidate.is_dir():
            filename = f"{prefix}_{date}.json"
            return candidate / filename
        return candidate
    return default_dir / f"{prefix}_{date}.json"


def load_papers(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of papers.")
    return sorted(data, key=lambda p: p.get("id", ""))


def filter_papers(
    papers: Sequence[dict],
    remove_stopwords: bool,
    min_chars: int = 300,
    min_tokens: int = 50,
) -> Tuple[List[dict], List[str], List[str]]:
    kept: List[dict] = []
    filtered_out: List[str] = []
    texts: List[str] = []
    for paper in tqdm(papers, desc="Filtering papers", unit="paper"):
        abstract = paper.get("abstract", "") or ""
        if len(abstract) < min_chars:
            filtered_out.append(paper.get("id", "unknown"))
            continue
        if not re.search(r"[A-Za-z]", abstract):
            filtered_out.append(paper.get("id", "unknown"))
            continue
        combined = f"{paper.get('title', '')}\n{abstract}"
        processed = preprocess_text(combined, remove_stopwords=remove_stopwords)
        if len(processed.split()) < min_tokens:
            filtered_out.append(paper.get("id", "unknown"))
            continue
        kept.append(paper)
        texts.append(processed)
    return kept, filtered_out, texts


def build_pipeline_config(args: argparse.Namespace) -> Tuple[EmbeddingConfig, SummarizerConfig, Path, Path]:
    suffix = sanitize_suffix(args.suffix)
    input_prefix = f"arxiv_{suffix}" if suffix else "arxiv"
    output_prefix = f"clusters_{suffix}" if suffix else "clusters"
    input_path = resolve_path(args.input, args.date, input_prefix, DEFAULT_INPUT_DIR)
    output_path = resolve_path(args.output, args.date, output_prefix, DEFAULT_OUTPUT_DIR)

    cache_path = Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE
    if cache_path.is_dir():
        cache_path = cache_path / "embeddings.db"
    embedding_config = EmbeddingConfig(
        backend=args.backend,
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
        cache_path=cache_path,
    )
    summarizer_config = SummarizerConfig(model=args.summarizer_model)
    return embedding_config, summarizer_config, input_path, output_path


def select_representatives(
    group: ClusterGroup, probabilities: np.ndarray, papers: Sequence[dict], limit: int = 6
) -> List[dict]:
    if group.indices.size == 0:
        return []
    scores = probabilities[group.indices]
    sorted_idx = group.indices[np.argsort(-scores)]
    selected = sorted_idx[:limit]
    representatives = []
    for idx in selected:
        paper = papers[int(idx)]
        representatives.append(
            {
                "id": paper.get("id"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "primary_category": paper.get("primary_category"),
                "authors": paper.get("authors"),
            }
        )
    return representatives


def build_clusters_payload(
    result: ClusterResult,
    soft_papers: Sequence[dict],
    summarizer_config: SummarizerConfig,
) -> Tuple[List[dict], List[str]]:
    clusters_payload: List[dict] = []
    noise_ids = result.noise_ids
    for idx, group in enumerate(tqdm(result.clusters, desc="Summarizing clusters", unit="cluster")):
        reps = select_representatives(group, result.probabilities, soft_papers)
        try:
            summary = summarize_cluster(reps, summarizer_config)
        except Exception as exc:
            LOGGER.warning("LLM summary failed for cluster %d: %s", idx, exc)
            summary = {"topic": "", "keywords": [], "description": ""}

        cluster_papers = [soft_papers[int(i)] for i in group.indices]

        clusters_payload.append(
            {
                "cluster_id": idx,
                "size": group.size,
                "rescued": group.rescued,
                "topic": summary.get("topic", ""),
                "keywords": summary.get("keywords", []),
                "description": summary.get("description", ""),
                "representatives": reps,
                "papers": cluster_papers,
            }
        )
    return clusters_payload, noise_ids


def save_output(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def resolve_model_name(embedding_config: EmbeddingConfig) -> str:
    if embedding_config.model:
        return embedding_config.model
    if embedding_config.backend == "openai":
        return "text-embedding-3-large"
    return DEFAULT_FALLBACK_MODEL


def build_metadata(
    args: argparse.Namespace,
    filtered_count: int,
    input_count: int,
    cluster_count: int,
    noise_pre_filtered: int,
    noise_hdbscan: int,
    noise_rescued: int,
    embedding_config: EmbeddingConfig,
) -> dict:
    return {
        "date": args.date,
        "model": resolve_model_name(embedding_config),
        "summarizer_model": args.summarizer_model,
        "paper_count_input": input_count,
        "paper_count_filtered": filtered_count,
        "paper_count_clustered": filtered_count,
        "cluster_count": cluster_count,
        "noise_pre_filtered": noise_pre_filtered,
        "noise_hdbscan": noise_hdbscan,
        "noise_rescued": noise_rescued,
        "embedding_backend": embedding_config.backend,
        "cache_path": str(embedding_config.cache_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster arXiv papers with a robust pipeline.")
    parser.add_argument("--date", required=True, help="Date string for input/output files (YYYY-MM-DD).")
    parser.add_argument("--input", help="Input JSON file or directory (default data/arxiv_<date>.json).")
    parser.add_argument("--output", help="Output JSON file or directory (default data/clusters_<date>.json).")
    parser.add_argument("--backend", choices=["openai", "huggingface"], default=DEFAULT_BACKEND)
    parser.add_argument("--model", help="Embedding model (defaults to text-embedding-3-large or huggingface fallback).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding requests.")
    parser.add_argument("--device", help="Device hint for HuggingFace embeddings (cpu/cuda/auto).")
    parser.add_argument("--cache-dir", help="Directory for embedding cache (sqlite).")
    parser.add_argument("--stopword-filter", action="store_true", help="Strip a short list of stopwords during preprocessing.")
    parser.add_argument("--summarizer-model", default=DEFAULT_SUMMARY_MODEL, help="LLM for summarizing topics.")
    parser.add_argument("--suffix", help="Suffix appended to default filenames.")
    parser.add_argument("--plot-umap", dest="plot_umap", action="store_true", default=True, help="Enable saving UMAP scatter plots.")
    parser.add_argument("--no-plot-umap", dest="plot_umap", action="store_false", help="Skip UMAP plotting.")
    return parser


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    configure_logging()

    embedding_config, summarizer_config, input_path, output_path = build_pipeline_config(args)
    papers = load_papers(input_path)
    input_count = len(papers)
    LOGGER.info("Loaded %d papers from %s", input_count, input_path)

    filtered, pre_filtered_ids, preprocessed_texts = filter_papers(papers, args.stopword_filter)
    filtered_count = len(filtered)
    LOGGER.info("Kept %d papers after filtering (removed %d).", filtered_count, len(pre_filtered_ids))

    if filtered_count == 0:
        raise SystemExit("No papers remain after filtering; aborting.")

    embeddings = embed_texts(preprocessed_texts, embedding_config)
    cluster_result = perform_clustering(embeddings, filtered, ClusterConfig())

    if args.plot_umap:
        _maybe_plot_umap(np.asarray(embeddings), cluster_result.labels, output_path.parent)
    else:
        LOGGER.info("UMAP plotting disabled.")

    clusters_payload, noise_ids = build_clusters_payload(cluster_result, filtered, summarizer_config)
    metadata = build_metadata(
        args,
        filtered_count,
        input_count,
        len(clusters_payload),
        len(pre_filtered_ids),
        len(noise_ids),
        cluster_result.rescued_count,
        embedding_config,
    )

    payload = {"metadata": metadata, "clusters": clusters_payload, "noise": noise_ids}
    save_output(payload, output_path)
    LOGGER.info("Saved %d clusters to %s", len(clusters_payload), output_path)
    LOGGER.info("Noise papers (remaining): %d", len(noise_ids))


if __name__ == "__main__":
    main()
