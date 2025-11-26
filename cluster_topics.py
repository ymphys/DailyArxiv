import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

RANDOM_SEED = 42
DEFAULT_INPUT_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_BACKEND = "openai"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_HF_MODEL = "BAAI/bge-large-en"


@dataclass(frozen=True)
class ClusterConfig:
    date: str
    input_path: Path
    output_path: Path
    backend: str
    model: str
    batch_size: int
    min_cluster_size: int
    device: Optional[str]
    prefer_hdbscan: bool


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_input_path(raw: Optional[str], date_str: str) -> Path:
    if raw:
        path = Path(raw)
        if path.is_dir():
            return path / f"arxiv_{date_str}.json"
        return path
    return DEFAULT_INPUT_DIR / f"arxiv_{date_str}.json"


def resolve_output_path(raw: Optional[str], date_str: str) -> Path:
    if raw:
        path = Path(raw)
        if path.is_dir():
            return path / f"clusters_{date_str}.json"
        return path
    return DEFAULT_OUTPUT_DIR / f"clusters_{date_str}.json"


def load_papers(input_path: Path) -> List[Dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of papers.")
    return data


def embed_texts_openai(texts: Sequence[str], model: str, batch_size: int) -> List[List[float]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package is required for the OpenAI backend.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; required for OpenAI embedding backend.")

    client = OpenAI(api_key=api_key)
    embeddings: List[List[float]] = []

    for idx in range(0, len(texts), batch_size):
        batch = texts[idx : idx + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings


def resolve_device(requested: Optional[str]) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def embed_texts_huggingface(texts: Sequence[str], model_name: str, device: Optional[str]) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers package is required for the huggingface backend.") from exc

    resolved_device = resolve_device(device)
    model = SentenceTransformer(model_name, device=resolved_device)
    embeddings = model.encode(list(texts), batch_size=16, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def embed_texts(
    texts: Sequence[str],
    backend: str,
    model: str,
    batch_size: int,
    device: Optional[str],
) -> List[List[float]]:
    if backend == "openai":
        return embed_texts_openai(texts, model, batch_size)
    if backend == "huggingface":
        return embed_texts_huggingface(texts, model, device)
    raise ValueError(f"Unknown embedding backend: {backend}")


def hdbscan_cluster(embeddings: np.ndarray, min_cluster_size: int) -> Tuple[np.ndarray, str]:
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("hdbscan package not installed; required for HDBSCAN clustering.") from exc

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    return labels, "hdbscan"


def kmeans_cluster(embeddings: np.ndarray) -> Tuple[np.ndarray, str]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_samples = embeddings.shape[0]
    max_k = min(20, n_samples - 1)
    if max_k < 2:
        labels = np.zeros(n_samples, dtype=int)
        return labels, "kmeans"

    best_score = -1.0
    best_labels = None
    for k in range(2, max_k + 1):
        clusterer = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init="auto")
        labels = clusterer.fit_predict(embeddings)
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_labels = labels

    if best_labels is None:
        best_labels = np.zeros(n_samples, dtype=int)

    return best_labels, "kmeans"


def cluster_embeddings(
    embeddings: Sequence[Sequence[float]],
    min_cluster_size: int,
    prefer_hdbscan: bool = True,
) -> Tuple[np.ndarray, str]:
    array = np.asarray(embeddings, dtype=float)
    if array.size == 0:
        return np.array([], dtype=int), "none"

    if prefer_hdbscan:
        try:
            return hdbscan_cluster(array, min_cluster_size)
        except ImportError:
            pass

    return kmeans_cluster(array)


def build_cluster_payload(
    papers: Sequence[Dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int,
) -> Tuple[Dict[str, Dict], List[str]]:
    clusters: Dict[str, Dict] = {}
    noise_ids: List[str] = []

    for label in set(labels):
        member_indices = np.where(labels == label)[0]
        if label == -1 or len(member_indices) < min_cluster_size:
            noise_ids.extend(papers[idx]["id"] for idx in member_indices)
            continue

        paper_ids = [papers[idx]["id"] for idx in member_indices]
        centroid = embeddings[member_indices].mean(axis=0).tolist()
        cluster_name = f"cluster_{len(clusters) + 1}"
        clusters[cluster_name] = {
            "paper_ids": paper_ids,
            "centroid_embedding": centroid,
            "size": len(paper_ids),
        }

    return clusters, noise_ids


def save_clusters(payload: Dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def build_config(args: argparse.Namespace) -> ClusterConfig:
    input_path = resolve_input_path(args.input_path, args.date)
    output_path = resolve_output_path(args.output_path, args.date)
    model = args.model or (DEFAULT_OPENAI_MODEL if args.backend == "openai" else DEFAULT_HF_MODEL)

    return ClusterConfig(
        date=args.date,
        input_path=input_path,
        output_path=output_path,
        backend=args.backend,
        model=model,
        batch_size=args.batch_size,
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        prefer_hdbscan=not args.force_kmeans,
    )


def process_clusters(config: ClusterConfig) -> Dict:
    papers = load_papers(config.input_path)
    if not papers:
        raise ValueError("No papers found in the input file; cannot cluster.")

    texts = [f"{paper.get('title', '')}\n{paper.get('abstract', '')}".strip() for paper in papers]
    embeddings = embed_texts(texts, config.backend, config.model, config.batch_size, config.device)
    labels, method = cluster_embeddings(embeddings, config.min_cluster_size, config.prefer_hdbscan)

    clusters, noise_ids = build_cluster_payload(papers, np.asarray(embeddings), labels, config.min_cluster_size)

    payload = {
        "metadata": {
            "date": config.date,
            "input": str(config.input_path),
            "backend": config.backend,
            "model": config.model,
            "method": method,
            "min_cluster_size": config.min_cluster_size,
            "paper_count": len(papers),
            "cluster_count": len(clusters),
            "noise_count": len(noise_ids),
        },
        "clusters": clusters,
        "noise": noise_ids,
    }

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate embeddings and topic clusters from arXiv JSON data.")
    parser.add_argument("--date", required=True, help="Date string (YYYY-MM-DD) for the input/output naming.")
    parser.add_argument("--input-path", help="Path to the Phase 1 JSON file or directory (defaults to data/arxiv_<date>.json).")
    parser.add_argument("--output-path", help="Directory or file path for the cluster JSON (defaults to data/clusters_<date>.json).")
    parser.add_argument("--backend", choices=["openai", "huggingface"], default=DEFAULT_BACKEND, help="Embedding backend to use.")
    parser.add_argument("--model", help="Embedding model name; defaults to text-embedding-3-small or BAAI/bge-large-en.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding requests (OpenAI backend).")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Minimum cluster size; smaller clusters become noise.")
    parser.add_argument("--device", help="Device for HuggingFace models (cpu/cuda/auto).")
    parser.add_argument("--force-kmeans", action="store_true", help="Use KMeans instead of HDBSCAN even if available.")
    return parser


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    set_global_seed()
    config = build_config(args)

    print(f"Loading papers from {config.input_path} ...")
    print(f"Embedding backend: {config.backend} ({config.model})")
    print(f"Clustering with min size {config.min_cluster_size} (HDBSCAN preferred: {config.prefer_hdbscan})")

    payload = process_clusters(config)
    output_path = save_clusters(payload, config.output_path)

    print(f"Clusters saved to {output_path}")
    print(f"Clusters found: {payload['metadata']['cluster_count']}, noise papers: {payload['metadata']['noise_count']}")


if __name__ == "__main__":
    main()
