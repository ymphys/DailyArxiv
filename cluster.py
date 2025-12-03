import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

logger = logging.getLogger("dailyarxiv.cluster")


@dataclass(frozen=True)
class ClusterConfig:
    rescue_label_start: int = 1000


@dataclass
class ClusterGroup:
    label: int
    indices: np.ndarray
    rescued: bool
    size: int


@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray
    umap_embeddings: np.ndarray
    clusters: List[ClusterGroup]
    noise_ids: List[str]
    rescued_count: int
    rescued_labels: Set[int]


def _run_umap(embeddings: np.ndarray) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required for dimensionality reduction.") from exc

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=50,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _run_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> (np.ndarray, np.ndarray):
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("hdbscan is required for clustering.") from exc

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        cluster_selection_epsilon=0.5,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    probabilities = getattr(clusterer, "probabilities_", np.ones(len(embeddings)))
    return labels, probabilities


def _determine_min_cluster_size(n_samples: int) -> int:
    calculated = max(3, int(n_samples * 0.002))
    return max(3, calculated)


def _search_kmeans(noise_embeddings: np.ndarray, seed: int = 42) -> Optional[np.ndarray]:
    n_samples = noise_embeddings.shape[0]
    if n_samples < 4:
        return None

    min_k = 2
    max_k = min(20, n_samples - 1)
    if max_k < min_k + 1:
        return None

    best_labels = None
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        try:
            clusterer = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            labels = clusterer.fit_predict(noise_embeddings)
        except ValueError:
            continue

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(noise_embeddings, labels)
        if score > best_score:
            best_score = score
            best_labels = labels

    return best_labels


def _rescue_noise(
    labels: np.ndarray,
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    label_start: int,
) -> (np.ndarray, Set[int], np.ndarray, int):
    noise_mask = labels == -1
    noise_indices = np.where(noise_mask)[0]
    if noise_indices.size == 0:
        return labels, set(), probabilities, 0

    noise_embeddings = embeddings[noise_mask]
    fallback_labels = _search_kmeans(noise_embeddings)
    if fallback_labels is None:
        return labels, set(), probabilities, 0

    rescued_labels: Set[int] = set()
    updated_labels = labels.copy()

    mapping = {}
    next_label = label_start
    for source_label in sorted(set(fallback_labels)):
        mapping[source_label] = next_label
        rescued_labels.add(next_label)
        next_label += 1

    for noise_idx, fallback_label in zip(noise_indices, fallback_labels):
        updated_labels[noise_idx] = mapping[fallback_label]
        probabilities[noise_idx] = 1.0

    rescued_count = noise_indices.size
    return updated_labels, rescued_labels, probabilities, rescued_count


def _build_groups(labels: np.ndarray, rescued_labels: Set[int]) -> List[ClusterGroup]:
    groups: List[ClusterGroup] = []
    unique_labels = [label for label in sorted(set(labels)) if label != -1]
    for group_label in unique_labels:
        indices = np.where(labels == group_label)[0]
        groups.append(
            ClusterGroup(
                label=group_label,
                indices=indices,
                rescued=group_label in rescued_labels,
                size=len(indices),
            )
        )
    return groups


def perform_clustering(
    embeddings: Sequence[Sequence[float]],
    papers: Sequence[dict],
    config: ClusterConfig,
) -> ClusterResult:
    if not embeddings:
        raise ValueError("No embeddings provided for clustering.")

    array = np.asarray(embeddings, dtype=float)
    min_cluster_size = _determine_min_cluster_size(len(array))
    logger.info("Target min_cluster_size=%d", min_cluster_size)
    umap_embeddings = _run_umap(array)
    labels, probabilities = _run_hdbscan(umap_embeddings, min_cluster_size)

    noise_ratio = float(np.mean(labels == -1))
    logger.info("HDBSCAN noise ratio: %.2f", noise_ratio)

    rescued_labels: Set[int] = set()
    rescued_count = 0
    if noise_ratio > 0.35:
        logger.info("Rescuing noise via secondary clustering...")
        labels, rescued_labels, probabilities, rescued_count = _rescue_noise(
            labels, umap_embeddings, probabilities, config.rescue_label_start
        )

    clusters = _build_groups(labels, rescued_labels)
    all_noise = [papers[idx]["id"] for idx in np.where(labels == -1)[0]]

    return ClusterResult(
        labels=labels,
        probabilities=probabilities,
        umap_embeddings=umap_embeddings,
        clusters=clusters,
        noise_ids=all_noise,
        rescued_count=rescued_count,
        rescued_labels=rescued_labels,
    )
