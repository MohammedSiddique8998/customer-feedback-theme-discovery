from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float


def run_kmeans(features: np.ndarray, n_clusters: int, seed: int = 42, max_iter: int = 200, n_init: int = 20) -> KMeansResult:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    rng = np.random.default_rng(seed)
    best_result = None

    for _ in range(n_init):
        indices = rng.choice(features.shape[0], size=n_clusters, replace=False)
        centroids = features[indices].copy()
        labels = np.zeros(features.shape[0], dtype=int)

        for _ in range(max_iter):
            distances = squared_distances(features, centroids)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster_id in range(n_clusters):
                members = features[labels == cluster_id]
                if len(members) > 0:
                    centroids[cluster_id] = members.mean(axis=0)

        inertia = float(np.sum(np.min(squared_distances(features, centroids), axis=1)))
        result = KMeansResult(labels=labels.copy(), centroids=centroids.copy(), inertia=inertia)
        if best_result is None or result.inertia < best_result.inertia:
            best_result = result

    return best_result


def squared_distances(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)


def cosine_distance_matrix(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalised = features / norms
    return 1 - np.clip(normalised @ normalised.T, -1.0, 1.0)


def silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return 0.0

    distances = cosine_distance_matrix(features)
    scores = []
    for index, label in enumerate(labels):
        same_cluster = labels == label
        same_cluster[index] = False
        if np.any(same_cluster):
            a_value = float(np.mean(distances[index, same_cluster]))
        else:
            a_value = 0.0

        b_value = min(
            float(np.mean(distances[index, labels == other_label]))
            for other_label in unique_labels
            if other_label != label
        )
        denominator = max(a_value, b_value)
        scores.append(0.0 if denominator == 0 else (b_value - a_value) / denominator)
    return float(np.mean(scores))


def top_terms_by_cluster(matrix: np.ndarray, labels: np.ndarray, vocabulary: list[str], top_n: int = 8) -> dict[int, list[str]]:
    output = {}
    for cluster_id in sorted(set(labels.tolist())):
        members = matrix[labels == cluster_id]
        centroid = members.mean(axis=0)
        top_indices = np.argsort(centroid)[::-1][:top_n]
        output[int(cluster_id)] = [vocabulary[index] for index in top_indices if centroid[index] > 0]
    return output
