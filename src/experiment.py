import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from clustering import run_kmeans, silhouette_score, top_terms_by_cluster
from preprocessing import describe_records, load_dataset
from tfidf import build_vocabulary, tokenize_documents, transform_tfidf
from visualisation import pca_2d, save_scatter_svg, save_silhouette_svg


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: Path = Path("data/sample_sentences.tsv")
    output_dir: Path = Path("results")
    min_k: int = 2
    max_k: int = 8
    min_df: int = 1
    max_features: int = 500
    seed: int = 42


def select_best_k(features: np.ndarray, min_k: int, max_k: int, seed: int) -> tuple[int, dict[int, float], dict[int, object]]:
    scores = {}
    models = {}
    upper_k = min(max_k, features.shape[0] - 1)
    for k_value in range(min_k, upper_k + 1):
        result = run_kmeans(features, n_clusters=k_value, seed=seed)
        score = silhouette_score(features, result.labels)
        scores[k_value] = round(score, 4)
        models[k_value] = result
    best_k = max(scores, key=scores.get)
    return best_k, scores, models


def write_cluster_labels(records, labels: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "cluster", "sentence"])
        for record, label in zip(records, labels):
            writer.writerow([record.record_id, int(label), record.sentence])


def write_cluster_terms(cluster_terms: dict[int, list[str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster", "top_terms"])
        for cluster_id, terms in cluster_terms.items():
            writer.writerow([cluster_id, ", ".join(terms)])


def run_experiment(config: ExperimentConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_dataset(config.data_path)
    cleaned_documents = [record.cleaned_sentence for record in records]
    tokenised = tokenize_documents(cleaned_documents)
    vocabulary = build_vocabulary(tokenised, min_df=config.min_df, max_features=config.max_features)
    features = transform_tfidf(cleaned_documents, vocabulary)

    best_k, scores, models = select_best_k(features, config.min_k, config.max_k, config.seed)
    best_model = models[best_k]
    coordinates, explained_variance = pca_2d(features)
    cluster_terms = top_terms_by_cluster(features, best_model.labels, vocabulary)

    write_cluster_labels(records, best_model.labels, config.output_dir / "cluster_labels.csv")
    write_cluster_terms(cluster_terms, config.output_dir / "cluster_terms.csv")
    save_silhouette_svg(scores, config.output_dir / "silhouette_scores.svg")
    save_scatter_svg(coordinates, best_model.labels, config.output_dir / "pca_cluster_map.svg", "TF-IDF K-Means Clusters")

    metrics = {
        "dataset": describe_records(records),
        "vectorisation": {
            "method": "TF-IDF",
            "vocabulary_size": len(vocabulary),
            "matrix_shape": list(features.shape),
        },
        "clustering": {
            "algorithm": "K-Means",
            "k_values_tested": list(scores.keys()),
            "silhouette_scores": scores,
            "selected_k": best_k,
            "selected_k_silhouette": scores[best_k],
            "inertia": round(best_model.inertia, 4),
            "cluster_sizes": {
                str(cluster_id): int(np.sum(best_model.labels == cluster_id))
                for cluster_id in sorted(set(best_model.labels.tolist()))
            },
            "top_terms": cluster_terms,
        },
        "visualisation": {
            "method": "PCA",
            "explained_variance_ratio_first_two_components": [round(float(value), 4) for value in explained_variance],
        },
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (config.output_dir / "experiment_config.json").write_text(
        json.dumps({**asdict(config), "data_path": str(config.data_path), "output_dir": str(config.output_dir)}, indent=2),
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF K-Means text clustering experiment.")
    parser.add_argument("--data-path", type=Path, default=Path("data/sample_sentences.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(
        ExperimentConfig(
            data_path=args.data_path,
            output_dir=args.output_dir,
            min_k=args.min_k,
            max_k=args.max_k,
            min_df=args.min_df,
            max_features=args.max_features,
            seed=args.seed,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
