# Experiment Card

## Project

Customer Feedback Theme Discovery

## Intended Use

Portfolio demonstration of unsupervised text clustering, feature representation, cluster evaluation and visual interpretation.

## Dataset

The original internal prototype used a private TSV sentence dataset that is not included in this public repository. The committed outputs are generated from a small synthetic demo corpus in `data/sample_sentences.tsv` so the workflow can be inspected and rerun safely.

Users may replace the demo data with a permitted TSV file containing:

- `ID`
- `Sentence`

## Methods

- Text cleaning: lowercasing, alphabetic-token filtering, stopword removal and short-token filtering.
- Feature extraction: TF-IDF baseline.
- Clustering: K-Means over a tested K range.
- Model selection: best silhouette score.
- Visualisation: PCA projection to two dimensions.
- Interpretation: top TF-IDF terms per cluster.

## Metrics

- Silhouette score for each tested K.
- Selected K.
- Cluster sizes.
- K-Means inertia.
- PCA explained variance for the first two components.

## Limitations

- Demo corpus is synthetic and intentionally small.
- TF-IDF captures lexical similarity rather than deep semantic similarity.
- PCA visualisation is a simplification of high-dimensional distances.
- Cluster labels need human interpretation.
- Results will change with a different dataset.

## Future Improvements

- Add a sentence-transformer embedding pipeline for permitted environments.
- Add t-SNE or UMAP visualisation.
- Add topic labels generated from top terms.
- Add interactive dashboard for cluster inspection.
- Add stability testing across random seeds.
