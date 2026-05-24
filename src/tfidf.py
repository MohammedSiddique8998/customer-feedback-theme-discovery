import math
from collections import Counter, defaultdict

import numpy as np


def tokenize_documents(documents: list[str]) -> list[list[str]]:
    return [document.split() for document in documents]


def build_vocabulary(tokenised_documents: list[list[str]], min_df: int = 1, max_features: int = 500) -> list[str]:
    document_frequency = Counter()
    term_frequency = Counter()
    for tokens in tokenised_documents:
        term_frequency.update(tokens)
        document_frequency.update(set(tokens))

    candidates = [
        token for token, df in document_frequency.items()
        if df >= min_df
    ]
    candidates.sort(key=lambda token: (-term_frequency[token], token))
    return candidates[:max_features]


def transform_tfidf(documents: list[str], vocabulary: list[str]) -> np.ndarray:
    tokenised_documents = tokenize_documents(documents)
    index = {token: position for position, token in enumerate(vocabulary)}
    document_frequency = defaultdict(int)
    for tokens in tokenised_documents:
        for token in set(tokens):
            if token in index:
                document_frequency[token] += 1

    n_documents = len(documents)
    matrix = np.zeros((n_documents, len(vocabulary)), dtype=float)
    for row, tokens in enumerate(tokenised_documents):
        counts = Counter(token for token in tokens if token in index)
        if not counts:
            continue
        total_terms = sum(counts.values())
        for token, count in counts.items():
            tf = count / total_terms
            idf = math.log((1 + n_documents) / (1 + document_frequency[token])) + 1
            matrix[row, index[token]] = tf * idf

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
