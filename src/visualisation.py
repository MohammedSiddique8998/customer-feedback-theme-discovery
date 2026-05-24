from pathlib import Path

import numpy as np


PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centred = features - features.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    coordinates = centred @ vt[:2].T
    explained = np.var(coordinates, axis=0) / np.sum(np.var(centred, axis=0))
    return coordinates, explained


def save_scatter_svg(points: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, padding = 900, 620, 70
    x_values, y_values = points[:, 0], points[:, 1]
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    def scale_x(value):
        return padding + ((value - x_min) / x_span) * (width - 2 * padding)

    def scale_y(value):
        return height - padding - ((value - y_min) / y_span) * (height - 2 * padding)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{padding}" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111827">{title}</text>',
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#111827"/>',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#111827"/>',
        f'<text x="{width / 2 - 35}" y="{height - 20}" font-family="Arial" font-size="14" fill="#374151">PCA 1</text>',
        f'<text x="18" y="{height / 2}" font-family="Arial" font-size="14" fill="#374151" transform="rotate(-90 18,{height / 2})">PCA 2</text>',
    ]

    for point, label in zip(points, labels):
        color = PALETTE[int(label) % len(PALETTE)]
        elements.append(
            f'<circle cx="{scale_x(point[0]):.1f}" cy="{scale_y(point[1]):.1f}" r="6" '
            f'fill="{color}" fill-opacity="0.76" stroke="#ffffff" stroke-width="1"/>'
        )

    legend_x, legend_y = width - 210, 72
    for offset, cluster_id in enumerate(sorted(set(labels.tolist()))):
        color = PALETTE[int(cluster_id) % len(PALETTE)]
        elements.append(f'<circle cx="{legend_x}" cy="{legend_y + offset * 24}" r="6" fill="{color}"/>')
        elements.append(f'<text x="{legend_x + 16}" y="{legend_y + 5 + offset * 24}" font-family="Arial" font-size="13" fill="#111827">Cluster {cluster_id}</text>')

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def save_silhouette_svg(scores: dict[int, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, padding = 820, 500, 70
    k_values = sorted(scores)
    max_score = max(scores.values()) or 1.0
    bar_width = (width - 2 * padding) / len(k_values) * 0.65
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{padding}" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111827">Silhouette Score by K</text>',
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#111827"/>',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#111827"/>',
    ]
    for index, k_value in enumerate(k_values):
        x = padding + index * ((width - 2 * padding) / len(k_values)) + bar_width * 0.25
        bar_height = (scores[k_value] / max_score) * (height - 2 * padding)
        y = height - padding - bar_height
        elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>')
        elements.append(f'<text x="{x + bar_width / 2 - 5:.1f}" y="{height - padding + 20}" font-family="Arial" font-size="12" fill="#374151">{k_value}</text>')
        elements.append(f'<text x="{x - 4:.1f}" y="{y - 8:.1f}" font-family="Arial" font-size="11" fill="#111827">{scores[k_value]:.2f}</text>')
    elements.append(f'<text x="{width / 2 - 8}" y="{height - 22}" font-family="Arial" font-size="14" fill="#374151">K</text>')
    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")
