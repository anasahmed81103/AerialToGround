"""Inverse-frequency class weights from cached train label memmaps (Phase 2)."""

from __future__ import annotations

import numpy as np

from semantic_metrics import CLASS_NAMES


def pixel_histogram(label_path: str, num_classes: int = 4) -> np.ndarray:
    arr = np.load(label_path, mmap_mode='r')
    return np.bincount(arr.ravel(), minlength=num_classes).astype(np.float64)


def inverse_frequency_weights(label_path: str, num_classes: int = 4) -> dict:
    """weight_c = (1 / freq_c), normalized so mean(weight) == 1."""
    counts = pixel_histogram(label_path, num_classes)
    freq = counts / max(counts.sum(), 1.0)
    w = 1.0 / np.maximum(freq, 1e-8)
    w = w / w.mean()
    return {
        'counts': counts,
        'freq': freq,
        'weights': w,
    }


def parse_class_weights(spec: str, label_path: str, num_classes: int = 4) -> np.ndarray:
    spec = (spec or '').strip()
    if not spec:
        return np.ones(num_classes, dtype=np.float32)
    if spec.lower() == 'auto':
        return inverse_frequency_weights(label_path, num_classes)['weights'].astype(np.float32)
    parts = [float(x.strip()) for x in spec.split(',')]
    if len(parts) != num_classes:
        raise ValueError(f'--class_weights needs {num_classes} values, got {len(parts)}')
    w = np.asarray(parts, dtype=np.float64)
    w = w / w.mean()
    return w.astype(np.float32)


def format_weights(weights: np.ndarray) -> str:
    return '  '.join(f'{n}={weights[i]:.4f}' for i, n in enumerate(CLASS_NAMES))
