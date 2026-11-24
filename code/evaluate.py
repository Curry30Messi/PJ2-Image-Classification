"""Shared classification metrics with an explicit, fixed class order."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def score_predictions(y_true, y_pred, labels):
    """Return JSON-compatible overall metrics, per-class metrics and counts.

    labels is the complete ordered class list from the chapter 1 labels CSV.
    Confusion matrix rows are true labels and columns are predicted labels.
    A precision, recall or F1 with a zero denominator is assigned zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError("True and predicted labels must be matching, nonempty 1D arrays.")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("The class order must be nonempty and contain unique labels.")
    if not set(y_true).union(y_pred).issubset(labels):
        raise ValueError("Predictions or true labels contain a class outside the fixed class order.")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        dict(label=label, precision=float(p), recall=float(r), f1=float(f), support=int(n))
        for label, p, r, f, n in zip(labels, precision, recall, f1, support, strict=True)
    ]
    return dict(
        samples=int(y_true.size), accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(np.mean(f1)), labels=labels, per_class=per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )
