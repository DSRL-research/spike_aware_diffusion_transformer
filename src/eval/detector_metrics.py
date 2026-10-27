import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ECE_BINS = 15


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=np.float64)))


def ece(probs, labels, bins: int = ECE_BINS) -> float:
    p, y = np.ravel(probs), np.ravel(labels)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def threshold_metrics(probs, labels, threshold: float) -> dict:
    pred = np.ravel(probs) >= threshold
    y = np.ravel(labels).astype(bool)
    tp = (pred & y).sum()
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / y.sum() if y.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=float(precision), recall=float(recall), f1=float(f1))


def best_f1_threshold(probs, labels) -> float:
    precision, recall, thresholds = precision_recall_curve(np.ravel(labels), np.ravel(probs))
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def classification_metrics(logits, labels, threshold: float, prevalence_train: float) -> dict:
    p, y = np.ravel(sigmoid(logits)), np.ravel(labels).astype(np.float64)
    brier = float(np.mean((p - y) ** 2))
    out = dict(spike_rate=float(y.mean()), auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
               brier=brier, brier_skill=float(1 - brier / np.mean((prevalence_train - y) ** 2)), ece=ece(p, y),
               threshold=float(threshold))
    out.update(threshold_metrics(p, y, threshold))
    return out
