"""Ordinal-aware classification and label-shift correction.

The four age classes are ordered (1 < 2 < 3 < 4), but a plain classifier treats
them as four unrelated names, so calling a class-1 pixel "class 4" costs the same
as calling it "class 2". Two cheap ways to use the order:

* ``OrdinalRF`` - Frank & Hall (2001) decomposition. Turn the 4-way ordinal
  problem into 3 binary "is the class > k?" Random Forests and recombine. Works
  with any binary learner and needs no soft-label machinery.

* ``reweight_posterior`` / ``estimate_target_prior`` - the class mix differs
  between Madrid and Amsterdam (label shift). If a Madrid-trained model predicts
  probabilities, dividing by the Madrid prior and multiplying by the Amsterdam
  prior re-points it at the target mix. The target prior can be counted from the
  few-shot labels, or estimated from unlabelled Amsterdam with the Saerens et al.
  (2002) EM fixed point.

``mae_in_steps`` reports the average error in class-steps - a second number
alongside macro-F1 that rewards being *close* on the ordinal scale, which the
rubric's "interpretation of the F1 scores" section can use.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

RF_PARAMS = dict(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)


# ── Ordinal classifier ─────────────────────────────────────────────────────

class OrdinalRF:
    """Frank & Hall ordinal Random Forest for contiguous integer classes.

    For classes ``c_1 < ... < c_K`` it fits ``K-1`` binary models, the k-th
    predicting ``P(y > c_k)``. Then
    ``P(y = c_1)   = 1 - P(y > c_1)``
    ``P(y = c_k)   = P(y > c_{k-1}) - P(y > c_k)``
    ``P(y = c_K)   = P(y > c_{K-1})``
    clipped at 0 and renormalised.
    """

    def __init__(self, rf_params: dict | None = None):
        self.rf_params = rf_params or RF_PARAMS

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalRF":
        self.classes_ = np.unique(y)
        self.models_ = []
        for k in self.classes_[:-1]:
            m = RandomForestClassifier(**self.rf_params)
            m.fit(X, (y > k).astype(int))
            self.models_.append(m)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # P(y > c_k) for each split, guarded if a fold saw only one label
        gt = []
        for m in self.models_:
            p = m.predict_proba(X)
            gt.append(p[:, 1] if p.shape[1] == 2 else np.zeros(len(X)))
        gt = np.column_stack(gt)                      # (n, K-1)
        n, K = len(X), len(self.classes_)
        out = np.empty((n, K))
        out[:, 0] = 1.0 - gt[:, 0]
        for j in range(1, K - 1):
            out[:, j] = gt[:, j - 1] - gt[:, j]
        out[:, -1] = gt[:, -1]
        out = np.clip(out, 1e-9, None)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ── Label-shift correction ────────────────────────────────────────────────

def class_prior(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    return counts / counts.sum()


def reweight_posterior(
    proba: np.ndarray, source_prior: np.ndarray, target_prior: np.ndarray
) -> np.ndarray:
    """Re-point source-trained probabilities at the target class mix."""
    w = proba * (target_prior / source_prior)[None, :]
    return w / w.sum(axis=1, keepdims=True)


def estimate_target_prior(
    proba: np.ndarray,
    source_prior: np.ndarray,
    iters: int = 50,
    tol: float = 1e-6,
    floor: float = 0.02,
) -> np.ndarray:
    """Saerens-Latinne-Decaestecker EM estimate of the target prior.

    Uses only the source-trained probabilities on unlabelled target rows - no
    target labels. Iterates a fixed point until the prior stops moving.

    EM diverges when the base classifier is poorly calibrated on the target (a
    big domain gap will do it), collapsing classes to zero. ``floor`` clips each
    class at a minimum share every step to keep the estimate usable; with a
    well-calibrated classifier it never binds.
    """
    prior = source_prior.copy()
    for _ in range(iters):
        w = proba * (prior / source_prior)[None, :]
        post = w / w.sum(axis=1, keepdims=True)
        new = np.clip(post.mean(axis=0), floor, None)
        new = new / new.sum()
        if np.abs(new - prior).max() < tol:
            break
        prior = new
    return prior


# ── Metric ────────────────────────────────────────────────────────────────

def mae_in_steps(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error measured in class-steps (0 = exact, 1 = off by one)."""
    return float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean())
