"""Unsupervised feature-distribution alignment for Madrid -> Amsterdam.

None of these use target labels, so they are leakage-safe: they only look at the
shape of the unlabelled Amsterdam feature cloud, which the challenge hands you in
full.

Three levels of alignment, cheapest first:

* ``rescale_to`` - match the per-feature mean and variance of the target
  (first + diagonal-second moment). This is what "refit the StandardScaler on
  Amsterdam" does; in the quick baseline it was worth ~+4.5 macro-F1.
* ``coral_transform`` - match the full covariance (CORAL, Sun et al. 2016):
  whiten the source with its own covariance, then recolour with the target's.
  Used on the Madrid features before training the zero-shot classifier.
* ``zca_whiten`` - decorrelate a matrix by its own covariance. Useful before a
  Euclidean prototype classifier so distance is not dominated by the large block
  of correlated band-mean features.
"""

from __future__ import annotations

import numpy as np


def _sym_inv_sqrt(cov: np.ndarray, eps: float) -> np.ndarray:
    """Symmetric inverse square root of a covariance via eigendecomposition."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    return (vecs / np.sqrt(vals)) @ vecs.T


def _sym_sqrt(cov: np.ndarray, eps: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def rescale_to(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Shift/scale each column of ``X`` to ``ref``'s per-feature mean and std."""
    mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-8
    return ((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)) * sd + mu


def coral_transform(
    X_source: np.ndarray, X_target: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Align source second-order statistics to the target (CORAL).

    Returns a re-coloured copy of ``X_source`` whose covariance matches
    ``X_target``'s. Train the Stage-1 classifier on this, then apply it to raw
    target features.
    """
    Xs = X_source - X_source.mean(axis=0)
    cov_s = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    cov_t = np.cov(X_target - X_target.mean(axis=0), rowvar=False) \
        + eps * np.eye(Xs.shape[1])
    a = _sym_inv_sqrt(cov_s, eps) @ _sym_sqrt(cov_t, eps)
    return Xs @ a + X_target.mean(axis=0)


def make_coral(X_source: np.ndarray, X_target: np.ndarray, eps: float = 1e-5):
    """Return a fitted ``transform(X)`` mapping source-domain rows to target space.

    The affine map is learned once from the two unlabelled clouds and can then be
    applied to any source-domain matrix (e.g. CV folds).
    """
    mu_s = X_source.mean(axis=0)
    mu_t = X_target.mean(axis=0)
    cov_s = np.cov(X_source - mu_s, rowvar=False) + eps * np.eye(X_source.shape[1])
    cov_t = np.cov(X_target - mu_t, rowvar=False) + eps * np.eye(X_source.shape[1])
    a = _sym_inv_sqrt(cov_s, eps) @ _sym_sqrt(cov_t, eps)

    def transform(X: np.ndarray) -> np.ndarray:
        return (X - mu_s) @ a + mu_t

    return transform


def zca_whiten(X_ref: np.ndarray, eps: float = 1e-5):
    """Return a ``transform(X)`` that centres and decorrelates by ``X_ref``'s covariance.

    Pass ``X_ref = X_target`` for a transductive whitening of the target before
    prototype matching.
    """
    mu = X_ref.mean(axis=0)
    cov = np.cov(X_ref - mu, rowvar=False) + eps * np.eye(X_ref.shape[1])
    w = _sym_inv_sqrt(cov, eps)

    def transform(X: np.ndarray) -> np.ndarray:
        return (X - mu) @ w

    return transform
