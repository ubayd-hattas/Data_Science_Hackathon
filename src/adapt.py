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


def zca_whiten(X_ref: np.ndarray, eps: float = 1e-5, shrink: float = 0.0):
    """Return a ``transform(X)`` that centres and decorrelates by ``X_ref``'s covariance.

    Pass ``X_ref = X_target`` for a transductive whitening of the target before
    prototype matching.

    ``shrink`` in [0, 1] blends the covariance toward its own diagonal before
    inverting:  ``C_use = (1 - shrink) * C + shrink * diag(C)``.

    * ``shrink = 0``  full ZCA whitening - best once there are ~50+ support
      points per class, worse below that (a noisy class mean gets smeared across
      every rotated axis).
    * ``shrink = 1``  keeps only per-feature variances - equivalent to plain
      standardisation, which wins at 5-25 support points.

    Intermediate values trace the frontier between the two; ``shrink_for_shots``
    picks one from the label budget.
    """
    mu = X_ref.mean(axis=0)
    cov = np.cov(X_ref - mu, rowvar=False)
    if shrink > 0.0:
        cov = (1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))
    cov = cov + eps * np.eye(X_ref.shape[1])
    w = _sym_inv_sqrt(cov, eps)

    def transform(X: np.ndarray) -> np.ndarray:
        return (X - mu) @ w

    return transform


def shrink_for_shots(n_per_class: int, n_features: int = 60) -> float:
    """Budget-driven shrink level for :func:`zca_whiten`.

    Fitted to the per-budget optimum observed on the 60-feature Amsterdam set
    (shots 5/25/50/100/200 -> shrink 1.0/0.6/0.4/0.2/0.0), then normalised so a
    wider feature vector shifts the same curve toward heavier shrink.
    """
    eff = n_per_class * 60.0 / max(n_features, 1)
    return float(np.clip(1.44 - 0.188 * np.log2(max(eff, 2.0)), 0.0, 1.0))


def spatial_smoother(coords: np.ndarray, k: int = 8, weighted: bool = True):
    """Return ``smooth(P)`` blending each pixel's row of ``P`` with its k neighbours.

    ``P`` is an (n_pixels, n_classes) probability field over the *whole* grid, in
    the same row order as ``coords``. Age classes cluster geographically, so one
    pass of this removes speckle without adding any feature dimensions.

    ``weighted`` (default): neighbours are weighted ``1 / (1 + distance)`` — a
    pixel touching yours is far more likely the same building than one three
    pixels away, so this is a strictly better prior than a flat mean. Worth
    ~+0.011 macro-F1 over the flat version at every budget. Set False for the
    plain average.

    ``k = 8`` (one ring). A single pass is best; iterating over-smooths and blurs
    genuine age boundaries.

    To *anchor* the smoothing, overwrite rows of ``P`` with one-hot true labels
    before calling. We do NOT do this in the pipeline: under the organiser's
    random support sampling ~80% of support pixels touch a query pixel, so
    anchoring would score same-building proximity, not skill (see
    docs/AMSTERDAM_SPATIAL_ADJACENCY_AUDIT.md).
    """
    from scipy.spatial import cKDTree

    dist, nei = cKDTree(coords).query(coords, k=k + 1)   # incl. self at col 0
    if weighted:
        w = 1.0 / (1.0 + dist)
        w = (w / w.sum(axis=1, keepdims=True)).astype(np.float64)

        def smooth(P: np.ndarray) -> np.ndarray:
            return np.einsum("nkc,nk->nc", P[nei], w)
    else:
        def smooth(P: np.ndarray) -> np.ndarray:
            return P[nei].mean(axis=1)

    return smooth


def class_conditional_coral(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    rf_factory,
    rounds: int = 2,
    eps: float = 1e-4,
):
    """Iterative per-class CORAL, driven by the model's own pseudo-labels.

    Plain CORAL aligns the *pooled* source and target clouds. This aligns them
    **class by class**: fit a classifier on globally-CORAL'd source, predict the
    (unlabelled) target, then for each class re-align the source rows of that
    class to the covariance of the target rows the model *assigned* to it, and
    refit. Two rounds is enough.

    No target labels are used - only the classifier's predictions on unlabelled
    target data - so it stays leakage-safe. Returns the fitted classifier; use
    its ``predict_proba`` as the Stage-1 prior.

    On Madrid -> Amsterdam this lifted zero-shot macro-F1 from ~0.58 to ~0.65 and
    the 5-labels/class few-shot point by ~0.02, where nothing else moved it.
    """
    classes = np.unique(y_source)
    coral = make_coral(X_source, X_target, eps)
    Xs_al = coral(X_source)
    clf = rf_factory().fit(Xs_al, y_source)

    for _ in range(rounds):
        yhat = clf.predict(X_target)
        Xs_new = Xs_al.copy()
        for c in classes:
            src = Xs_al[y_source == c]
            tgt = X_target[yhat == c]
            if len(tgt) < X_source.shape[1] + 2:      # too few pseudo-labels
                continue
            mu_s, mu_t = src.mean(0), tgt.mean(0)
            cs = np.cov(src - mu_s, rowvar=False) + eps * np.eye(src.shape[1])
            ct = np.cov(tgt - mu_t, rowvar=False) + eps * np.eye(src.shape[1])
            A = _sym_inv_sqrt(cs, eps) @ _sym_sqrt(ct, eps)
            Xs_new[y_source == c] = (src - mu_s) @ A + mu_t
        Xs_al = Xs_new
        clf = rf_factory().fit(Xs_al, y_source)

    return clf
