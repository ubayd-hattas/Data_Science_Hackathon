"""Evaluation harness: Madrid CV, zero-shot transfer, few-shot prototype curve.

Mirrors the protocol in Notebook 4 but as importable, seeded functions so a new
idea (a different feature set, an embedding, a domain-adaptation step) is a
one-argument swap with a number attached.

The few-shot functions take an optional ``transform`` callable. Pass
``TripletEmbedding(...).fit(X_madrid, y_madrid).transform`` to run prototypes in
a Madrid-trained embedding space; pass ``None`` to reproduce Notebook 4's
raw-feature prototypes. Comparing the two answers the question the notebook
leaves open: does anything trained on Madrid actually help Amsterdam?

Leakage discipline (the rubric's disqualification risk):
* the support/query split is drawn from Amsterdam labels only;
* any ``transform`` must have been fitted on Madrid, never on Amsterdam;
* Madrid CV folds split on pixels, and this dataset has one row per pixel, so
  no pixel spans folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

Transform = Callable[[np.ndarray], np.ndarray]

DEFAULT_SHOTS = (5, 25, 50, 100, 200)
RF_PARAMS = dict(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)


# ── Madrid cross-validation ──────────────────────────────────────────────────

@dataclass
class CVResult:
    fold_f1: np.ndarray
    mean: float
    std: float

    def __str__(self) -> str:
        return f"Madrid CV macro-F1: {self.mean:.4f} +/- {self.std:.4f}  (n={len(self.fold_f1)})"


def madrid_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_repeats: int = 5,
    rf_params: dict | None = None,
    seed: int = 42,
) -> CVResult:
    """Repeated stratified k-fold macro-F1 for a Random Forest on Madrid."""
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    scores = []
    for tr, va in rskf.split(X, y):
        clf = RandomForestClassifier(**(rf_params or RF_PARAMS))
        clf.fit(X[tr], y[tr])
        scores.append(f1_score(y[va], clf.predict(X[va]), average="macro"))
    scores = np.asarray(scores)
    return CVResult(scores, float(scores.mean()), float(scores.std()))


def fit_final_rf(X: np.ndarray, y: np.ndarray, rf_params: dict | None = None):
    """Random Forest trained on all of Madrid — the model used for zero-shot."""
    clf = RandomForestClassifier(**(rf_params or RF_PARAMS))
    return clf.fit(X, y)


def zero_shot(clf, X_target: np.ndarray, y_target: np.ndarray) -> float:
    """Macro-F1 of a Madrid-trained classifier applied straight to the target."""
    return float(f1_score(y_target, clf.predict(X_target), average="macro"))


# ── Few-shot prototype transfer ─────────────────────────────────────────────

def prototype_predict(
    X_support: np.ndarray, y_support: np.ndarray, X_query: np.ndarray
) -> np.ndarray:
    """Nearest-prototype classifier: assign each query to the closest class mean."""
    classes = np.unique(y_support)
    protos = np.stack([X_support[y_support == c].mean(axis=0) for c in classes])
    d = np.linalg.norm(X_query[:, None, :] - protos[None, :, :], axis=2)
    return classes[d.argmin(axis=1)]


@dataclass
class FewShotResult:
    shots: tuple[int, ...]
    per_shot: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def means(self) -> np.ndarray:
        return np.array([self.per_shot[s].mean() for s in self.shots])

    @property
    def stds(self) -> np.ndarray:
        return np.array([self.per_shot[s].std() for s in self.shots])

    def table(self) -> str:
        rows = ["  shots/class    macro-F1"]
        for s in self.shots:
            v = self.per_shot[s]
            rows.append(f"  {s:>10d}    {v.mean():.4f} +/- {v.std():.4f}")
        return "\n".join(rows)


def few_shot_curve(
    X_target: np.ndarray,
    y_target: np.ndarray,
    shots=DEFAULT_SHOTS,
    n_trials: int = 10,
    transform: Transform | None = None,
    seed: int = 42,
) -> FewShotResult:
    """Prototype-transfer macro-F1 vs. label budget, averaged over random draws.

    For each budget ``n``: sample ``n`` support pixels per class from the target,
    predict the rest by nearest prototype, score macro-F1. Repeat ``n_trials``
    times with fresh draws.

    If ``transform`` is given it is applied once to the whole target matrix
    before splitting — it must already be fitted (on Madrid), so this stays
    leakage-free.
    """
    Z = transform(X_target) if transform is not None else X_target
    y = np.asarray(y_target)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    out = FewShotResult(tuple(shots))

    for n in shots:
        trial_scores = []
        for _ in range(n_trials):
            support = []
            for c in classes:
                idx = np.where(y == c)[0]
                support.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
            support = np.asarray(support)
            query = np.ones(len(y), dtype=bool)
            query[support] = False
            pred = prototype_predict(Z[support], y[support], Z[query])
            trial_scores.append(
                f1_score(y[query], pred, average="macro", zero_division=0)
            )
        out.per_shot[n] = np.asarray(trial_scores)

    return out


# ── Reporting ──────────────────────────────────────────────────────────────

def summary_table(cv: CVResult, f1_zero: float, fs: FewShotResult) -> str:
    lines = [
        "| Setting | macro-F1 |",
        "|---|---|",
        f"| Madrid CV (5x5) | {cv.mean:.3f} +/- {cv.std:.3f} |",
        f"| Zero-shot -> Amsterdam | {f1_zero:.3f} |",
    ]
    for s in fs.shots:
        v = fs.per_shot[s]
        lines.append(f"| Few-shot, {s}/class | {v.mean():.3f} +/- {v.std():.3f} |")
    return "\n".join(lines)


def plot_curve(fs: FewShotResult, f1_zero: float, cv: CVResult, path: str) -> None:
    """Save the F1-vs-log2(label budget) plot the submission requires."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    shots = np.array(fs.shots)
    ax.errorbar(shots, fs.means, yerr=fs.stds, fmt="o-", capsize=4,
                color="#fd8d3c", lw=2, label="Few-shot prototype transfer")
    for s in fs.shots:
        ax.scatter([s] * len(fs.per_shot[s]), fs.per_shot[s],
                   color="#fd8d3c", alpha=0.3, s=18, zorder=3)
    ax.axhline(f1_zero, ls="--", color="steelblue", label=f"Zero-shot ({f1_zero:.3f})")
    ax.axhline(cv.mean, ls="--", color="green",
               label=f"Madrid CV ({cv.mean:.3f})")
    ax.fill_between(shots, cv.mean - cv.std, cv.mean + cv.std,
                    color="green", alpha=0.1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(shots)
    ax.set_xticklabels(shots)
    ax.set_xlabel("Labelled Amsterdam pixels per class")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Few-shot transfer: Amsterdam F1 vs. label budget")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Few-shot with a trained classifier head ────────────────────────────────

def few_shot_classifier_curve(
    X_target: np.ndarray,
    y_target: np.ndarray,
    make_clf,
    shots=DEFAULT_SHOTS,
    n_trials: int = 10,
    transform: "Transform | None" = None,
    seed: int = 42,
    extra_metric=None,
) -> "FewShotResult":
    """Like ``few_shot_curve`` but fits ``make_clf()`` on the support set.

    ``make_clf`` is a zero-arg factory returning an unfitted estimator with
    ``fit`` / ``predict``. Use it to compare a small Random Forest or an
    ``OrdinalRF`` head against the nearest-prototype rule at each label budget.
    If ``extra_metric(y_true, y_pred)`` is given, its mean over trials is stored
    on ``result.per_shot_extra``.
    """
    Z = transform(X_target) if transform is not None else X_target
    y = np.asarray(y_target)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    out = FewShotResult(tuple(shots))
    out.per_shot_extra = {}

    for n in shots:
        scores, extras = [], []
        for _ in range(n_trials):
            support = []
            for c in classes:
                idx = np.where(y == c)[0]
                support.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
            support = np.asarray(support)
            query = np.ones(len(y), dtype=bool)
            query[support] = False
            clf = make_clf()
            clf.fit(Z[support], y[support])
            pred = clf.predict(Z[query])
            scores.append(f1_score(y[query], pred, average="macro", zero_division=0))
            if extra_metric is not None:
                extras.append(extra_metric(y[query], pred))
        out.per_shot[n] = np.asarray(scores)
        if extra_metric is not None:
            out.per_shot_extra[n] = np.asarray(extras)

    return out


# ── Prototype classifier object + self-training ────────────────────────────

class PrototypeClassifier:
    """Nearest-class-mean with soft scores (softmax of negative sq. distance)."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.proto_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def _neg_d2(self, X):
        return -((X[:, None, :] - self.proto_[None, :, :]) ** 2).sum(axis=2)

    def predict(self, X):
        return self.classes_[self._neg_d2(X).argmax(axis=1)]

    def predict_proba(self, X):
        z = self._neg_d2(X) / self.temperature
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def few_shot_selftrain_curve(
    X_target: np.ndarray,
    y_target: np.ndarray,
    make_clf,
    shots=DEFAULT_SHOTS,
    n_trials: int = 10,
    transform: "Transform | None" = None,
    seed: int = 42,
    rounds: int = 3,
    add_frac: float = 0.15,
) -> "FewShotResult":
    """Few-shot curve with iterative self-training on the unlabelled query pool.

    Each round: fit ``make_clf()`` on the labelled + pseudo-labelled set, score
    every remaining pool point's confidence, promote the most confident ones
    (balanced across predicted classes) to pseudo-labels, repeat. Final score is
    macro-F1 over the original query set. No true target label beyond the support
    set is ever used.
    """
    Z = transform(X_target) if transform is not None else X_target
    y = np.asarray(y_target)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    out = FewShotResult(tuple(shots))

    for n in shots:
        scores = []
        for _ in range(n_trials):
            support = []
            for c in classes:
                idx = np.where(y == c)[0]
                support.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
            support = np.asarray(support)
            query = np.where(~np.isin(np.arange(len(y)), support))[0]

            lab_idx = list(support)
            lab_y = list(y[support])
            pool = list(query)
            per_round = max(1, int(add_frac * len(query) / max(rounds, 1)))

            for _r in range(rounds):
                if not pool:
                    break
                clf = make_clf().fit(Z[np.array(lab_idx)], np.array(lab_y))
                proba = clf.predict_proba(Z[np.array(pool)])
                pred = classes[proba.argmax(axis=1)]
                conf = proba.max(axis=1)
                take = []
                for c in classes:                       # balanced promotion
                    ci = np.where(pred == c)[0]
                    if len(ci) == 0:
                        continue
                    ci = ci[np.argsort(conf[ci])[::-1][: per_round // len(classes) + 1]]
                    take.extend(ci.tolist())
                take = sorted(set(take))
                for t in take:
                    lab_idx.append(pool[t])
                    lab_y.append(int(pred[t]))
                pool = [p for i, p in enumerate(pool) if i not in set(take)]

            clf = make_clf().fit(Z[np.array(lab_idx)], np.array(lab_y))
            pred_q = clf.predict(Z[query])
            scores.append(f1_score(y[query], pred_q, average="macro", zero_division=0))
        out.per_shot[n] = np.asarray(scores)

    return out


def few_shot_ensemble_curve(
    X_target: np.ndarray,
    y_target: np.ndarray,
    prior_proba: np.ndarray,
    make_clf,
    shots=DEFAULT_SHOTS,
    n_trials: int = 10,
    transform: "Transform | None" = None,
    seed: int = 42,
    beta_div: float = 50.0,
    beta_floor: float = 0.4,
) -> "FewShotResult":
    """Blend a few-shot head's probabilities with a fixed source-model prior.

    ``prior_proba`` is an (n_target, n_classes) array of class probabilities from
    a Madrid-trained model (e.g. a CORAL-aligned Random Forest) evaluated on the
    whole target once. Each trial fits ``make_clf()`` on the whitened support set
    and combines: ``beta * head + (1 - beta) * prior``, with
    ``beta = clip(shots / beta_div, beta_floor, 0.95)`` so the local head takes
    over as labels accumulate. Leakage-safe: the prior uses no target labels, the
    head only the support set.
    """
    Z = transform(X_target) if transform is not None else X_target
    y = np.asarray(y_target)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    out = FewShotResult(tuple(shots))

    for n in shots:
        beta = float(np.clip(n / beta_div, beta_floor, 0.95))
        scores = []
        for _ in range(n_trials):
            support = []
            for c in classes:
                idx = np.where(y == c)[0]
                support.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
            support = np.asarray(support)
            query = np.where(~np.isin(np.arange(len(y)), support))[0]
            clf = make_clf().fit(Z[support], y[support])
            blended = beta * clf.predict_proba(Z[query]) + (1 - beta) * prior_proba[query]
            pred = classes[blended.argmax(axis=1)]
            scores.append(f1_score(y[query], pred, average="macro", zero_division=0))
        out.per_shot[n] = np.asarray(scores)

    return out


def few_shot_smoothed_curve(
    X_target: np.ndarray,
    y_target: np.ndarray,
    prior_proba: np.ndarray,
    make_clf,
    smoother,
    shots=DEFAULT_SHOTS,
    n_trials: int = 10,
    transform: "Transform | None" = None,
    seed: int = 42,
    beta_div: float = 50.0,
    beta_floor: float = 0.4,
) -> "FewShotResult":
    """``few_shot_ensemble_curve`` plus anchored spatial smoothing of the output.

    After blending the local head with the source prior, the class probabilities
    are written into a full-grid field, support pixels are overwritten with their
    true one-hot labels, and ``smoother`` (from
    :func:`src.adapt.spatial_smoother`) averages each pixel with its geographic
    neighbours before the winner is picked.

    Still leakage-free: the only target labels used are the support set's, which
    the protocol grants, and they are applied to their own rows.
    """
    Z = transform(X_target) if transform is not None else X_target
    y = np.asarray(y_target)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    out = FewShotResult(tuple(shots))

    for n in shots:
        beta = float(np.clip(n / beta_div, beta_floor, 0.95))
        scores = []
        for _ in range(n_trials):
            support = []
            for c in classes:
                idx = np.where(y == c)[0]
                support.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
            support = np.asarray(support)
            query = np.where(~np.isin(np.arange(len(y)), support))[0]

            clf = make_clf().fit(Z[support], y[support])
            field = np.empty((len(y), len(classes)))
            field[query] = (beta * clf.predict_proba(Z[query])
                            + (1 - beta) * prior_proba[query])
            onehot = np.zeros((len(support), len(classes)))
            onehot[np.arange(len(support)), np.searchsorted(classes, y[support])] = 1.0
            field[support] = onehot

            pred = classes[smoother(field)[query].argmax(axis=1)]
            scores.append(f1_score(y[query], pred, average="macro", zero_division=0))
        out.per_shot[n] = np.asarray(scores)

    return out
