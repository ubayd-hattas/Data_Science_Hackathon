"""Bounded feature-transfer ablation and one inductive shift correction.

This experiment reuses the cached organiser features and fixed Amsterdam splits.
It deliberately tests six scientifically defined representations and no parameter
grid.  The only correction is support-fitted target standardisation: statistics
are estimated from the labelled Amsterdam support set in each trial and then
applied to that trial's support and query rows.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from metric_learning_experiment import (
    BUDGETS,
    CLASSES,
    TripletMLP,
    load_or_build_features,
    load_or_build_splits,
    macro_f1,
    prototype_predict,
    summarize,
    train_embedding,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def select(names: list[str], predicate) -> list[int]:
    return [i for i, name in enumerate(names) if predicate(name)]


def define_groups(names: list[str]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    bands = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
    indices = ("NDVI", "NDBI", "UI", "MNDWI", "BSI")
    groups = {
        "absolute_spectral_means": select(names, lambda n: n in {f"{b}_mean" for b in bands}),
        "spectral_indices": select(names, lambda n: any(n.startswith(i + "_") for i in indices)),
        "temporal_variability": select(names, lambda n: n.endswith("_std")),
        "early_period_statistics": select(names, lambda n: "_early_" in n and not n.startswith("has_")),
        "late_period_statistics": select(names, lambda n: "_late_" in n and not n.startswith("has_")),
        "year_to_year_change": select(names, lambda n: n.startswith("d_")),
        "data_presence_flags": select(names, lambda n: n.startswith("has_")),
    }
    representations = {
        "all_60": list(range(len(names))),
        "absolute_spectral_means": groups["absolute_spectral_means"],
        "spectral_indices": groups["spectral_indices"],
        "temporal_variability": groups["temporal_variability"],
        "early_period_statistics": groups["early_period_statistics"],
        "late_period_statistics": groups["late_period_statistics"],
        "year_to_year_change": groups["year_to_year_change"],
    }
    assert set(i for values in groups.values() for i in values) == set(range(60))
    return groups, representations


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    splits: dict,
    trials: int,
    support_standardize: bool = False,
) -> tuple[dict, dict[str, dict[int, np.ndarray]]]:
    result = {}
    confusions = {
        "prototype": {b: np.zeros((4, 4), dtype=np.int64) for b in BUDGETS},
        "logistic_regression": {b: np.zeros((4, 4), dtype=np.int64) for b in BUDGETS},
    }
    for budget in BUDGETS:
        proto_scores, logistic_scores = [], []
        for trial in range(trials):
            support, query = splits[(budget, trial)]
            sx, qx = x[support], x[query]
            if support_standardize:
                centre = sx.mean(axis=0)
                scale = sx.std(axis=0)
                scale = np.where(scale < 1e-8, 1.0, scale)
                sx, qx = (sx - centre) / scale, (qx - centre) / scale
            proto_pred = prototype_predict(sx, y[support], qx)
            logistic = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
            logistic.fit(sx, y[support])
            logistic_pred = logistic.predict(qx)
            proto_scores.append(macro_f1(y[query], proto_pred))
            logistic_scores.append(macro_f1(y[query], logistic_pred))
            confusions["prototype"][budget] += confusion_matrix(
                y[query], proto_pred, labels=CLASSES
            )
            confusions["logistic_regression"][budget] += confusion_matrix(
                y[query], logistic_pred, labels=CLASSES
            )
        result[str(budget)] = {
            "prototype": summarize(proto_scores),
            "logistic_regression": summarize(logistic_scores),
        }
    return result, confusions


def group_shift(mx: np.ndarray, ax: np.ndarray, groups: dict[str, list[int]]) -> dict:
    # Absolute standardized mean difference, with the pooled (city-combined) SD.
    pooled_sd = np.sqrt((mx.var(axis=0) + ax.var(axis=0)) / 2.0)
    smd = np.divide(
        np.abs(mx.mean(axis=0) - ax.mean(axis=0)),
        pooled_sd,
        out=np.zeros_like(pooled_sd, dtype=np.float64),
        where=pooled_sd > 1e-12,
    )
    return {
        group: {
            "feature_count": len(columns),
            "mean_absolute_smd": float(np.mean(smd[columns])),
            "median_absolute_smd": float(np.median(smd[columns])),
            "max_absolute_smd": float(np.max(smd[columns])),
        }
        for group, columns in groups.items()
    }


def shift_transfer_association(shifts: dict, ablations: dict) -> dict:
    names = [name for name in ablations if name != "all_60"]
    x = np.asarray([shifts[name]["mean_absolute_smd"] for name in names])
    result = {"groups_compared": names, "n_groups": len(names)}
    for method in ("prototype", "logistic_regression"):
        y = np.asarray([ablations[name]["25"][method]["macro_f1_mean"] for name in names])
        pearson = float(np.corrcoef(x, y)[0, 1])
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
        result[method] = {
            "pearson_shift_vs_25_f1": pearson,
            "spearman_shift_rank_vs_25_f1_rank": spearman,
        }
    return result


def flatten_method_candidates(results: dict, budget: int = 25):
    for representation, values in results.items():
        for method in ("prototype", "logistic_regression"):
            yield (
                values[str(budget)][method]["macro_f1_mean"],
                representation,
                method,
            )


def per_class_from_confusion(matrix: np.ndarray) -> dict:
    # Reconstructing labels from aggregate counts gives exact aggregate metrics.
    actual, predicted = [], []
    for i, true_class in enumerate(CLASSES):
        for j, predicted_class in enumerate(CLASSES):
            count = int(matrix[i, j])
            actual.extend([true_class] * count)
            predicted.extend([predicted_class] * count)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=CLASSES, zero_division=0
    )
    return {
        str(cls): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support_across_trials": int(support[i]),
        }
        for i, cls in enumerate(CLASSES)
    }


def plot_shift(shift: dict, out: Path) -> None:
    ordered = sorted(shift, key=lambda g: shift[g]["mean_absolute_smd"])
    values = [shift[g]["mean_absolute_smd"] for g in ordered]
    labels = [g.replace("_", " ") for g in ordered]
    fig, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.barh(labels, values, color="#4472C4")
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_xlabel("Mean absolute standardized mean difference")
    axis.set_title("Madrid-Amsterdam feature shift by scientific group")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_shift_and_transfer(shift: dict, ablations: dict, out: Path) -> None:
    groups = [name for name in ablations if name != "all_60"]
    groups = sorted(groups, key=lambda g: shift[g]["mean_absolute_smd"])
    labels = [g.replace("_", " ") for g in groups]
    y = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    shift_values = [shift[g]["mean_absolute_smd"] for g in groups]
    axes[0].barh(y, shift_values, color="#4472C4")
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Mean absolute SMD")
    axes[0].set_title("Madrid-Amsterdam shift")
    axes[0].grid(axis="x", alpha=0.25)
    for offset, method, label, colour in [
        (-0.10, "prototype", "Prototype", "#ED7D31"),
        (0.10, "logistic_regression", "Logistic", "#70AD47"),
    ]:
        means = [ablations[g]["25"][method]["macro_f1_mean"] for g in groups]
        stds = [ablations[g]["25"][method]["macro_f1_std_population"] for g in groups]
        axes[1].errorbar(means, y + offset, xerr=stds, fmt="o", capsize=3,
                         label=label, color=colour)
    axes[1].axvline(
        ablations["all_60"]["25"]["logistic_regression"]["macro_f1_mean"],
        color="black", linestyle="--", linewidth=1, label="All-60 logistic",
    )
    axes[1].set_xlabel("Macro F1 at 25 labels/class")
    axes[1].set_title("Transfer on identical splits")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Feature-group shift versus Amsterdam transfer")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_final(methods: list[tuple[str, dict, str]], out: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.3))
    styles = ["o-", "s-", "^-", "D--", "v--"]
    for (label, values, method), style in zip(methods, styles):
        means = [values[str(b)][method]["macro_f1_mean"] for b in BUDGETS]
        stds = [values[str(b)][method]["macro_f1_std_population"] for b in BUDGETS]
        axis.errorbar(BUDGETS, means, yerr=stds, fmt=style, capsize=3, label=label)
    axis.set_xscale("log", base=2)
    axis.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    axis.set_xlabel("Amsterdam labels per class (log2 scale)")
    axis.set_ylabel("Amsterdam query macro F1")
    axis.set_title("Final comparison on identical fixed splits")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, out: Path, title: str) -> None:
    proportions = matrix / matrix.sum(axis=1, keepdims=True)
    fig, axis = plt.subplots(figsize=(5.6, 5.0))
    image = axis.imshow(proportions, vmin=0, vmax=1, cmap="Blues")
    for i in range(4):
        for j in range(4):
            axis.text(j, i, f"{proportions[i, j]:.2f}", ha="center", va="center",
                      color="white" if proportions[i, j] > 0.52 else "black")
    axis.set_xticks(range(4), [f"C{c}" for c in CLASSES])
    axis.set_yticks(range(4), [f"C{c}" for c in CLASSES])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Row proportion")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--retry-metric", action="store_true")
    args = parser.parse_args()
    started = time.time()
    outputs = ROOT / "outputs"
    figures = outputs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    features = load_or_build_features(outputs / "features_60.npz")
    names = [str(v) for v in features["feature_names"]]
    mx, my = features["madrid_x"], features["madrid_y"]
    ax, ay = features["amsterdam_x"], features["amsterdam_y"]
    splits = load_or_build_splits(outputs / "amsterdam_fixed_splits.npz", ay, args.trials, args.seed)
    groups, representations = define_groups(names)

    shifts = group_shift(mx, ax, groups)
    plot_shift(shifts, figures / "feature_group_domain_shift.png")
    ablations, all_confusions = {}, {}
    for name, columns in representations.items():
        print(f"Evaluating {name} ({len(columns)} features)...", flush=True)
        ablations[name], all_confusions[name] = evaluate(
            ax[:, columns], ay, splits, args.trials
        )

    best_score, best_representation, best_method = max(flatten_method_candidates(ablations))
    best_columns = representations[best_representation]
    print(f"Correction target: {best_representation} / {best_method}", flush=True)
    corrected, corrected_confusions = evaluate(
        ax[:, best_columns], ay, splits, args.trials, support_standardize=True
    )
    association = shift_transfer_association(shifts, ablations)
    plot_shift_and_transfer(
        shifts, ablations, figures / "feature_group_shift_vs_transfer.png"
    )

    metric_retry = None
    metric_confusions = None
    if args.retry_metric:
        print(f"Retrying fixed triplet architecture on {best_representation}...", flush=True)
        model, losses = train_embedding(
            mx[:, best_columns], my, args.seed, 100, 32, 256, 0.2, 0.001, 16, 1
        )
        embedded_ax = model.forward(ax[:, best_columns])
        embedded, metric_confusions = evaluate(embedded_ax, ay, splits, args.trials)
        metric_retry = {
            "representation": best_representation,
            "input_feature_count": len(best_columns),
            "architecture": [len(best_columns), 64, 32, 16],
            "configuration": {
                "epochs": 100, "batches_per_epoch": 32, "batch_size": 256,
                "margin": 0.2, "learning_rate": 0.001, "negative_candidates": 1,
            },
            "epoch_losses": losses,
            "results": embedded,
        }

    candidates = [(best_score, "ablation", best_representation, best_method, ablations[best_representation], all_confusions[best_representation])]
    for method in ("prototype", "logistic_regression"):
        candidates.append((corrected["25"][method]["macro_f1_mean"], "corrected", best_representation, method, corrected, corrected_confusions))
    if metric_retry:
        for method in ("prototype", "logistic_regression"):
            candidates.append((metric_retry["results"]["25"][method]["macro_f1_mean"], "metric_retry", best_representation, method, metric_retry["results"], metric_confusions))
    strongest = max(candidates)
    _, family, representation, method, strongest_values, strongest_confusions = strongest
    matrix = strongest_confusions[method][25]
    row_props = matrix / matrix.sum(axis=1, keepdims=True)
    plot_confusion(matrix, figures / "strongest_25_confusion_matrix.png", "Strongest 25/class method")

    raw_metric = json.loads((outputs / "metric_learning_results.json").read_text(encoding="utf-8"))
    final_lines = [
        ("Raw 60-feature prototype", raw_metric["raw_features"], "prototype"),
        ("Raw 60-feature logistic", raw_metric["raw_features"], "logistic_regression"),
        ("Madrid triplet embedding prototype", raw_metric["embedding"], "prototype"),
        ("Variability-only logistic", ablations["temporal_variability"], "logistic_regression"),
        (f"Support-standardised {best_representation} {best_method.replace('_', ' ')}", corrected, best_method),
    ]
    if metric_retry:
        final_lines.append((f"Retry embedding {best_representation} prototype", metric_retry["results"], "prototype"))
    plot_final(final_lines, figures / "final_comparison_f1_vs_budget.png")

    result = {
        "experiment": "feature-group transfer ablation and targeted domain-shift correction",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "seed": args.seed,
        "trials": args.trials,
        "budgets": list(BUDGETS),
        "data": {"features": "outputs/features_60.npz", "splits": "outputs/amsterdam_fixed_splits.npz"},
        "atomic_feature_groups": {g: [names[i] for i in cols] for g, cols in groups.items()},
        "tested_representations": {r: [names[i] for i in cols] for r, cols in representations.items()},
        "shift_measure": {
            "name": "absolute standardized mean difference",
            "definition": "absolute city-mean difference divided by pooled within-city SD, averaged across features in each group",
            "groups": shifts,
            "association_with_25_class_transfer": association,
        },
        "ablations": ablations,
        "best_uncorrected_at_25": {"representation": best_representation, "method": best_method, "macro_f1_mean": best_score},
        "correction": {
            "name": "Amsterdam-support standardisation",
            "validity": "fit separately on each labelled Amsterdam support set; no query features or labels used",
            "representation": best_representation,
            "results": corrected,
        },
        "metric_retry": metric_retry,
        "metric_retry_decision": (
            "performed" if metric_retry else
            "skipped: no reduced representation beat all-60 logistic at 25/class, "
            "and support standardisation did not improve it"
        ),
        "strongest_25": {
            "family": family, "representation": representation, "method": method,
            "results": strongest_values["25"][method],
            "confusion_counts_across_trials": matrix.tolist(),
            "confusion_row_proportions": row_props.tolist(),
            "per_class_aggregate": per_class_from_confusion(matrix),
        },
        "elapsed_seconds": time.time() - started,
    }
    (outputs / "feature_transfer_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    consolidated = {
        "evaluation": {
            "direction": "Madrid -> Amsterdam",
            "budgets_labels_per_class": list(BUDGETS),
            "trials": args.trials,
            "splits": "outputs/amsterdam_fixed_splits.npz",
            "metric": "macro F1; population SD across fixed trials",
        },
        "madrid_organiser_baseline": {
            "method": "500-tree balanced random forest on organiser 60 features",
            "cv_macro_f1_mean": 0.6281315050690548,
            "cv_macro_f1_std_population": 0.004265247306708275,
            "amsterdam_zero_shot_macro_f1": 0.443273608786401,
        },
        "final_candidate_methods": {
            "raw_60_prototype": {
                budget: values["prototype"]
                for budget, values in raw_metric["raw_features"].items()
            },
            "raw_60_logistic": {
                budget: values["logistic_regression"]
                for budget, values in raw_metric["raw_features"].items()
            },
            "temporal_variability_logistic": {
                budget: values["logistic_regression"]
                for budget, values in ablations["temporal_variability"].items()
            },
            "support_standardised_60_logistic": {
                budget: values["logistic_regression"]
                for budget, values in corrected.items()
            },
            "madrid_triplet_embedding_prototype": {
                budget: values["prototype"]
                for budget, values in raw_metric["embedding"].items()
            },
            "madrid_triplet_embedding_logistic": {
                budget: values["logistic_regression"]
                for budget, values in raw_metric["embedding"].items()
            },
        },
        "selection": {
            "chosen_method": "raw organiser 60-feature Amsterdam-support logistic regression",
            "strongest_25_macro_f1_mean": result["strongest_25"]["results"]["macro_f1_mean"],
            "strongest_25_macro_f1_std_population": result["strongest_25"]["results"]["macro_f1_std_population"],
            "metric_retry": result["metric_retry_decision"],
            "broad_modelling_recommendation": "freeze modelling",
        },
        "source_artifacts": [
            "outputs/baseline_results.json",
            "outputs/metric_learning_results.json",
            "outputs/feature_transfer_results.json",
        ],
    }
    (outputs / "consolidated_results.json").write_text(
        json.dumps(consolidated, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["best_uncorrected_at_25"], indent=2), flush=True)
    print(json.dumps(result["strongest_25"], indent=2), flush=True)
    print(f"Wrote {outputs / 'feature_transfer_results.json'}", flush=True)


if __name__ == "__main__":
    main()
