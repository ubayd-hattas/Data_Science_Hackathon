"""Minimal, reproducible Madrid -> Amsterdam metric-learning experiment.

Reuses the organiser's exact 60-feature pipeline, creates deterministic shared
Amsterdam support/query splits, and compares raw versus learned embeddings.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from organiser_baseline import feature_columns, preprocess_city, prototype_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (5, 25, 50, 100, 200)
CLASSES = np.array([1, 2, 3, 4])


def load_or_build_features(cache: Path) -> dict[str, np.ndarray]:
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    print("Building exact organiser 60-feature matrices...", flush=True)
    madrid, _ = preprocess_city(ROOT / "Data" / "madrid_train.parquet")
    amsterdam, _ = preprocess_city(ROOT / "Data" / "amsterdam_data.parquet")
    names, _ = feature_columns()
    madrid_raw = madrid[names].to_numpy(dtype=np.float32)
    amsterdam_raw = amsterdam[names].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(madrid_raw)
    arrays = {
        "madrid_x": scaler.transform(madrid_raw).astype(np.float32),
        "madrid_y": madrid["age_class"].to_numpy(dtype=np.int64),
        "amsterdam_x": scaler.transform(amsterdam_raw).astype(np.float32),
        "amsterdam_y": amsterdam["age_class"].to_numpy(dtype=np.int64),
        "madrid_pixel_keys": madrid[["px_key", "py_key"]].to_numpy(dtype=np.int64),
        "amsterdam_pixel_keys": amsterdam[["px_key", "py_key"]].to_numpy(dtype=np.int64),
        "feature_names": np.asarray(names),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    return arrays


def load_or_build_splits(
    path: Path, labels: np.ndarray, trials: int, seed: int
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata == {"seed": seed, "trials": trials, "budgets": list(BUDGETS)}:
                return {
                    (budget, trial): (
                        data[f"support_{budget}_{trial}"],
                        data[f"query_{budget}_{trial}"],
                    )
                    for budget in BUDGETS
                    for trial in range(trials)
                }

    rng = np.random.default_rng(seed)
    payload: dict[str, np.ndarray] = {
        "metadata": np.asarray(json.dumps({"seed": seed, "trials": trials, "budgets": list(BUDGETS)}))
    }
    result = {}
    for budget in BUDGETS:
        for trial in range(trials):
            support = np.concatenate(
                [rng.choice(np.flatnonzero(labels == cls), budget, replace=False) for cls in CLASSES]
            )
            support = np.sort(support)
            query = np.flatnonzero(~np.isin(np.arange(len(labels)), support))
            payload[f"support_{budget}_{trial}"] = support
            payload[f"query_{budget}_{trial}"] = query
            result[(budget, trial)] = (support, query)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return result


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))


class TripletMLP:
    """Small ReLU MLP with L2 output normalization and NumPy backprop."""

    def __init__(self, dims: tuple[int, ...], seed: int):
        rng = np.random.default_rng(seed)
        self.weights = [
            (rng.standard_normal((din, dout)) * np.sqrt(2.0 / din)).astype(np.float32)
            for din, dout in zip(dims[:-1], dims[1:])
        ]
        self.biases = [np.zeros(dout, dtype=np.float32) for dout in dims[1:]]

    def forward(self, x: np.ndarray, cache: bool = False):
        activations = [x]
        preacts = []
        h = x
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = h @ weight + bias
            preacts.append(z)
            h = np.maximum(z, 0) if layer < len(self.weights) - 1 else z
            activations.append(h)
        norms = np.sqrt(np.sum(h * h, axis=1, keepdims=True) + 1e-8)
        embedding = h / norms
        if cache:
            return embedding, (activations, preacts, norms)
        return embedding

    def backward(self, grad_embedding: np.ndarray, cache):
        activations, preacts, norms = cache
        raw = activations[-1]
        dot = np.sum(grad_embedding * raw, axis=1, keepdims=True)
        grad = grad_embedding / norms - raw * dot / (norms**3)
        grad_w, grad_b = [], []
        batch_size = len(grad)
        for layer in reversed(range(len(self.weights))):
            grad_w.append(activations[layer].T @ grad / batch_size)
            grad_b.append(grad.mean(axis=0))
            if layer:
                grad = (grad @ self.weights[layer].T) * (preacts[layer - 1] > 0)
        return list(reversed(grad_w)), list(reversed(grad_b))


def train_embedding(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    margin: float,
    learning_rate: float,
    embedding_dim: int,
    negative_candidates: int,
) -> tuple[TripletMLP, list[float]]:
    rng = np.random.default_rng(seed)
    model = TripletMLP((x.shape[1], 64, 32, embedding_dim), seed)
    parameters = model.weights + model.biases
    first = [np.zeros_like(param) for param in parameters]
    second = [np.zeros_like(param) for param in parameters]
    class_indices = {cls: np.flatnonzero(y == cls) for cls in CLASSES}
    losses = []
    step = 0
    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(batches_per_epoch):
            anchor_classes = rng.choice(CLASSES, size=batch_size)
            anchor_idx = np.array([rng.choice(class_indices[c]) for c in anchor_classes])
            positive_idx = np.array([rng.choice(class_indices[c]) for c in anchor_classes])
            # Avoid identical anchor/positive pixels (extremely rare but easy to guarantee).
            same = positive_idx == anchor_idx
            while same.any():
                positive_idx[same] = [rng.choice(class_indices[c]) for c in anchor_classes[same]]
                same = positive_idx == anchor_idx
            # Online hard-negative mining from a small random candidate pool.
            # A pool of one exactly recovers ordinary random triplet sampling.
            negative_pool = np.empty((batch_size, negative_candidates), dtype=np.int64)
            for candidate in range(negative_candidates):
                negative_classes = np.array(
                    [rng.choice(CLASSES[CLASSES != cls]) for cls in anchor_classes]
                )
                negative_pool[:, candidate] = [
                    rng.choice(class_indices[c]) for c in negative_classes
                ]
            if negative_candidates == 1:
                negative_idx = negative_pool[:, 0]
            else:
                anchor_embedding = model.forward(x[anchor_idx])
                candidate_embedding = model.forward(x[negative_pool.reshape(-1)]).reshape(
                    batch_size, negative_candidates, -1
                )
                candidate_distances = np.sum(
                    (anchor_embedding[:, None, :] - candidate_embedding) ** 2, axis=2
                )
                negative_idx = negative_pool[
                    np.arange(batch_size), candidate_distances.argmin(axis=1)
                ]
            joined = np.concatenate([x[anchor_idx], x[positive_idx], x[negative_idx]])
            emb, cache = model.forward(joined, cache=True)
            a, p, n = np.split(emb, 3)
            d_pos = np.sum((a - p) ** 2, axis=1)
            d_neg = np.sum((a - n) ** 2, axis=1)
            active = d_pos - d_neg + margin > 0
            epoch_losses.append(float(np.maximum(d_pos - d_neg + margin, 0).mean()))
            grad = np.zeros_like(emb)
            grad_a = 2 * (n - p)
            grad_p = 2 * (p - a)
            grad_n = 2 * (a - n)
            grad[:batch_size][active] = grad_a[active]
            grad[batch_size : 2 * batch_size][active] = grad_p[active]
            grad[2 * batch_size :][active] = grad_n[active]
            grad_w, grad_b = model.backward(grad, cache)
            gradients = grad_w + grad_b
            step += 1
            for i, (param, gradient) in enumerate(zip(parameters, gradients)):
                np.clip(gradient, -5, 5, out=gradient)
                first[i] = 0.9 * first[i] + 0.1 * gradient
                second[i] = 0.999 * second[i] + 0.001 * gradient * gradient
                m_hat = first[i] / (1 - 0.9**step)
                v_hat = second[i] / (1 - 0.999**step)
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        losses.append(float(np.mean(epoch_losses)))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Triplet epoch {epoch + 1:3d}/{epochs}: loss={losses[-1]:.4f}", flush=True)
    return model, losses


def evaluate_representation(
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    trials: int,
) -> tuple[dict, dict[int, np.ndarray]]:
    results = {}
    confusion_totals = {budget: np.zeros((4, 4), dtype=np.int64) for budget in BUDGETS}
    for budget in BUDGETS:
        prototype_scores, logistic_scores = [], []
        for trial in range(trials):
            support, query = splits[(budget, trial)]
            proto_prediction = prototype_predict(x[support], y[support], x[query])
            prototype_scores.append(macro_f1(y[query], proto_prediction))
            confusion_totals[budget] += confusion_matrix(y[query], proto_prediction, labels=CLASSES)
            logistic = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
            logistic.fit(x[support], y[support])
            logistic_scores.append(macro_f1(y[query], logistic.predict(x[query])))
        results[str(budget)] = {
            "prototype": summarize(prototype_scores),
            "logistic_regression": summarize(logistic_scores),
        }
    return results, confusion_totals


def summarize(values: list[float]) -> dict:
    return {
        "scores": values,
        "macro_f1_mean": float(np.mean(values)),
        "macro_f1_std_population": float(np.std(values)),
    }


def make_plots(raw_results: dict, embedding_results: dict, mx, my, ax, ay, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    styles = [
        (raw_results, "prototype", "Raw prototype", "o-"),
        (raw_results, "logistic_regression", "Raw logistic", "s-"),
        (embedding_results, "prototype", "Embedding prototype", "o--"),
        (embedding_results, "logistic_regression", "Embedding logistic", "s--"),
    ]
    for result, method, label, style in styles:
        means = [result[str(b)][method]["macro_f1_mean"] for b in BUDGETS]
        stds = [result[str(b)][method]["macro_f1_std_population"] for b in BUDGETS]
        axis.errorbar(BUDGETS, means, yerr=stds, fmt=style, capsize=3, label=label)
    axis.set(xlabel="Amsterdam labels per class", ylabel="Amsterdam query macro F1", xscale="log")
    axis.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "macro_f1_vs_budget.png", dpi=180)
    plt.close(fig)

    rng = np.random.default_rng(42)
    m_idx = rng.choice(len(mx), min(6000, len(mx)), replace=False)
    a_idx = rng.choice(len(ax), min(6000, len(ax)), replace=False)
    joined = np.vstack([mx[m_idx], ax[a_idx]])
    coords = PCA(n_components=2).fit_transform(joined)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    offset = len(m_idx)
    for axis, points, labels, title in [
        (axes[0], coords[:offset], my[m_idx], "Madrid embeddings"),
        (axes[1], coords[offset:], ay[a_idx], "Amsterdam embeddings"),
    ]:
        scatter = axis.scatter(points[:, 0], points[:, 1], c=labels, cmap="viridis", s=4, alpha=0.35)
        axis.set_title(title)
        axis.set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    handles, _ = scatter.legend_elements()
    fig.legend(handles, [f"Class {c}" for c in CLASSES], loc="center right")
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    fig.savefig(out_dir / "embedding_pca_by_city.png", dpi=180)
    plt.close(fig)


def markdown_table(raw: dict, embedding: dict) -> str:
    lines = [
        "| Labels/class | Raw prototype | Raw logistic | Embedding prototype | Embedding logistic |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        vals = []
        for result, method in [
            (raw, "prototype"), (raw, "logistic_regression"),
            (embedding, "prototype"), (embedding, "logistic_regression"),
        ]:
            item = result[str(budget)][method]
            vals.append(f'{item["macro_f1_mean"]:.4f} +/- {item["macro_f1_std_population"]:.4f}')
        lines.append(f"| {budget} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batches-per-epoch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--negative-candidates", type=int, default=1)
    args = parser.parse_args()
    started = time.time()
    outputs = ROOT / "outputs"
    features = load_or_build_features(outputs / "features_60.npz")
    mx, my = features["madrid_x"], features["madrid_y"]
    ax, ay = features["amsterdam_x"], features["amsterdam_y"]
    splits = load_or_build_splits(outputs / "amsterdam_fixed_splits.npz", ay, args.trials, args.seed)

    print("Evaluating shared-split raw baselines...", flush=True)
    raw_results, _ = evaluate_representation(ax, ay, splits, args.trials)
    print("Training Madrid-only triplet embedding...", flush=True)
    model, losses = train_embedding(
        mx, my, args.seed, args.epochs, args.batches_per_epoch, args.batch_size,
        args.margin, args.learning_rate, args.embedding_dim, args.negative_candidates,
    )
    madrid_embedding = model.forward(mx)
    amsterdam_embedding = model.forward(ax)
    print("Evaluating Amsterdam support adaptation in embedding space...", flush=True)
    embedding_results, embedding_confusions = evaluate_representation(
        amsterdam_embedding, ay, splits, args.trials
    )
    make_plots(
        raw_results, embedding_results, madrid_embedding, my,
        amsterdam_embedding, ay, outputs / "figures",
    )

    config = vars(args)
    result = {
        "experiment": "Madrid-trained triplet embedding -> Amsterdam few-shot adaptation",
        "direction": "Madrid -> Amsterdam",
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "data": {
            "features": "outputs/features_60.npz", "feature_count": int(mx.shape[1]),
            "madrid_rows": len(mx), "amsterdam_rows": len(ax),
            "splits": "outputs/amsterdam_fixed_splits.npz",
            "split_policy": "10 deterministic stratified random trials; support/query disjoint",
            "scaling": "StandardScaler fit on Madrid only; same scaled features used by all methods",
        },
        "configuration": config,
        "metric_training": {
            "source_labels": "Madrid only", "query_labels_used": False,
            "architecture": [60, 64, 32, args.embedding_dim],
            "activations": "ReLU, ReLU, linear, L2 normalize",
            "loss": "squared-Euclidean triplet loss",
            "sampling": (
                "uniform class anchors; random same-class positive; closest of "
                f"{args.negative_candidates} random different-class negative candidate(s)"
            ),
            "epoch_losses": losses,
        },
        "raw_features": raw_results,
        "embedding": embedding_results,
        "embedding_prototype_confusion_counts": {
            str(k): v.tolist() for k, v in embedding_confusions.items()
        },
        "elapsed_seconds": time.time() - started,
    }
    output_path = outputs / "metric_learning_results.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(markdown_table(raw_results, embedding_results), flush=True)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
