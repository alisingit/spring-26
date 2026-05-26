from itertools import permutations
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from gmm import GaussianMixtureEM


RANDOM_STATE = 42
N_COMPONENTS = 3
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataset = load_iris(as_frame=True)
    x = dataset.data
    y = dataset.target.to_numpy(dtype=int)
    target_names = dataset.target_names
    return x, y, target_names


def build_custom_model() -> GaussianMixtureEM:
    return GaussianMixtureEM(
        n_components=N_COMPONENTS,
        max_iter=300,
        tol=1e-5,
        reg_covar=1e-6,
        n_init=10,
        random_state=RANDOM_STATE,
    )


def build_sklearn_model() -> GaussianMixture:
    return GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="full",
        max_iter=300,
        tol=1e-5,
        reg_covar=1e-6,
        n_init=10,
        random_state=RANDOM_STATE,
    )


def clustering_accuracy(
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    n_components: int,
) -> tuple[float, dict[int, int], np.ndarray]:
    classes = np.unique(y_true)
    clusters = np.arange(n_components)
    if len(classes) != len(clusters):
        raise ValueError("number of classes must match number of clusters for accuracy calculation")

    best_accuracy = -1.0
    best_mapping: dict[int, int] = {}
    best_predictions = np.empty_like(y_true)

    for class_order in permutations(classes):
        mapping = {int(cluster): int(class_label) for cluster, class_label in zip(clusters, class_order, strict=True)}
        predictions = np.array([mapping[int(label)] for label in cluster_labels], dtype=int)
        accuracy = float(np.mean(predictions == y_true))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping
            best_predictions = predictions

    return best_accuracy, best_mapping, best_predictions


def evaluate_model(model, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    clusters = model.predict(x)
    accuracy, mapping, mapped_predictions = clustering_accuracy(y, clusters, model.n_components)
    return {
        "log_likelihood": float(model.score(x)),
        "bic": float(model.bic(x)),
        "aic": float(model.aic(x)),
        "accuracy": accuracy,
        "ari": float(adjusted_rand_score(y, clusters)),
        "mapping": mapping,
        "confusion_matrix": confusion_matrix(y, mapped_predictions),
        "clusters": clusters,
        "mapped_predictions": mapped_predictions,
    }


def run_experiment(x: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float | int | str]] = []
    fitted_models: dict[str, object] = {}

    for model_name, factory in (
        ("Собственная GMM", build_custom_model),
        ("sklearn GMM", build_sklearn_model),
    ):
        model = factory()
        start_time = perf_counter()
        model.fit(x)
        fit_time = perf_counter() - start_time

        metrics = evaluate_model(model, x, y)
        rows.append(
            {
                "model": model_name,
                "log_likelihood": metrics["log_likelihood"],
                "bic": metrics["bic"],
                "aic": metrics["aic"],
                "accuracy": metrics["accuracy"],
                "ari": metrics["ari"],
                "fit_time_sec": fit_time,
                "n_iter": int(model.n_iter_),
                "converged": str(bool(model.converged_)),
            }
        )
        fitted_models[model_name] = {
            "model": model,
            "fit_time_sec": fit_time,
            **metrics,
        }

    return pd.DataFrame(rows), fitted_models


def plot_metric_comparison(results: pd.DataFrame) -> None:
    metrics = ["log_likelihood", "accuracy", "ari"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    for ax, metric in zip(axes, metrics, strict=True):
        ax.bar(results["model"], results[metric], color=["#4C72B0", "#55A868"])
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=10)

    fig.suptitle("Сравнение собственной GMM и sklearn")
    plt.savefig(ARTIFACTS_DIR / "metrics_comparison.png", dpi=160)
    plt.close()


def plot_likelihood_curve(custom_model: GaussianMixtureEM) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(custom_model.log_likelihood_history_) + 1), custom_model.log_likelihood_history_, marker="o")
    ax.set_title("Сходимость EM-алгоритма")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Средний log-likelihood")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "likelihood_curve.png", dpi=160)
    plt.close()


def plot_pca_clusters(
    x: np.ndarray,
    y: np.ndarray,
    fitted_models: dict[str, object],
    target_names: np.ndarray,
) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    projected = pca.fit_transform(x)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [("Истинные классы", y)]
    panels.extend((model_name, result["mapped_predictions"]) for model_name, result in fitted_models.items())

    for ax, (title, labels) in zip(axes, panels, strict=True):
        scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="viridis", edgecolor="black", linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

    legend = axes[0].legend(
        handles=scatter.legend_elements()[0],
        labels=[str(name) for name in target_names],
        title="Класс",
        loc="best",
    )
    axes[0].add_artist(legend)
    fig.suptitle("PCA-проекция тестовой выборки")
    plt.savefig(ARTIFACTS_DIR / "pca_clusters.png", dpi=160)
    plt.close()


def save_results(results: pd.DataFrame, fitted_models: dict[str, object]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ARTIFACTS_DIR / "results.csv", index=False)

    custom_result = fitted_models["Собственная GMM"]
    custom_model = custom_result["model"]
    plot_metric_comparison(results)
    plot_likelihood_curve(custom_model)
    plot_pca_clusters(custom_result["x"], custom_result["y"], fitted_models, custom_result["target_names"])


def save_run_summary(
    x: pd.DataFrame,
    y: np.ndarray,
    target_names: np.ndarray,
    results: pd.DataFrame,
    fitted_models: dict[str, object],
) -> None:
    lines = [
        "# Сводка запуска lab4",
        "",
        f"Датасет: Iris из `sklearn.datasets`, объектов: {x.shape[0]}, признаков: {x.shape[1]}.",
        "Классы: " + ", ".join(f"{name}={int(np.sum(y == index))}" for index, name in enumerate(target_names)) + ".",
        "",
        "## Результаты",
        "",
        "| Модель | Mean LL | BIC | AIC | Accuracy | ARI | Fit time, sec | Iterations |",
        "|--------|---------|-----|-----|----------|-----|---------------|------------|",
    ]

    for _, row in results.iterrows():
        lines.append(
            "| {model} | {ll:.4f} | {bic:.2f} | {aic:.2f} | {accuracy:.4f} | "
            "{ari:.4f} | {time:.3f} | {n_iter} |".format(
                model=row["model"],
                ll=row["log_likelihood"],
                bic=row["bic"],
                aic=row["aic"],
                accuracy=row["accuracy"],
                ari=row["ari"],
                time=row["fit_time_sec"],
                n_iter=int(row["n_iter"]),
            )
        )

    lines.extend(["", "## Матрицы ошибок после оптимального сопоставления компонент", ""])
    for model_name, result in fitted_models.items():
        lines.append(f"### {model_name}")
        lines.append("")
        matrix = result["confusion_matrix"]
        lines.extend(
            [
                "| Истинный \\ Предсказанный | class_0 | class_1 | class_2 |",
                "|--------------------------|---------|---------|---------|",
            ]
        )
        for row_index, row in enumerate(matrix):
            lines.append(f"| class_{row_index} | {row[0]} | {row[1]} | {row[2]} |")
        lines.append("")

    (ARTIFACTS_DIR / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    x, y, target_names = load_dataset()
    x_scaled = StandardScaler().fit_transform(x)
    results, fitted_models = run_experiment(x_scaled, y)

    for result in fitted_models.values():
        result["x"] = x_scaled
        result["y"] = y
        result["target_names"] = target_names

    save_results(results, fitted_models)
    save_run_summary(x, y, target_names, results, fitted_models)
    print((ARTIFACTS_DIR / "run_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
