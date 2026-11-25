"""Select HSV bins and LinearSVC C using training-only three-fold validation."""

import csv
import hashlib
import json
import platform
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import scipy
import sklearn
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from evaluate import score_predictions
from extract_color import CONFIGS
from prepare_splits import write_csv


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
C_VALUES = (0.1, 1.0, 10.0)
SEED = 42
FOLDS = 3
SVC_SETTINGS = dict(
    penalty="l2", loss="squared_hinge", dual="auto", multi_class="ovr",
    tol=1e-4, max_iter=10000, random_state=SEED, fit_intercept=True,
    intercept_scaling=1, class_weight=None,
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_features(dataset, bins, manifest, rows):
    """Reject stale or misaligned chapter 2 caches before fitting a classifier."""
    name = "x".join(map(str, bins))
    path = RESULTS / "cache" / "color" / f"{dataset}_{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. First run python code/extract_color.py.")
    with np.load(path, allow_pickle=False) as cache:
        if cache["manifest_sha256"].item() != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError(f"Stale manifest hash in {path}; regenerate the color cache.")
        for field in ("sample_id", "label", "split"):
            if not np.array_equal(cache[field], [row[field] for row in rows]):
                raise ValueError(f"Cache {field} does not match manifest rows in {path}.")
        if not np.array_equal(cache["bins"], bins):
            raise ValueError(f"Unexpected histogram bins in {path}.")
        x = cache["X"].copy()
    if (
        x.shape != (len(rows), int(np.prod(bins))) or not np.isfinite(x).all()
        or (x < 0).any() or not np.allclose(x.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError(f"Invalid feature dimensions or L1 normalization in {path}.")
    return x


def training_folds(y_train):
    """Use this same split rule for both global and future local classifiers."""
    return list(StratifiedKFold(
        n_splits=FOLDS, shuffle=True, random_state=SEED
    ).split(np.zeros(len(y_train)), y_train))


def fit_classifier(x, y, c):
    model = LinearSVC(C=c, **SVC_SETTINGS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model


def search_parameters(dataset, features, y, folds, labels):
    fold_rows = []
    search_rows = []
    for bins, x in features.items():
        for c in C_VALUES:
            trials = []
            for fold, (fit, validation) in enumerate(folds, 1):
                started = perf_counter()
                model = fit_classifier(x[fit], y[fit], c)
                fit_seconds = perf_counter() - started
                scores = score_predictions(y[validation], model.predict(x[validation]), labels)
                record = dict(
                    dataset=dataset, bins=bins, dimension=x.shape[1], C=c, fold=fold,
                    fit_samples=len(fit), validation_samples=len(validation),
                    accuracy=scores["accuracy"], macro_f1=scores["macro_f1"],
                    fit_seconds=fit_seconds, iterations=int(model.n_iter_),
                )
                trials.append(record)
            fold_rows.extend(trials)
            result = dict(
                dataset=dataset, bins=bins, dimension=x.shape[1], C=c,
                mean_macro_f1=float(np.mean([t["macro_f1"] for t in trials])),
                std_macro_f1=float(np.std([t["macro_f1"] for t in trials])),
                mean_accuracy=float(np.mean([t["accuracy"] for t in trials])),
                fit_seconds=sum(t["fit_seconds"] for t in trials),
            )
            search_rows.append(result)
            print(f"{dataset} {bins} C={c:g}: CV Macro-F1 {result['mean_macro_f1']:.6f}", flush=True)
    # Exact ties prefer fewer bins, followed by the smaller C.
    best = min(search_rows, key=lambda r: (-r["mean_macro_f1"], r["dimension"], r["C"]))
    return best, search_rows, fold_rows


def plot_confusion(path, counts, labels, dataset):
    counts = np.asarray(counts)
    fractions = counts / counts.sum(axis=1, keepdims=True)
    fig = Figure(figsize=(8.1, 6.6), layout="constrained")
    ax = fig.subplots()
    im = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    positions = np.arange(len(labels))
    ax.set(
        xticks=positions, xticklabels=labels, yticks=positions, yticklabels=labels,
        xlabel="Predicted class", ylabel="True class", title=dataset,
    )
    ax.tick_params(axis="x", labelrotation=55)
    for text in ax.get_xticklabels():
        text.set_horizontalalignment("right")
    for (row, col), count in np.ndenumerate(counts):
        ax.text(col, row, str(count), ha="center", va="center", fontsize=9,
                color="white" if fractions[row, col] > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Fraction within each true class")
    fig.savefig(path, dpi=180)


def run_dataset(dataset, output):
    manifest = RESULTS / "data_audit" / f"{dataset}_split.csv"
    rows = read_csv(manifest)
    ids = np.array([r["sample_id"] for r in rows])
    y = np.array([r["label"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    label_rows = read_csv(RESULTS / "data_audit" / f"{dataset}_labels.csv")
    labels = [r["label"] for r in sorted(label_rows, key=lambda r: int(r["label_id"]))]
    if (
        len(set(ids)) != len(ids) or set(splits) != {"train", "test"}
        or set(y[splits == "train"]) != set(labels) or set(y[splits == "test"]) != set(labels)
    ):
        raise ValueError(f"Invalid sample IDs, splits or class coverage for {dataset}.")
    train = splits == "train"
    test = splits == "test"
    features = {}
    for bins in CONFIGS:
        key = "x".join(map(str, bins))
        features[key] = load_features(dataset, bins, manifest, rows)
    folds = training_folds(y[train])
    best, search, fold_results = search_parameters(
        dataset, {key: x[train] for key, x in features.items()}, y[train], folds, labels
    )
    x = features[best["bins"]]
    started = perf_counter()
    model = fit_classifier(x[train], y[train], best["C"])
    fit_seconds = perf_counter() - started
    started = perf_counter()
    predicted = model.predict(x[test])
    predict_seconds = perf_counter() - started
    scores = score_predictions(y[test], predicted, labels)

    output.mkdir(parents=True, exist_ok=True)
    predictions = [dict(sample_id=i, y_true=t, y_pred=p)
                   for i, t, p in zip(ids[test], y[test], predicted, strict=True)]
    write_csv(output / f"{dataset}_predictions.csv", ["sample_id", "y_true", "y_pred"], predictions)
    write_csv(output / f"{dataset}_per_class.csv", list(scores["per_class"][0]), scores["per_class"])
    counts = [dict(y_true=label, **dict(zip(labels, row, strict=True)))
              for label, row in zip(labels, scores["confusion_matrix"], strict=True)]
    write_csv(output / f"{dataset}_confusion.csv", ["y_true", *labels], counts)
    assignments = np.empty(train.sum(), dtype=int)
    for fold, (_, validation) in enumerate(folds, 1):
        assignments[validation] = fold
    write_csv(output / f"{dataset}_folds.csv", ["sample_id", "label", "validation_fold"], [
        dict(sample_id=i, label=t, validation_fold=int(f))
        for i, t, f in zip(ids[train], y[train], assignments, strict=True)
    ])
    plot_confusion(output / f"{dataset}_confusion.png", scores["confusion_matrix"], labels, dataset)
    summary = dict(
        dataset=dataset, bins=best["bins"], dimension=best["dimension"], C=best["C"],
        train_samples=int(train.sum()), test_samples=int(test.sum()),
        cv_macro_f1=best["mean_macro_f1"], cv_std_macro_f1=best["std_macro_f1"],
        accuracy=scores["accuracy"], macro_f1=scores["macro_f1"],
        fit_seconds=fit_seconds, predict_seconds=predict_seconds, iterations=int(model.n_iter_),
    )
    cache = RESULTS / "cache" / "color_models"
    cache.mkdir(parents=True, exist_ok=True)
    # Numeric model parameters support independent prediction checks without unpickling.
    np.savez_compressed(cache / f"{dataset}.npz", coef=model.coef_, intercept=model.intercept_,
                        classes=model.classes_, bins=np.array(best["bins"]), C=best["C"])
    config = dict(
        **summary, labels=labels, estimator_settings=SVC_SETTINGS,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        feature_file=f"code/results/cache/color/{dataset}_{best['bins']}.npz",
        feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
    )
    print(f"{dataset}: selected {best['bins']}, C={best['C']:g}; "
          f"test Accuracy={scores['accuracy']:.6f}, Macro-F1={scores['macro_f1']:.6f}", flush=True)
    return summary, search, fold_results, config


def main():
    output = RESULTS / "color_classification"
    summaries, searches, trials, datasets = [], [], [], {}
    # One numerical thread makes runtime measurements and repeated fits comparable.
    with threadpool_limits(limits=1):
        for dataset in ("food101", "imagenette"):
            summary, search, folds, config = run_dataset(dataset, output)
            summaries.append(summary)
            searches.extend(search)
            trials.extend(folds)
            datasets[dataset] = config
    write_csv(output / "summary.csv", list(summaries[0]), summaries)
    write_csv(output / "cv_search.csv", list(searches[0]), searches)
    write_csv(output / "cv_folds.csv", list(trials[0]), trials)
    settings = dict(
        seed=SEED, folds=FOLDS, shuffle=True, C_candidates=C_VALUES, bin_candidates=CONFIGS,
        selection="Largest mean validation Macro-F1; exact ties by dimension, then C.",
        preprocessing="Chapter 2 per-image L1 histogram; no further transform.",
        macro_f1="Unweighted mean of F1 over the fixed complete class list; zero_division=0.",
        confusion_orientation="Rows are true classes; columns are predicted classes.",
        test_policy="Only the selected model is evaluated on the official final test split.",
        timing="CV fit sums; final fit and predict exclude feature extraction, IO and scoring.",
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        scikit_learn=sklearn.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform(), processor=platform.processor(), numerical_threads=1,
        datasets=datasets,
    )
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
