"""Local, explicitly supervised audio-feature classifiers.

Scores are model outputs for a user-labelled corpus. They are not calibrated
real-world authenticity probabilities.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASIC_FEATURE_NAMES = (
    "rms_dbfs", "peak_dbfs", "dc_offset_mean_amplitude", "clipping_sample_fraction_at_0_999",
    "low_energy_frame_fraction_relative_minus_40_db", "spectral_centroid_mean_hz",
    "spectral_flatness_mean", "spectral_entropy_normalized_mean", "spectral_rolloff_85_mean_hz",
    "zero_crossing_rate_mean", "spectral_flux_mean",
)
ADVANCED_FEATURE_NAMES = BASIC_FEATURE_NAMES + tuple(
    [f"{family}_mean_c{index}" for family in ("mfcc", "lfcc", "cqcc_style") for index in range(1, 5)]
) + (
    "mean_f0_hz", "f0_std_hz", "jitter_local_proxy_percent", "shimmer_local_proxy_percent",
    "autocorrelation_hnr_proxy_mean_db", "voiced_frame_fraction",
    "spectral_spread_mean_hz", "spectral_slope_mean_db_per_octave", "spectral_crest_mean",
    "spectral_flux_std", "frame_rms_std_amplitude", "high_band_power_fraction",
    "mean_residual_phase_variation_radians", "residual_phase_jump_ratio", "group_delay_outlier_ratio",
)
FEATURE_SETS = {"basic": BASIC_FEATURE_NAMES, "advanced": ADVANCED_FEATURE_NAMES}
ALGORITHMS = {
    "logistic_regression": "Logistic regression",
    "svm_rbf": "Support vector machine (RBF)",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "small_neural_network": "Small neural network (MLP)",
}


def feature_vector(report: dict) -> list[float]:
    t, f = report["time_domain"], report["frequency_domain"]
    values = [t.get(name) for name in BASIC_FEATURE_NAMES[:5]] + [f.get(name) for name in BASIC_FEATURE_NAMES[5:]]
    return _finite(values)


def advanced_feature_vector(report: dict) -> list[float]:
    values = feature_vector(report)
    cepstral = report.get("cepstral", {})
    for family in ("mfcc", "lfcc", "cqcc_style"):
        coefficients = cepstral.get(family, {}).get("mean_coefficients", [])
        values.extend(coefficients[1:5] if len(coefficients) >= 5 else [None] * 4)
    prosodic = report.get("prosodic", {})
    values.extend(prosodic.get(name) for name in (
        "mean_f0_hz", "f0_std_hz", "jitter_local_proxy_percent", "shimmer_local_proxy_percent",
        "autocorrelation_hnr_proxy_mean_db", "voiced_frame_fraction",
    ))
    spectral = report.get("extended_spectral", {})
    bands = spectral.get("band_power_fractions", {})
    values.extend(spectral.get(name) for name in (
        "spectral_spread_mean_hz", "spectral_slope_mean_db_per_octave", "spectral_crest_mean",
        "spectral_flux_std", "frame_rms_std_amplitude",
    ))
    values.append(bands.get("high_at_or_above_2000_hz"))
    phase = report.get("phase", {})
    values.extend(phase.get(name) for name in (
        "mean_residual_phase_variation_radians", "residual_phase_jump_ratio", "group_delay_outlier_ratio",
    ))
    return _finite(values)


def stored_feature_vectors(report: dict) -> dict[str, list[float]]:
    vectors = {"basic": feature_vector(report)}
    # Older saved History reports contain only the original feature schema.
    # Do not make missing advanced measurements look like real zero-valued
    # measurements; those rows can still train the basic model.
    if all(name in report for name in ("cepstral", "phase", "prosodic", "extended_spectral")):
        vectors["advanced"] = advanced_feature_vector(report)
    return vectors


def _finite(values) -> list[float]:
    return [0.0 if value is None or not np.isfinite(value) else float(value) for value in values]


def _values_for_set(features, feature_set: str) -> list[float] | None:
    if isinstance(features, dict):
        values = features.get(feature_set)
        return values if isinstance(values, list) else None
    if feature_set == "basic" and isinstance(features, list):
        return features
    return None


def _make_estimator(algorithm: str):
    if algorithm == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
    if algorithm == "svm_rbf":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42))
    if algorithm == "random_forest":
        return RandomForestClassifier(n_estimators=250, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=1)
    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=2, random_state=42)
    if algorithm == "small_neural_network":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1.0, max_iter=1000, early_stopping=False, random_state=42))
    raise ValueError("Choose Logistic regression, RBF SVM, Random forest, Gradient boosting, or Small neural network.")


def _prepare_split(examples: list[dict], feature_set: str):
    if feature_set not in FEATURE_SETS:
        raise ValueError("Choose the basic DSP or extended DSP/cepstral feature set.")
    usable = [row for row in examples if _values_for_set(row["features"], feature_set) is not None]
    if len(usable) < 16:
        raise ValueError(f"The {feature_set} feature set has only {len(usable)} labeled recordings; at least 16 are needed. Re-import labeled recordings to extract newer features.")
    labels = np.asarray([1 if row["label"] == "synthetic" else 0 for row in usable], dtype=np.int32)
    groups = np.asarray([row["group_id"] for row in usable], dtype=str)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Training needs both human and AI-generated labels.")
    group_counts = {label: len(set(groups[labels == target])) for label, target in (("human", 0), ("synthetic", 1))}
    if min(group_counts.values()) < 4:
        raise ValueError("Need at least four distinct recording groups in each class for a grouped holdout.")
    x = np.asarray([_values_for_set(row["features"], feature_set) for row in usable], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_SETS[feature_set]):
        raise ValueError("Saved training vectors do not match the selected feature schema. Re-import the recordings first.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_ix, test_ix = next(splitter.split(x, labels, groups))
    if len(set(labels[train_ix])) < 2 or len(set(labels[test_ix])) < 2:
        raise ValueError("The grouped holdout did not contain both classes. Add more balanced recording groups.")
    meta = {
        "train_files": int(len(train_ix)), "holdout_files": int(len(test_ix)),
        "train_groups": int(len(set(groups[train_ix]))), "holdout_groups": int(len(set(groups[test_ix]))),
        "class_counts": {"human": int((labels == 0).sum()), "synthetic": int((labels == 1).sum())},
        "distinct_groups_by_class": group_counts,
    }
    return usable, x, labels, groups, train_ix, test_ix, meta


def _fit(model, x, y, algorithm: str):
    if algorithm == "gradient_boosting":
        counts = np.bincount(y, minlength=2)
        weights = np.asarray([len(y) / (2.0 * max(counts[label], 1)) for label in y])
        model.fit(x, y, sample_weight=weights)
    elif algorithm == "small_neural_network":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x, y)
    else:
        model.fit(x, y)


def _evaluate(model, x, y, meta: dict) -> dict:
    scores = model.predict_proba(x)[:, 1]
    predictions = (scores >= 0.5).astype(np.int32)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    false_positive_rate, true_positive_rate, _ = roc_curve(y, scores, pos_label=1)
    false_negative_rate = 1.0 - true_positive_rate
    difference = false_positive_rate - false_negative_rate
    crossing = np.flatnonzero(difference >= 0)
    if crossing.size:
        upper = int(crossing[0])
        lower = max(0, upper - 1)
        span = difference[upper] - difference[lower]
        fraction = 0.0 if span == 0 else float(-difference[lower] / span)
        eer_fpr = false_positive_rate[lower] + fraction * (false_positive_rate[upper] - false_positive_rate[lower])
        eer_fnr = false_negative_rate[lower] + fraction * (false_negative_rate[upper] - false_negative_rate[lower])
        eer = float((eer_fpr + eer_fnr) / 2.0)
    else:
        eer_index = int(np.argmin(np.abs(difference)))
        eer = float((false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2.0)
    return {
        **meta,
        "holdout_accuracy": float(accuracy_score(y, predictions)),
        "holdout_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "holdout_auc": float(roc_auc_score(y, scores)),
        "holdout_eer": eer,
        "holdout_false_accept_rate_at_0_5": float(matrix[0, 1] / matrix[0].sum()),
        "holdout_false_reject_rate_at_0_5": float(matrix[1, 0] / matrix[1].sum()),
        "confusion_matrix_rows_true_human_synthetic_cols_pred_human_synthetic": matrix.tolist(),
    }


def compare_models(examples: list[dict], feature_set: str = "advanced") -> dict:
    """Run a same-split exploratory comparison; does not save or select a model."""
    usable, x, labels, _groups, train_ix, test_ix, meta = _prepare_split(examples, feature_set)
    comparisons = []
    for name, label in ALGORITHMS.items():
        try:
            model = _make_estimator(name)
            _fit(model, x[train_ix], labels[train_ix], name)
            comparisons.append({"algorithm": name, "name": label, "metrics": _evaluate(model, x[test_ix], labels[test_ix], meta)})
        except Exception as exc:
            comparisons.append({"algorithm": name, "name": label, "error": str(exc)[:220]})
    return {"feature_set": feature_set, "feature_count": x.shape[1], "labeled_examples": len(usable), "models": comparisons,
            "warning": "Exploratory same-split comparison only. Do not choose a production detector from this small holdout."}


def train_model(examples: list[dict], model_path: Path, *, algorithm: str = "logistic_regression", feature_set: str = "basic", training_source: str = "all") -> dict:
    usable, x, labels, _groups, train_ix, test_ix, meta = _prepare_split(examples, feature_set)
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown classifier. Choose one of the five listed model families.")
    evaluation_model = _make_estimator(algorithm)
    _fit(evaluation_model, x[train_ix], labels[train_ix], algorithm)
    metrics = _evaluate(evaluation_model, x[test_ix], labels[test_ix], meta)
    # Keep the grouped holdout estimate from the evaluation fit, then make the
    # deployable local artifact use every explicitly labeled feature vector.
    model = _make_estimator(algorithm)
    _fit(model, x, labels, algorithm)
    metrics["final_fit_examples"] = int(len(usable))
    bundle = {
        "version": 1, "classifier": model, "algorithm": algorithm, "algorithm_name": ALGORITHMS[algorithm],
        "feature_set": feature_set, "feature_names": list(FEATURE_SETS[feature_set]), "training_source": training_source,
        "positive_class": "synthetic", "threshold": 0.5, "score_is_calibrated_probability": False,
        "training_examples_used": len(usable), "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "warning": "Experimental user-labeled corpus model. Holdout metrics are exploratory estimates; the saved classifier is refit on all labeled examples. Scores are uncalibrated and are not real-world probabilities.",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return {key: value for key, value in bundle.items() if key != "classifier"}


def load_model(model_path: Path):
    if not model_path.is_file():
        return None
    bundle = joblib.load(model_path)
    feature_set = bundle.get("feature_set", "basic")
    expected = FEATURE_SETS.get(feature_set)
    if bundle.get("version") != 1 or expected is None or bundle.get("feature_names") != list(expected):
        raise ValueError("The saved model has an unsupported version or feature schema.")
    # Older workspace checkpoints predate the model-family metadata fields.
    # Recover their identity from the fitted sklearn estimator for display and
    # export, without changing the saved checkpoint or score behavior.
    if not bundle.get("algorithm") or not bundle.get("algorithm_name"):
        classifier = bundle["classifier"]
        steps = getattr(classifier, "named_steps", {})
        estimator = list(steps.values())[-1] if steps else classifier
        class_name = type(estimator).__name__
        inferred = {
            "LogisticRegression": "logistic_regression",
            "SVC": "svm_rbf",
            "RandomForestClassifier": "random_forest",
            "GradientBoostingClassifier": "gradient_boosting",
            "MLPClassifier": "small_neural_network",
        }.get(class_name)
        if inferred:
            if not bundle.get("algorithm"):
                bundle["algorithm"] = inferred
            if not bundle.get("algorithm_name"):
                bundle["algorithm_name"] = ALGORITHMS[inferred]
        else:
            if not bundle.get("algorithm"):
                bundle["algorithm"] = "legacy"
            if not bundle.get("algorithm_name"):
                bundle["algorithm_name"] = f"Legacy {class_name}"
    return bundle


def score_report(report: dict, bundle: dict) -> float:
    feature_set = bundle.get("feature_set", "basic")
    values = advanced_feature_vector(report) if feature_set == "advanced" else feature_vector(report)
    return float(bundle["classifier"].predict_proba(np.asarray([values], dtype=np.float64))[0, 1])


def set_model_threshold(model_path: Path, threshold: float) -> dict:
    """Persist a user-selected decision threshold without changing model weights."""
    if not 0.2 <= threshold <= 0.8:
        raise ValueError("Decision threshold must be between 0.20 and 0.80.")
    bundle = load_model(model_path)
    if bundle is None:
        raise ValueError("Train a local classifier before setting its decision threshold.")
    bundle["threshold"] = float(threshold)
    bundle["threshold_source"] = "user_selected"
    joblib.dump(bundle, model_path)
    return {"threshold": bundle["threshold"], "threshold_source": bundle["threshold_source"]}
