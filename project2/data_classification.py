"""
DecodeLabs - Project 2: Data Classification Using AI
------------------------------------------------------
Goal: Build a basic supervised-learning classification model on a
small, well-known dataset (Iris) and properly train/test/validate it.

Pipeline (IPO Model, matching the slide deck):

    INPUT   -> Load the Iris dataset, inspect it, scale features
               (StandardScaler: mean=0, variance=1)
    PROCESS -> Shuffle + split into train/test sets, tune "K",
               train a K-Nearest Neighbors classifier
    OUTPUT  -> Predict on the held-out test set, then validate with
               a confusion matrix, accuracy, precision, recall, and F1
               (because raw accuracy alone can be an "accuracy mirage")

Key skills demonstrated: data handling, supervised learning basics,
model training, and honest model evaluation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving plots to file
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------
# PHASE 1: INPUT — Load & Inspect the Dataset
# ---------------------------------------------------------------------
def load_and_inspect_data():
    iris = load_iris()
    X, y = iris.data, iris.target

    print("=" * 60)
    print("PHASE 1: INPUT — Raw Material (The Iris Benchmark)")
    print("=" * 60)
    print(f"Samples   : {X.shape[0]}")
    print(f"Features  : {X.shape[1]}  -> {iris.feature_names}")
    print(f"Classes   : {len(iris.target_names)} -> {list(iris.target_names)}")
    print()

    return X, y, iris.feature_names, iris.target_names


# ---------------------------------------------------------------------
# PHASE 2: PROCESS — Split, Scale, Tune, and Train
# ---------------------------------------------------------------------
def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Shuffle + split into train/test, then scale using stats from the
    TRAINING set only (fit_transform on train, transform on test) to
    avoid leaking test-set information into the model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 60)
    print("PHASE 2: PROCESS — Structural Integrity (The Split)")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    print("Feature scaling applied (StandardScaler: mean=0, variance=1)")
    print()

    return X_train_scaled, X_test_scaled, y_train, y_test


def find_optimal_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    """'Tuning the Engine': try a range of K values and track the error
    rate for each, so we can pick the K at the 'elbow' instead of
    guessing. Also saves an elbow-curve chart to disk."""
    error_rates = []

    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        error_rates.append(np.mean(predictions != y_test))

    optimal_k = list(k_range)[int(np.argmin(error_rates))]

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), error_rates, marker="o")
    plt.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal K = {optimal_k}")
    plt.title("Tuning the Engine: Choosing K")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("elbow_curve.png")
    plt.close()

    print("=" * 60)
    print("PHASE 2b: Tuning K (Elbow Method)")
    print("=" * 60)
    print(f"Optimal K found: {optimal_k}  (chart saved to elbow_curve.png)")
    print()

    return optimal_k


def train_model(X_train, y_train, k):
    """The Scikit-Learn workflow: Instantiate -> Fit -> (Predict later)."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------
# PHASE 3: OUTPUT — Predict & Validate
# ---------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, target_names):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names)

    print("=" * 60)
    print("PHASE 3: OUTPUT — Validation")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report (Precision / Recall / F1):")
    print(report)

    # Save a visual confusion matrix too, since a table of numbers is
    # harder to read at a glance than a labeled grid.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Iris Classification")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("\n(Confusion matrix chart saved to confusion_matrix.png)")

    return accuracy, cm, report


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    X, y, feature_names, target_names = load_and_inspect_data()
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    optimal_k = find_optimal_k(X_train, y_train, X_test, y_test)
    model = train_model(X_train, y_train, optimal_k)
    evaluate_model(model, X_test, y_test, target_names)


if __name__ == "__main__":
    main()
