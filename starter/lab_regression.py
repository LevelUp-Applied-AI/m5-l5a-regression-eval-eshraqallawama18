"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab _regression.py
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """

    df = pd.read_csv(filepath)

    print("Shape:", df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nChurn Distribution:")
    print(df["churned"].value_counts())
    print(df["churned"].value_counts(normalize=True))

    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Classification: use stratify for churned
    if target_col == "churned":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"Train size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")
        print(f"Train churn rate: {y_train.mean():.3f}")
        print(f"Test churn rate: {y_test.mean():.3f}")

    # Regression: do NOT use stratify
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        print(f"Train size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    return pipeline


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    return pipeline


def build_lasso_pipeline():
    """Build a Pipeline with StandardScaler and Lasso regression.

    Returns:
        sklearn Pipeline object.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.1))
    ])

    return pipeline

def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics."""

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    plt.show()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return metrics


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics."""

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")

    metrics = {
        "mae": mae,
        "r2": r2
    }

    return metrics


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """

    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy"
    )

    print("Fold Scores:")

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.3f}")

    print(f"\nMean Accuracy: {scores.mean():.3f}")
    print(f"Std Dev: {scores.std():.3f}")

    return scores


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Select numeric features for classification
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split
            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"Logistic Regression: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg
            ridge_pipe = build_ridge_pipeline()
            
            if ridge_pipe:
               reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
               print(f"Ridge Regression: {reg_metrics}")

               lasso_pipe = build_lasso_pipeline()

               ridge_pipe.fit(X_tr, y_tr)
               lasso_pipe.fit(X_tr, y_tr)

               ridge_coef = ridge_pipe.named_steps["ridge"].coef_
               lasso_coef = lasso_pipe.named_steps["lasso"].coef_

               print("\nFeature Coefficients")
               print("-" * 60)

               for feature, ridge, lasso in zip(X_tr.columns, ridge_coef, lasso_coef):
                     print(f"{feature:<20} Ridge={ridge:>8.3f}   Lasso={lasso:>8.3f}")

               print("\nFeatures driven to zero by Lasso:")

               for feature, coef in zip(X_tr.columns, lasso_coef):
                     if coef == 0:
                         print(feature)     
                         
                         
                         
"""
Task 7 - Summary of Findings

1. Tenure, monthly charges, and total charges appear to be among the
most important features for predicting customer churn because they
have larger model coefficients.

2. The logistic regression model achieved moderate performance.
Recall is more important than precision for churn prediction because
missing a customer who is likely to leave is usually more costly than
incorrectly predicting that a customer will churn.

3. To improve performance, I would try hyperparameter tuning,
feature engineering, threshold tuning, and more advanced models
such as Random Forest or Gradient Boosting.
"""           