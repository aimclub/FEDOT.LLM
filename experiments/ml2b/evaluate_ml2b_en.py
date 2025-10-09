"""
Script to evaluate ML2B Russian experiment submissions against ground truth targets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def load_competition_info() -> Tuple[Dict[str, Union[str, List[str]]], Dict[str, str]]:
    """Load target columns and metrics from JSON files."""
    with open("experiments/target_columns.json", "r") as f:
        target_columns = json.load(f)

    with open("experiments/competition_descriptions.json", "r") as f:
        metrics = json.load(f)

    return target_columns, metrics


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_map_at_k(y_true: pd.DataFrame, y_pred: pd.DataFrame, k: int = 5) -> float:
    """Calculate Mean Average Precision at K for multi-label classification."""
    # This is a simplified implementation - adjust based on actual competition format
    scores = []
    for true_labels, pred_labels in zip(y_true.values, y_pred.values):
        # Convert to lists and handle properly
        true_set = set(str(true_labels).split())
        pred_list = str(pred_labels).split()[:k]

        if len(true_set) == 0:
            continue

        score = 0.0
        num_hits = 0

        for i, pred in enumerate(pred_list, 1):
            if pred in true_set:
                num_hits += 1
                score += num_hits / i

        scores.append(score / min(len(true_set), k))

    return np.mean(scores) if scores else 0.0


def calculate_wmae(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Calculate Weighted Mean Absolute Error (specific implementation needed)."""
    # Simplified implementation - adjust based on competition specifics
    return mean_absolute_error(y_true, y_pred)


def calculate_f_beta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0) -> float:
    """Calculate F-Beta score."""
    # For binary predictions, convert probabilities if needed
    if y_pred.dtype == float and np.all((y_pred >= 0) & (y_pred <= 1)):
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = y_pred

    return f1_score(y_true, y_pred_binary, beta=beta, average='binary')


def evaluate_submission(
    submission_path: Path,
    target_path: Path,
    target_cols: Union[str, List[str]],
    metric_name: str,
) -> Tuple[float, str]:
    """
    Evaluate a single submission against ground truth.

    Returns:
        Tuple of (score, status) where status is 'success' or error message
    """
    try:
        # Load submission and target files
        submission = pd.read_csv(submission_path)
        target = pd.read_csv(target_path)

        # Handle single or multiple target columns
        if isinstance(target_cols, list):
            cols = target_cols
        else:
            cols = [target_cols]

        # Verify all required columns exist
        for col in cols:
            if col not in submission.columns:
                return np.nan, f"Missing column: {col} in submission"
            if col not in target.columns:
                return np.nan, f"Missing column: {col} in target"

        # Check row counts match
        if len(submission) != len(target):
            return np.nan, f"Row count mismatch: submission={len(submission)}, target={len(target)}"

        # Calculate metric based on competition type
        if len(cols) == 1:
            y_true = target[cols[0]].values
            y_pred = submission[cols[0]].values
        else:
            y_true = target[cols].values
            y_pred = submission[cols].values

        # Calculate appropriate metric
        metric_lower = metric_name.lower()

        if metric_lower == "rmse":
            score = calculate_rmse(y_true, y_pred)
        elif metric_lower == "mae":
            score = mean_absolute_error(y_true, y_pred)
        elif metric_lower == "auc":
            # For multi-class, use average
            if len(np.unique(y_true)) > 2:
                score = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
            else:
                score = roc_auc_score(y_true, y_pred)
        elif metric_lower == "accuracy score" or metric_lower == "categorizationaccuracy":
            # Round predictions if they're probabilities
            if y_pred.dtype == float:
                y_pred = np.round(y_pred).astype(int)
            score = accuracy_score(y_true, y_pred)
        elif metric_lower == "logloss":
            # Ensure predictions are probabilities
            score = log_loss(y_true, y_pred)
        elif metric_lower == "r2_score":
            score = r2_score(y_true, y_pred)
        elif metric_lower == "f_beta":
            score = calculate_f_beta(y_true, y_pred)
        elif metric_lower == "map" or "map@" in metric_lower:
            k = 5 if "@" not in metric_lower else int(metric_lower.split("@")[1])
            score = calculate_map_at_k(target[cols], submission[cols], k=k)
        elif metric_lower == "wmae":
            score = calculate_wmae(target[cols], submission[cols])
        else:
            return np.nan, f"Unknown metric: {metric_name}"

        return score, "success"

    except Exception as e:
        return np.nan, f"Error: {str(e)}"


def main():
    """Main evaluation function."""
    # Setup paths
    ml2b_dir = Path("experiments/ml2b_en")
    ml2b_target_dir = Path("experiments/ml2b_target")

    # Load configuration
    target_columns, metrics = load_competition_info()

    # Results storage
    results = []

    # Get list of competitions from target directory
    competitions = sorted([d.name for d in ml2b_target_dir.iterdir() if d.is_dir()])

    print(f"Evaluating {len(competitions)} competitions from {ml2b_dir}...")

    for competition in competitions:
        # Check if competition is in target_columns.json
        if competition not in target_columns:
            print(f"  Skipping {competition}: not in target_columns.json")
            continue

        submission_path = ml2b_dir / competition / "output" / "submission.csv"
        target_path = ml2b_target_dir / competition / "target.csv"

        # Check if submission exists
        if not submission_path.exists():
            print(f"  {competition}: submission not found")
            results.append({
                "competition": competition,
                "metric": metrics.get(competition, "unknown"),
                "score": np.nan,
                "status": "no submission file",
            })
            continue

        # Check if target exists
        if not target_path.exists():
            print(f"  {competition}: target not found")
            results.append({
                "competition": competition,
                "metric": metrics.get(competition, "unknown"),
                "score": np.nan,
                "status": "no target file",
            })
            continue

        # Get metric and target columns
        metric_name = metrics.get(competition, "unknown")
        target_cols = target_columns[competition]

        # Evaluate
        score, status = evaluate_submission(
            submission_path, target_path, target_cols, metric_name
        )

        results.append({
            "competition": competition,
            "metric": metric_name,
            "score": score,
            "status": status,
        })

        if status == "success":
            print(f"  {competition}: {metric_name} = {score:.6f}")
        else:
            print(f"  {competition}: {status}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = "experiments/ml2b_ru_evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary statistics
    successful = results_df[results_df["status"] == "success"]
    print(f"\nSummary:")
    print(f"  Total competitions: {len(results_df)}")
    print(f"  Successful evaluations: {len(successful)}")
    print(f"  Failed evaluations: {len(results_df) - len(successful)}")

    if len(successful) > 0:
        print(f"\nSuccessful evaluations:")
        for _, row in successful.iterrows():
            print(f"  {row['competition']:50s} {row['metric']:20s} {row['score']:.6f}")


if __name__ == "__main__":
    main()
