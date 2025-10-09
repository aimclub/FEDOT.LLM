#!/usr/bin/env python3
"""
MLZero Tasks Evaluation Script
Evaluates the performance of solutions against target.csv files
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
import os

# Task configurations: (directory, metric_type, target_column)
TASKS = {
    'abalone': {
        'dir': 'mlzero/abalone/competition',
        'metric': 'RMSE',
        'target_col': 'rings',
        'submission_file': 'submission.csv'
    },
    'airlines': {
        'dir': 'mlzero/airlines/competition',
        'metric': 'AUC',
        'target_col': 'Delay',
        'submission_file': 'submission.csv'
    },
    'covertype': {
        'dir': 'mlzero/covertype/competition',
        'metric': 'F1_weighted',
        'target_col': 'class',
        'submission_file': 'submission.csv'
    },
    'bio': {
        'dir': 'mlzero/bio/competition',
        'metric': 'AUC',
        'target_col': 'target',
        'submission_file': 'submission.csv'
    },
    'yolanda': {
        'dir': 'mlzero/yolanda/competition',
        'metric': 'RMSE',
        'target_col': '101',
        'submission_file': 'submission.csv'
    }
}

def calculate_metric(y_true, y_pred, metric_type, target_col=None):
    """Calculate the specified metric"""
    if metric_type == 'RMSE':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric_type == 'AUC':
        return roc_auc_score(y_true, y_pred)
    elif metric_type == 'F1_weighted':
        return f1_score(y_true, y_pred, average='weighted')
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

def evaluate_task(task_name, config):
    """Evaluate a single task"""
    try:
        # Load target file (try competition dir first, then parent dir)
        target_file = os.path.join(config['dir'], 'target.csv')
        if not os.path.exists(target_file):
            # Try parent directory
            parent_dir = os.path.dirname(config['dir'])
            target_file = os.path.join(parent_dir, 'target.csv')
            if not os.path.exists(target_file):
                return None, f"Target file not found in {config['dir']} or {parent_dir}"

        target_df = pd.read_csv(target_file)

        # Load submission file
        submission_file = os.path.join(config['dir'], config['submission_file'])
        if not os.path.exists(submission_file):
            return None, f"Submission file not found: {submission_file}"

        submission_df = pd.read_csv(submission_file)

        # Merge on id to ensure alignment
        merged = target_df.merge(submission_df, on='id', suffixes=('_true', '_pred'))

        # Get true and predicted values
        target_col = config['target_col']
        y_true = merged[f'{target_col}_true'] if f'{target_col}_true' in merged.columns else merged[target_col]

        # Handle different column naming conventions
        if f'{target_col}_pred' in merged.columns:
            y_pred = merged[f'{target_col}_pred']
        elif target_col in submission_df.columns:
            y_pred = merged[target_col] if f'{target_col}_true' in merged.columns else submission_df[target_col]
        else:
            return None, f"Could not find prediction column for {target_col}"

        # Calculate metric
        score = calculate_metric(y_true, y_pred, config['metric'], target_col)

        return score, None

    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    print("=" * 80)
    print("MLZero Tasks Performance Evaluation")
    print("=" * 80)

    results = {}

    for task_name, config in TASKS.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {task_name.upper()}")
        print(f"Directory: {config['dir']}")
        print(f"Metric: {config['metric']}")
        print(f"Target column: {config['target_col']}")
        print(f"{'='*80}")

        score, error = evaluate_task(task_name, config)

        if error:
            print(f"❌ {error}")
            results[task_name] = {'metric': config['metric'], 'score': None, 'status': 'Failed', 'error': error}
        else:
            print(f"✅ {config['metric']} Score: {score:.6f}")
            results[task_name] = {'metric': config['metric'], 'score': score, 'status': 'Success', 'error': None}

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Task':<15} {'Metric':<15} {'Score':<20} {'Status':<10}")
    print("-" * 80)

    for task_name, result in results.items():
        score_str = f"{result['score']:.6f}" if result['score'] is not None else "N/A"
        print(f"{task_name:<15} {result['metric']:<15} {score_str:<20} {result['status']:<10}")

    print("=" * 80)

    # Count successes
    successful = sum(1 for r in results.values() if r['status'] == 'Success')
    total = len(results)
    print(f"\nSuccessfully evaluated: {successful}/{total} tasks")

    if successful < total:
        print("\nFailed tasks:")
        for task_name, result in results.items():
            if result['status'] == 'Failed':
                print(f"  - {task_name}: {result['error']}")

if __name__ == "__main__":
    main()
