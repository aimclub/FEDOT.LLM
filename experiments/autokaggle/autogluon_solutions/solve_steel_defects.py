#!/usr/bin/env python3
"""
Kaggle Playground Series S4E3 - Steel Plate Defect Prediction Script using AutoGluon
This script predicts probabilities for 7 types of steel plate defects using AutoGluon's TabularPredictor
Multi-label classification problem
"""

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import os

# Configuration
DATA_DIR = "autokaggle/playground-series-s4e3"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(DATA_DIR, "submission.csv")
TARGET_COLUMNS = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
RANDOM_STATE = 42
TRAIN_VAL_SPLIT = 0.8

print("=" * 80)
print("Playground Series S4E3 - Steel Plate Defect Prediction using AutoGluon")
print("=" * 80)

# Load data
print("\n1. Loading data...")
train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Display basic information about target columns
print(f"\nTarget columns distribution:")
for target in TARGET_COLUMNS:
    print(f"{target}: {train_data[target].sum()} positive samples ({train_data[target].mean()*100:.2f}%)")

# Store test ids for submission
test_ids = test_data['id'].copy()

# Identify feature columns (all except id and target columns)
feature_columns = [col for col in train_data.columns if col not in ['id'] + TARGET_COLUMNS]
print(f"\n2. Preprocessing data...")
print(f"Number of features: {len(feature_columns)}")

# Check for missing values
print(f"\nMissing values in train: {train_data[feature_columns].isnull().sum().sum()}")
print(f"Missing values in test: {test_data[feature_columns].isnull().sum().sum()}")

# Prepare data for multi-label prediction
# We'll train separate models for each target
predictions_dict = {}

for i, target in enumerate(TARGET_COLUMNS):
    print(f"\n{'='*80}")
    print(f"Training model {i+1}/{len(TARGET_COLUMNS)} for target: {target}")
    print(f"{'='*80}")

    # Prepare train data with current target
    train_subset = train_data[feature_columns + [target]].copy()
    test_subset = test_data[feature_columns].copy()

    # Split data into train and validation (80/20 split)
    print(f"\n3. Splitting data (80/20 train/val split)...")
    train_df, val_df = train_test_split(
        train_subset,
        test_size=1-TRAIN_VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=train_subset[target]
    )

    print(f"Training set size: {len(train_df)} ({len(train_df)/len(train_subset)*100:.1f}%)")
    print(f"Validation set size: {len(val_df)} ({len(val_df)/len(train_subset)*100:.1f}%)")

    # Train AutoGluon model
    print(f"\n4. Training AutoGluon TabularPredictor for {target}...")

    predictor = TabularPredictor(
        label=target,
        eval_metric='roc_auc',  # ROC AUC is the evaluation metric
        problem_type='binary',
        verbosity=2,
        path=f"AutogluonModels/s4e3_{target}"
    )

    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        time_limit=60,  # 4 minutes per target (total ~28 minutes for 7 targets)
        presets='best_quality',
        use_bag_holdout=True
    )

    # Evaluate on validation set
    print(f"\n5. Evaluating model on validation set...")
    val_performance = predictor.evaluate(val_df)
    print(f"Validation performance for {target}: {val_performance}")

    # Make predictions on test set (get probabilities)
    print(f"\n6. Making predictions on test set for {target}...")
    predictions_proba = predictor.predict_proba(test_subset)

    # For binary classification, get probability of positive class
    if isinstance(predictions_proba, pd.DataFrame):
        predictions_dict[target] = predictions_proba[1]  # Probability of class 1
    else:
        predictions_dict[target] = predictions_proba

# Create submission file
print(f"\n{'='*80}")
print(f"7. Creating submission file...")
print(f"{'='*80}")

submission = pd.DataFrame({'id': test_ids})
for target in TARGET_COLUMNS:
    submission[target] = predictions_dict[target]

submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission saved to: {SUBMISSION_FILE}")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))

print(f"\nPrediction statistics:")
for target in TARGET_COLUMNS:
    print(f"{target}: mean={submission[target].mean():.4f}, min={submission[target].min():.4f}, max={submission[target].max():.4f}")

print("\n" + "=" * 80)
print("Script completed successfully!")
print("=" * 80)
