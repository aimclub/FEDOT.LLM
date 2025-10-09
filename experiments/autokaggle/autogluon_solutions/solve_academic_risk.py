#!/usr/bin/env python3
"""
Kaggle Playground Series S4E6 - Academic Risk Prediction Script using AutoGluon
This script predicts academic risk of students in higher education using AutoGluon's TabularPredictor
"""

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import os

# Configuration
DATA_DIR = "autokaggle/playground-series-s4e6"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(DATA_DIR, "submission.csv")
TARGET_COLUMN = "Target"
RANDOM_STATE = 42
TRAIN_VAL_SPLIT = 0.8

print("=" * 80)
print("Playground Series S4E6 - Academic Risk Prediction using AutoGluon")
print("=" * 80)

# Load data
print("\n1. Loading data...")
train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Display basic information
print(f"\nTarget distribution:")
print(train_data[TARGET_COLUMN].value_counts().sort_index())

# Identify columns to drop (id is not useful for prediction)
# We'll keep id for test data to create submission
columns_to_drop = ['id']

# Store test ids for submission
test_ids = test_data['id'].copy()

# Drop unnecessary columns from train data
print(f"\n2. Preprocessing data...")
print(f"Dropping columns: {columns_to_drop}")
train_features = train_data.drop(columns=columns_to_drop)
test_features = test_data.drop(columns=columns_to_drop)

print(f"Features used for training: {[col for col in train_features.columns if col != TARGET_COLUMN]}")
print(f"Number of features: {len(train_features.columns) - 1}")  # -1 for target

# Check for missing values
print(f"\nMissing values in train: {train_features.isnull().sum().sum()}")
print(f"Missing values in test: {test_features.isnull().sum().sum()}")

# Split data into train and validation (80/20 split)
print(f"\n3. Splitting data (80/20 train/val split)...")
train_df, val_df = train_test_split(
    train_features,
    test_size=1-TRAIN_VAL_SPLIT,
    random_state=RANDOM_STATE,
    stratify=train_features[TARGET_COLUMN]
)

print(f"Training set size: {len(train_df)} ({len(train_df)/len(train_features)*100:.1f}%)")
print(f"Validation set size: {len(val_df)} ({len(val_df)/len(train_features)*100:.1f}%)")

# Train AutoGluon model
print(f"\n4. Training AutoGluon TabularPredictor...")
print("This may take several minutes...")

predictor = TabularPredictor(
    label=TARGET_COLUMN,
    eval_metric='accuracy',  # Accuracy is the evaluation metric
    problem_type='multiclass',
    verbosity=2
)

predictor.fit(
    train_data=train_df,
    tuning_data=val_df,
    time_limit=60,  # 5 minutes time limit
    presets='best_quality',  # Use best quality preset for better performance
    use_bag_holdout=True  # Enable using tuning_data as holdout in bagged mode
)

# Evaluate on validation set
print("\n5. Evaluating model on validation set...")
val_performance = predictor.evaluate(val_df)
print(f"Validation performance: {val_performance}")

# Show feature importance
print("\n6. Feature importance:")
feature_importance = predictor.feature_importance(val_df)
print(feature_importance)

# Make predictions on test set
print("\n7. Making predictions on test set...")
predictions = predictor.predict(test_features)

# Create submission file
print(f"\n8. Creating submission file...")
submission = pd.DataFrame({
    'id': test_ids,
    'Target': predictions
})

submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission saved to: {SUBMISSION_FILE}")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))

print(f"\nPrediction distribution:")
print(submission['Target'].value_counts().sort_index())

print("\n" + "=" * 80)
print("Script completed successfully!")
print("=" * 80)
