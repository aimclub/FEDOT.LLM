import os
import pandas as pd
from pathlib import Path
import shutil

# Define source and target directories
source_root = Path("competition/SELA-datasets")
target_root = Path("competition/processed")

# Ensure the target root exists
target_root.mkdir(exist_ok=True)

# Iterate through each dataset subfolder
for dataset_folder in source_root.iterdir():
    if dataset_folder.is_dir():
        dataset_name = dataset_folder.name
        print(f"Processing dataset: {dataset_name}")

        # Define paths to input files
        train_file = dataset_folder / "split_train.csv"
        dev_file = dataset_folder / "split_dev.csv"
        test_file = dataset_folder / "split_test_wo_target.csv"
        test_target_file = dataset_folder / "split_test_target.csv"

        # Define output directory and ensure it exists
        output_dir = target_root / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read and concatenate train + dev
        train_df = pd.read_csv(train_file)
        dev_df = pd.read_csv(dev_file)
        train_combined = pd.concat([train_df, dev_df], ignore_index=True)

        # Save the combined train file
        train_combined.to_csv(output_dir / "train.csv", index=False)

        # Copy test_target as test.csv and sample_submission.csv
        shutil.copyfile(test_file, output_dir / "test.csv")
        shutil.copyfile(test_target_file, output_dir / "sample_submission.csv")

        print(f"Finished processing {dataset_name}")

print("All datasets processed.")