import pandas as pd
from pathlib import Path
import shutil
import ast

def prepare_competition_data(source_dir, target_dir, metadata_csv):
    """
    Process competition datasets: split train.csv 90/10 into train/test and create sample_submission.csv

    Args:
        source_dir: Source directory containing competition folders
        target_dir: Target directory to save processed datasets
        metadata_csv: Path to Russian.csv with target column information
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Load metadata with target information
    metadata = pd.read_csv(metadata_csv)

    #metadata = metadata[metadata["comp-id"] == "alfa-university-income-prediction"]
    print(metadata)
    # Create a mapping from comp-id to target(s)
    target_map = {}
    for _, row in metadata.iterrows():
        comp_id = row.get('comp-id')
        target = row.get('target')
        if pd.notna(comp_id) and pd.notna(target):
            # Try to parse if it's a list representation
            if target.startswith('['):
                try:
                    target = ast.literal_eval(target)
                except:
                    pass
            target_map[comp_id] = target

    print(f"Loaded {len(target_map)} competitions from metadata\n")

    # Iterate through all folders in source directory
    for competition_folder in source_path.iterdir():
        if not competition_folder.is_dir():
            continue

        comp_id = competition_folder.name

        # Skip if not in metadata
        if comp_id not in target_map:
            print(f"Skipping {comp_id}: not found in metadata")
            continue

        train_file = competition_folder / "train.csv"

        # Skip if train.csv doesn't exist
        if not train_file.exists():
            print(f"Skipping {comp_id}: train.csv not found")
            continue

        print(f"Processing {comp_id}...")

        try:
            # Read train.csv
            df = pd.read_csv(train_file, encoding='ascii')

            # Get target column(s) from metadata
            targets = target_map[comp_id]
            if not isinstance(targets, list):
                targets = [targets]

            # Split 90/10
            split_idx = int(len(df) * 0.9)
            train_split = df.iloc[:split_idx]
            test_split = df.iloc[split_idx:]

            # Create target folder
            target_folder = target_path / comp_id
            target_folder.mkdir(parents=True, exist_ok=True)

            # Save train.csv
            #train_split.to_csv(target_folder / "train.csv", index=False)
            #print(f"  Saved train.csv ({len(train_split)} rows)")

            # Save test.csv without target columns
            #test_columns = [col for col in test_split.columns if col not in targets]
            #test_split[test_columns].to_csv(target_folder / "test.csv", index=False)
            #print(f"  Saved test.csv ({len(test_split)} rows, {len(test_columns)} columns)")

            # Save target.csv - only target columns
            target_columns = [col for col in test_split.columns if col in targets]
            test_split[target_columns].to_csv(target_folder / "target.csv", index=False)
            print(f"  Saved target.csv ({len(test_split)} rows, {len(target_columns)} columns)")

            # Create sample_submission.csv
            # Get ID column (first column)
            id_col_name = df.columns[0]
            test_ids = test_split[id_col_name]

            # Create submission dataframe starting with ID column
            submission = pd.DataFrame({id_col_name: test_ids})

            # Add target column(s) with sample values from first train row
            for target_col in targets:
                if target_col in train_split.columns:
                    sample_value = train_split.iloc[0][target_col]
                    submission[target_col] = sample_value
                else:
                    print(f"  Warning: target column '{target_col}' not found in train data")

            # Save sample_submission.csv
            #submission.to_csv(target_folder / "sample_submission.csv", index=False)
            #print(f"  Saved sample_submission.csv ({len(submission)} rows, {len(submission.columns)} columns)")

        except Exception as e:
            print(f"  Error processing {comp_id}: {e}")
            continue

if __name__ == "__main__":
    source_dir = "/home/stas/Documents/GitHub/ml2b/competitions/data"
    target_dir = "/home/stas/Documents/GitHub/FEDOT.LLM/experiments/ml2b"
    metadata_csv = "/home/stas/Documents/GitHub/FEDOT.LLM/experiments/ml2b/Russian.csv"

    prepare_competition_data(source_dir, target_dir, metadata_csv)
    print("\nDone!")
