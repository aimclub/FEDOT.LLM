import pandas as pd
from pathlib import Path

def extract_target_from_description(row):
    """
    Extract target column name(s) from the data_card and description fields.
    Returns either a single target name or a list of target names.
    """
    comp_id = row.get('comp-id', '')
    data_card = str(row.get('data_card', ''))
    description = str(row.get('description', ''))

    # Manual mapping based on analysis of the descriptions
    target_mapping = {
        'wids-datathon-2020': 'hospital_death',
        'ieor-242-nyc-taxi': 'duration',
        'explicit-content-detection': 'target',
        'emnist-handwritten-chars': 'label',
        'uwaterloo-stat441-jewelry': 'Revenue',
        'movie-genre-classification': 'label',
        'financial-engineering-1': 'col_5',
        'financial-engineering-2': 'col_5',
        'financial-engineering-3': ['col_5', 'col_8'],
        'biker-tour-recommendation': 'interested',
        'actuarial-loss-prediction': 'UltimateIncurredClaimCost',
        'she-hacks-2021': 'Count',
        'ml2021spring-hw1': 'tested_positive',
        'ai-cancer-predictions': 'diagnosis',
        'syde-522-winter-2021': 'label',
        'tabular-playground-series-aug-2021': 'loss',
        'google-brain-ventilator': 'pressure',
        'porto-seguro-challenge': 'y',
        'crime-learn': 'ViolentCrimesPerPop',
        'stroke-prediction-s3e2': 'stroke',
        'alfa-university-income-prediction': 'income',
        'playground-series-s5e6': 'Fertilizer Name',
        'ml-olympiad-bd-2025': 'RiskLevel',
        '2024-datalab-cup1': 'Popularity',
        'ece460j-fall24': 'Status',
        'thapar-summer-school-2025-hack-iii': 'output',
        'rutgers-data101-fall2022-assignment-12': 'trip_duration',
    }

    return target_mapping.get(comp_id, None)

def add_target_column(csv_path, output_path):
    """
    Read English.csv and add a target column with target column name(s).

    Args:
        csv_path: Path to the English.csv file
        output_path: Path to save the updated CSV
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    print(f"Read {len(df)} rows from {csv_path}")

    # Add target column
    df['target'] = df.apply(extract_target_from_description, axis=1)

    # Convert list values to string representation
    df['target'] = df['target'].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Save the updated CSV
    df.to_csv(output_path, index=False)

    print(f"Saved updated CSV to {output_path}")

    # Print summary
    print(f"\nTarget column summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with target: {df['target'].notna().sum()}")
    print(f"  Rows without target: {df['target'].isna().sum()}")

    # Show examples
    print(f"\nExamples with targets:")
    examples = df[df['target'].notna()][['comp_name', 'comp-id', 'target']].head(10)
    for _, row in examples.iterrows():
        print(f"  {row['comp-id']}: {row['target']}")

if __name__ == "__main__":
    csv_path = "/home/stas/Documents/GitHub/FEDOT.LLM/experiments/ml2b/Russian.csv"
    output_path = "/home/stas/Documents/GitHub/FEDOT.LLM/experiments/ml2b/Russian_with_targets.csv"

    add_target_column(csv_path, output_path)
