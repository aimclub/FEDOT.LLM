import pandas as pd
import json

def create_competition_descriptions(csv_path):
    """
    Read competitions CSV, filter tabular competitions, and create descriptions dictionary.

    Args:
        csv_path: Path to the Russian.csv file

    Returns:
        Dictionary mapping comp-id to concatenated description
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Filter: data_type == "tabular" and comp-id is not NaN
    filtered_df = df[(df["data_type"] == "tabular") & (df["comp-id"].notna())]

    print(f"Total rows: {len(df)}")
    print(f"Filtered rows (tabular with comp-id): {len(filtered_df)}")

    # Create descriptions dictionary
    descriptions = {}

    for _, row in filtered_df.iterrows():
        comp_id = row["comp-id"]

        # Get columns and handle NaN values
        comp_name = str(row.get("comp_name", "")) if pd.notna(row.get("comp_name")) else ""
        domain = str(row.get("domain", "")) if pd.notna(row.get("domain")) else ""
        data_card = str(row.get("data_card", "")) if pd.notna(row.get("data_card")) else ""
        metric = str(row.get("metric", "")) if pd.notna(row.get("metric")) else ""
        description = str(row.get("description", "")) if pd.notna(row.get("description")) else ""

        # Concatenate with newlines
        full_description = "\n".join([
            #comp_name,
            #domain,
            #data_card,
            metric,
            #description
        ])

        descriptions[comp_id] = full_description

    return descriptions

if __name__ == "__main__":
    csv_path = "/home/stas/Documents/GitHub/ml2b/competitions/tasks/English.csv"

    descriptions_dict = create_competition_descriptions(csv_path)

    print(f"\nCreated descriptions for {len(descriptions_dict)} competitions")

    # Save to JSON file
    output_path = "/home/stas/Documents/GitHub/FEDOT.LLM/experiments/competition_descriptions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(descriptions_dict, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")

    # Print first example
    if descriptions_dict:
        first_comp_id = list(descriptions_dict.keys())[0]
        print(f"\nExample - Competition ID: {first_comp_id}")
        print("Description:")
        print(descriptions_dict[first_comp_id])
