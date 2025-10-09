"""Convert solution.py files from ml2b_ru to Jupyter notebooks."""
from pathlib import Path
import json


def create_notebook_from_solution(solution_path: Path, output_path: Path):
    """
    Convert a solution.py file to a Jupyter notebook.

    Args:
        solution_path: Path to the solution.py file
        output_path: Path where the notebook should be saved
    """
    # Read the solution code
    with open(solution_path, "r", encoding="utf-8") as f:
        solution_code = f.read()

    # Create notebook structure
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["!pip install fedot"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": solution_code.split("\n")
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write notebook
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


def main():
    """Convert all solution.py files in ml2b_ru to notebooks."""
    ml2b_ru_base = Path(__file__).parent / "ml2b_ru"
    notebooks_output = ml2b_ru_base / "y_russian_notebooks"

    # Find all solution.py files
    solution_files = list(ml2b_ru_base.glob("*/output/solution.py"))

    print(f"Found {len(solution_files)} solution files")

    converted = 0
    for solution_path in solution_files:
        # Extract competition name
        comp_id = solution_path.parent.parent.name

        # Create output path
        notebook_path = notebooks_output / f"{comp_id}.ipynb"

        try:
            create_notebook_from_solution(solution_path, notebook_path)
            print(f"✓ Converted {comp_id}")
            converted += 1
        except Exception as e:
            print(f"✗ Failed to convert {comp_id}: {e}")

    print(f"\nConversion complete: {converted}/{len(solution_files)} notebooks created")
    print(f"Output directory: {notebooks_output}")


if __name__ == "__main__":
    main()
