from pathlib import Path
import json


def check_experiment_status(output_path: Path) -> str:
    """
    Check the status of an experiment based on output files.

    Returns:
        - "failed" if no output directory exists
        - "solution only" if only solution.py exists
        - "complete" if both solution.py and submission.csv exist
    """
    if not output_path.exists():
        return "failed"

    solution_file = output_path / "solution.py"
    submission_file = output_path / "submission.csv"

    has_solution = solution_file.exists()
    has_submission = submission_file.exists()

    if has_solution and has_submission:
        return "complete"
    elif has_solution:
        return "solution only"
    else:
        return "failed"


def summarize_experiments():
    """Summarize the status of all experiments based on competition_descriptions.json"""

    # Load competition descriptions
    descriptions_path = Path(__file__).parent / "competition_descriptions.json"

    if not descriptions_path.exists():
        print(f"Error: {descriptions_path} not found")
        return

    with open(descriptions_path, "r", encoding="utf-8") as f:
        competitions = json.load(f)

    ml2b_base = Path(__file__).parent / "ml2b"

    # Count statuses
    status_counts = {
        "complete": 0,
        "solution only": 0,
        "failed": 0
    }

    results = []

    for comp_id in competitions.keys():
        dataset_path = ml2b_base / comp_id
        output_path = dataset_path / "output"

        status = check_experiment_status(output_path)
        status_counts[status] += 1
        results.append((comp_id, status))

    # Print summary
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total competitions: {len(competitions)}")
    print(f"Complete: {status_counts['complete']}")
    print(f"Solution only: {status_counts['solution only']}")
    print(f"Failed: {status_counts['failed']}")
    print("=" * 80)

    # Print details by status
    for status_type in ["complete", "solution only", "failed"]:
        comps_with_status = [comp_id for comp_id, status in results if status == status_type]
        if comps_with_status:
            print(f"\n{status_type.upper()}:")
            for comp_id in comps_with_status:
                print(f"  • {comp_id}")


if __name__ == "__main__":
    summarize_experiments()
