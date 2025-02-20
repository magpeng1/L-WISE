import argparse
import pandas as pd
import sys


def calculate_difficulty(dirmap_df):
    dirmap_df['difficulty'] = 1 - (
        (dirmap_df['robust_gt_logit'] - dirmap_df['robust_gt_logit'].min()) /
        (dirmap_df['robust_gt_logit'].max() - dirmap_df['robust_gt_logit'].min())
    )
    return dirmap_df


def main():
    parser = argparse.ArgumentParser(description='Process dirmap CSV to add difficulty column')
    parser.add_argument('--dirmap_path', type=str, required=True,
                       help='Path to the dirmap CSV file (will be overwritten)')
    args = parser.parse_args()

    # Read the CSV
    try:
        dataset_dirmap = pd.read_csv(args.dirmap_path)
    except FileNotFoundError:
        print(f"Error: Could not find file at {args.dirmap_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Check for required column
    if 'robust_gt_logit' not in dataset_dirmap.columns:
        print("Error: CSV must contain 'robust_gt_logit' column")
        sys.exit(1)

    # Check if difficulty column exists
    if 'difficulty' in dataset_dirmap.columns:
        response = input("'difficulty' column already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborting.")
            sys.exit(0)

    # Calculate difficulty
    dataset_dirmap = calculate_difficulty(dataset_dirmap)

    # Save the results
    try:
        dataset_dirmap.to_csv(args.dirmap_path, index=False)
        print(f"Successfully processed and saved to {args.dirmap_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()