#!/usr/bin/env python3
import sys
import pandas as pd
import os

def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_pass_csv> <output_result_csv>")
        sys.exit(1)

    # Get file paths from command-line arguments
    input_pass_csv = sys.argv[1]  # e.g., /path/to/pass.csv
    output_result_csv = sys.argv[2]  # e.g., /path/to/result.csv

    # Check if input file exists
    if not os.path.exists(input_pass_csv):
        print(f"Error: Input file {input_pass_csv} does not exist")
        sys.exit(1)

    # Read the input pass.csv
    try:
        df = pd.read_csv(input_pass_csv)
    except Exception as e:
        print(f"Error reading {input_pass_csv}: {e}")
        sys.exit(1)

    # If output file exists, use its columns; otherwise, use input file's columns
    try:
        if os.path.exists(output_result_csv):
            # Read existing result.csv to get columns and check for duplicates
            existing_df = pd.read_csv(output_result_csv)
            expected_columns = existing_df.columns.tolist()  # Use columns from existing output

            # Check if input CSV has the required columns
            if not all(col in df.columns for col in expected_columns):
                print(f"Error: {input_pass_csv} missing required columns: {expected_columns}")
                sys.exit(1)

            # Select only the expected columns from input
            df = df[expected_columns]

            # Check for duplicates based on model_path and dataset
            if "model_path" in df.columns and "dataset" in df.columns:
                if not existing_df[
                    (existing_df["model_path"] == df["model_path"].iloc[0]) &
                    (existing_df["dataset"] == df["dataset"].iloc[0])
                ].empty:
                    print(f"Warning: Entry for {df['model_path'].iloc[0]} and {df['dataset'].iloc[0]} already exists in {output_result_csv}, skipping...")
                    return

            # Append without writing header
            df.to_csv(output_result_csv, mode="a", header=False, index=False)
        else:
            # Use input file's columns for the first write
            df.to_csv(output_result_csv, mode="w", header=True, index=False)

        print(f"Successfully appended {input_pass_csv} to {output_result_csv}")
    except Exception as e:
        print(f"Error writing to {output_result_csv}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()