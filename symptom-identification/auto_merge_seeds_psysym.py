import pandas as pd
import os
import argparse
import json

def merge_classification_reports(base_path):
    seeds = [42, 43, 44, 45, 46]
    seed_dirs = [f"seed_{i}" for i in seeds]
    all_dfs = []

    for idx, seed in enumerate(seed_dirs):
        csv_path = os.path.join(base_path, seed, "fold_0", "classification_report_test_transformed.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.insert(0, "seed", idx)
            df = df.drop(columns=["Unnamed: 0"], errors='ignore')  # Drop index column if it exists
            all_dfs.append(df)
        else:
            print(f"There is no file: {csv_path}")

    final_df = pd.concat(all_dfs, ignore_index=True)

    output_path = os.path.join(base_path, "seed_merged_classification_report.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Merging is completed: {output_path}")


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge classification reports from multiple folds.")
    parser.add_argument("--dir_path", type=str, required=True, help="Base directory containing fold directories.")
    
    print("======Running auto_merge_seeds.py======")
    args = parser.parse_args()
    
    if not os.path.exists(args.dir_path):
        print(f"The path does not exist: {args.dir_path}")
    else:
        merge_classification_reports(args.dir_path)
