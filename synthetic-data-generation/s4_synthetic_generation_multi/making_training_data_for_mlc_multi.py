import pandas as pd
import numpy as np
import argparse
import re
import os
import json
import ast
from sklearn.model_selection import train_test_split


def load_symptom_dictionary(load_dir):
    with open(load_dir + 'batch_results_generating_description_depression.json', 'r') as f:
    # with open(load_dir + 'description_dictionary.json', 'r') as f:
        desc_dict = json.load(f)
    with open(load_dir + "symptom_dictionary.json", 'r') as f:
        symp_dict = json.load(f)
    return desc_dict, symp_dict

    
def mapping_symptom_label(row, symptom_to_idx):
    symptom_vector = np.zeros(len(symptom_to_idx), dtype=int)
    for r in row:
        if r in symptom_to_idx:
            idx = symptom_to_idx[r]
            symptom_vector[idx] = 1
        elif r not in symptom_to_idx:
            raise (f"Symptom {r} not found in the dictionary.")
        # if row in symptom_to_idx:
        #     idx = symptom_to_idx[row]
        #     symptom_vector[idx] = 1
        # elif row not in symptom_to_idx:
        #     raise (f"Symptom {row} not found in the dictionary.")
    return list(symptom_vector)


def make_description_column(row, symptom_dict):
    if row in symptom_dict:
        desc = ' '.join([item for items in symptom_dict[row]['Descriptions'] for item in items])
        # desc = ' '.join([item for items in symptom_dict[row] for item in items])
    else:
        print(f"Symptom {row} not found in the dictionary.")
    return desc

def split_nli_data(merged_df, train_size=0.8, val_size=0.2, random_state=42):
    train_df, val_df = train_test_split(
        merged_df,
        train_size=train_size,
        stratify=merged_df['label'], # stratified sampling
        random_state=random_state
    )
    
    # Print statistics
    print("\nData split statistics:")
    print(f"Total samples: {len(merged_df)}")
    print(f"Train samples: {len(train_df)} ({len(train_df)/len(merged_df):.2%})")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(merged_df):.2%})")
    
    # Print label distributions
    print("\nLabel distributions:")
    print("\nTrain set:")
    print(train_df['label'].value_counts(normalize=True))
    print("\nValidation set:")
    print(val_df['label'].value_counts(normalize=True))
    
    return train_df, val_df

def split_mlc_data(mlc_df, train_size=0.8, val_size=0.2, random_state=42):
    train_df, val_df = train_test_split(
        mlc_df,
        train_size=train_size,
        stratify=mlc_df['label'],  # stratified sampling
        random_state=random_state
    )
    return train_df, val_df


def save_mlc_df(output_dir, train_df, val_df):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_path = os.path.join(output_dir, "synth_train_" + str(len(train_df)) + "_multi.csv")
    val_path = os.path.join(output_dir, "synth_val_" + str(len(val_df)) + "_multi.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print("Saved train and validation dataframes to CSV files:", train_path, val_path)
    

def main(args):
    # Load dictionaries
    desc_dict, symp_dict = load_symptom_dictionary(args.load_dir)

    # Synthetic posts from multi-symptom generation (e.g. batch_api_for_generating_synthetic_data_multi_formatting.py)
    synth_path = os.path.join(args.input_dir, args.synthetic_data)
    synthetic_df = pd.read_csv(synth_path)

    mlc_df = synthetic_df.copy()
    mlc_df['symptom'] = mlc_df['symptom'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    mlc_df['label'] = mlc_df.apply(lambda x: mapping_symptom_label(x['symptom'], symp_dict), axis=1)

    merged_df = mlc_df.copy()

    merged_path = os.path.join(
        args.output_dir,
        'batch_sdg_df_merged_added_label_' + str(len(merged_df)) + '.csv',
    )
    merged_df.to_csv(merged_path, index=True)
    print(merged_path)
    print(merged_df['label'].value_counts())

    train_df, val_df = split_mlc_data(merged_df)
    save_mlc_df(args.output_dir, train_df, val_df)


if __name__ == '__main__':
    print("\n\n\n::Start Process::\n\n\n")

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default="./input/")
    parser.add_argument('--input_dir', type=str, default="./output/")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--synthetic_data', type=str, default="batch_results_sdg_multi_depression_5900.csv")
    parser.add_argument('--target', type=str, default="depression")
    args = parser.parse_args()

    # Start Process
    main(args)
    print("\n\n\n::End Process::\n\n\n")



