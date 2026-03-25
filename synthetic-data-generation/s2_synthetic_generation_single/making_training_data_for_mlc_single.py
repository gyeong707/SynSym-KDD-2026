"""Build MLC train/val CSVs from synthetic data + symptom_dictionary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_symptom_dictionary(symptom_dict_path: Union[str, Path]) -> dict:
    """Load symptom name -> class index mapping from a JSON file."""
    path = Path(symptom_dict_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mapping_symptom_label(symptom: str, symptom_to_idx: dict) -> list:
    """One-hot vector for a single symptom label."""
    if symptom not in symptom_to_idx:
        raise ValueError(f"Symptom {symptom!r} not found in symptom_dictionary.")
    symptom_vector = np.zeros(len(symptom_to_idx), dtype=int)
    symptom_vector[symptom_to_idx[symptom]] = 1
    return list(symptom_vector)


def split_mlc_data(ent_df: pd.DataFrame, train_size: float, random_state: int):
    return train_test_split(
        ent_df,
        train_size=train_size,
        random_state=random_state,
        stratify=ent_df["symptom"],
    )


def save_mlc_df(
    output_dir: Union[str, Path],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_tag: str,
    output_prefix: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / f"{output_prefix}_train_{len(train_df)}_{output_tag}.csv"
    val_path = out / f"{output_prefix}_val_{len(val_df)}_{output_tag}.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print("Saved:", train_path, val_path)


def main(args):
    symp_dict = load_symptom_dictionary(Path(args.load_dir) / "symptom_dictionary.json")

    synth_path = Path(args.input_dir) / args.synth
    synth_df = pd.read_csv(synth_path)

    merged_df = synth_df.copy()
    merged_df["label"] = merged_df["symptom"].apply(
        lambda s: mapping_symptom_label(s, symp_dict)
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged_path = out / f"batch_sdg_df_merged_added_label_{len(merged_df)}.csv"
    merged_df.to_csv(merged_path, index=True)
    print("Merged:", merged_path)
    print(merged_df["symptom"].value_counts())

    train_df, val_df = split_mlc_data(
        merged_df,
        train_size=args.train_size,
        random_state=args.random_state,
    )
    save_mlc_df(
        args.output_dir,
        train_df,
        val_df,
        args.output_tag,
        args.output_prefix,
    )


if __name__ == "__main__":
    print("::Start Process::")

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, default="./input/")
    parser.add_argument("--input_dir", type=str, default="./output/")
    parser.add_argument("--synth", type=str, default="batch_results_generating_synthetic_data_single_formatted.csv")
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--output_tag", type=str, default="single")
    parser.add_argument("--output_prefix", type=str, default="synth")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    main(args)
    print("::End Process::")
