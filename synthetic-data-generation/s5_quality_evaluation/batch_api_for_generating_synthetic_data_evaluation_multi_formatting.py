import os
import re
import json
import time
import pandas as pd
import argparse
from os import path
from openai import OpenAI
from copy import deepcopy
import ast


def load_entailment_dataset(load_dir, load_data):
    df = pd.read_csv(load_dir + load_data)
    print("The number of data: ", len(df))
    return df

def fix_json_string(json_str): # Exception handling
    fixed_str = re.sub(r'",\s*}', '"\n}', json_str)
    fixed_str = re.sub(r'}\s*{', '}, {', fixed_str)
    fixed_str = fixed_str.replace('"', '"').replace('"', '"')
    fixed_str = fixed_str.replace(",", ",")
    return fixed_str

def load_batch_result(input_dir, input_data):
    with open(input_dir + input_data, 'r') as f:
        results = [json.loads(line) for line in f]

    pred_result = []
    for res in results:
        temp = res['response']['body']['choices'][0]['message']['content']
        pred_result.append(temp)

    error_case = 0
    json_result = []
    for idx, res in enumerate(pred_result):
        # print("Processing the result: ", idx)
        try:
            pred_parse = json.loads(res)
            json_result.append(pred_parse)
        except:
            try:
                res = re.sub(r'"\n', '",\n', res)
                res = re.sub(r',\n}', '\n}', res)
                pred_parse = json.loads(res)
                json_result.append(pred_parse)
            except:
                try:
                    pred_parse = res.split('```json\n')[1].split('\n```')[0].strip()
                    pred_parse = re.sub(r'("\s*:\s*".*?)",\s*}', r'\1"\n}', pred_parse)
                    pred_parse = json.loads(pred_parse)
                    json_result.append(pred_parse)
                except Exception as e:
                    print("Error in parsing the result.")
                    print(res)
                    print(e)
                    json_result.append({})
                    error_case += 1

    print("The number of error cases: ", error_case)

    # Save file
    output_json_path = input_dir + 'parsed_results.json'
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

    print("len(json_result): ", len(json_result))
    return json_result, error_case


def make_df_result(data):
    print(len(data))
    post_res, score_res, reason_res = [], [], []
    for iter in zip(data):
        if iter == {}:
            print("Empty.")
            post_res.append(None)
            score_res.append(None)
            reason_res.append(None)
        else:
            scores = []
            reasons = []
            post_res.append(iter[0]['post'])
            for row in iter[0]['symptom_scores']:
                try:
                    scores.append(row['score'])
                    reasons.append(row['reason'])
                except:
                    scores.append(-1)
                    reasons.append("Error")
            score_res.append(scores)
            reason_res.append(reasons)

    print(len(post_res), len(score_res), len(reason_res))
    df = pd.DataFrame({'post': post_res, 'score': score_res, 'reason': reason_res})
    return df

def save_df_result(df, output_dir, output_name, len_df):
    if not os.path.exists(output_dir):
        print("The output directory is not existed.")
        os.makedirs(output_dir)
    df.to_csv(output_dir + output_name + str(len_df) + ".csv", index=False)
    print("The result is saved as ", output_dir + output_name + str(len_df) + ".csv")


def main(args):
    entailment_df = load_entailment_dataset(args.load_dir, args.load_data)
    result, error_case = load_batch_result(args.input_dir, args.input_data)
    df = make_df_result(result)
    save_df_result(df, args.output_dir, args.output_data, len(df))
    print("The number of error cases: ", error_case)


if __name__ == "__main__":
    print("\n\n\n::Start Process::\n\n\n")

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default="./input/")
    parser.add_argument('--load_data', type=str, default="batch_sdg_df_merged_added_label_with_knowledge_5900.csv")
    parser.add_argument('--input_dir', type=str, default="./output/")
    parser.add_argument('--input_data', type=str, default="batch_results_sdg_evaluation_multi.jsonl")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--output_data', type=str, default="batch_sdg_df_evaluation_multi_")
    args = parser.parse_args()

    # Start Process!
    main(args)
    print("\n\n\n::End Process::\n\n\n")

