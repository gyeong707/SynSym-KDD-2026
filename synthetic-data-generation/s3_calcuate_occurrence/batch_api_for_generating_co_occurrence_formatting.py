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


def load_symptom_dictionary(load_dir, reverse=False):
    with open(load_dir, 'r') as f:
        dictionary = json.load(f)
    if reverse == True:
        dictionary = {v: k for k, v in dictionary.items()}
    return dictionary

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
                res = re.sub(r)
                pred_parse = json.loads(res)
                json_result.append(pred_parse)
            except:
                try:
                    pred_parse = res.split('```json\n')[1].split('\n```')[0].strip()
                    pred_parse = re.sub(r'("severity"\s*:\s*"(?:mild|moderate|severe)"\s*),', r'\1', pred_parse)
                    pred_parse = json.loads(pred_parse)
                    json_result.append(pred_parse)
                except Exception as e:
                    print("Error in parsing the result.")
                    print("Error Case: ", res)
                    print(e)
                    json_result.append({})
                    error_case += 1


    print("The number of error cases: ", error_case)

    # Save file
    output_json_path = input_dir + 'parsed_results.json'
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

    return json_result, error_case


def make_df_result(data, dictionary):
    # print("Making the DataFrame result...")
    comb_list = []
    error_counts = 0
    for iter in data:
        print("ITERATION: ", iter)
        for row in iter:
            comb_list.append(iter[row]['symptoms'])

    comb_list_new = []
    print(len(comb_list))
    for comb in comb_list:
        temp = []
        # print("HERE", comb)
        for element in comb:
            # print(dictionary[int(element)])
            try:
                temp.append(dictionary[int(element)])
            except:
                break
        comb_list_new.append(temp)

    print("The number of error cases: ", error_counts)
    df = pd.DataFrame({
        'symptoms': comb_list_new,
    })
    print(df.isnull().sum())
    # null 값 제거
    df = df.dropna()
    return df

def save_df_result(df, output_dir, output_name, len_df):
    if not os.path.exists(output_dir):
        print("The output directory is not existed.")
        os.makedirs(output_dir)
    df.to_csv(output_dir + output_name + str(len_df) + ".csv", index=False)
    print("The result is saved as ", output_dir + output_name + str(len_df) + ".csv")


def main(args):
    dictionary = load_symptom_dictionary(args.load_dir, reverse=True)
    result, error_case = load_batch_result(args.input_dir, args.input_data)
    df = make_df_result(result, dictionary)
    save_df_result(df, args.output_dir, args.output_data, len(df))
    print("The number of error cases: ", error_case)


if __name__ == "__main__":
    print("\n\n\n::Start Process::\n\n\n")

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default="./input/symptom_dictionary.json")
    parser.add_argument('--input_dir', type=str, default="./output/")
    parser.add_argument('--input_data', type=str, default="batch_results.jsonl")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--output_data', type=str, default="batch_results_")
    parser.add_argument('--num_iter', type=int, default=20)
    args = parser.parse_args()

    # Start Process!
    main(args)
    print("\n\n\n::End Process::\n\n\n")

