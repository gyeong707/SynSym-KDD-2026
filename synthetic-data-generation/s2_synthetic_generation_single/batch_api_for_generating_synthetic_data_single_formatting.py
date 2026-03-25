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


def load_symptom_description(load_dir):
    with open(load_dir + 'batch_results_generating_description_depression.json', 'r') as f:
        description = json.load(f)
    return description


def fix_json_string(json_str): # Exception handling
    fixed_str = re.sub(r'",\s*}', '"\n}', json_str)
    fixed_str = re.sub(r'}\s*{', '}, {', fixed_str)
    fixed_str = fixed_str.replace('"', '"').replace('"', '"')
    fixed_str = fixed_str.replace(",", ",")

    return fixed_str

def make_symptom_description(symptom_dict, symptom_names, num_iter):
    symp_list =[] 
    desc_list = []
    print(len(symptom_names), type(symptom_names), symptom_names[0])
    for target_symptom in symptom_names: # symptom name
        for _ in range(num_iter): # num_iter
            symp_list.append(target_symptom)
            desc_list.append("None")
        for desc in symptom_dict[target_symptom]['Descriptions'][0]:
            for _ in range(num_iter): # num_iter 
                symp_list.append(target_symptom)
                desc_list.append(desc)
    print(len(symp_list), len(desc_list))
    return symp_list, desc_list

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
                    res = res.split('```json\n')[1].split('\n```')[0].strip()
                    pred_parse = json.loads(res)
                    json_result.append(pred_parse)
                except:
                    try:
                        res = re.sub(r'("type"\s*:\s*"(clinical|colloquial)")\s*,', r'\1', res)
                        pred_parse = json.loads(res)
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

    return json_result, error_case


def make_df_result(data, symp_dict, desc_dict):
    print(len(data), len(symp_dict), len(desc_dict))
    post_list, symp_list, desc_list, type_list = [], [], [], []
    error_counts = 0
    for iter, symp, desc in zip(data, symp_dict, desc_dict):
        if iter == {}:
            # print("Empty.")
            post_list.append(None)
            type_list.append(None)
            symp_list.append(symp)
            desc_list.append(desc)
            error_counts += 1
        else:
            for key in iter.keys():
                try: 
                    post_list.append(iter[key]['content'])
                    type_list.append(iter[key]['type'])
                    symp_list.append(symp)
                    desc_list.append(desc)
                except:
                    print("Error in parsing the result.")
                    post_list.append(None)
                    type_list.append(None)
                    symp_list.append(symp)
                    desc_list.append(desc)
                    error_counts += 1

    print("The number of error cases: ", error_counts)
    df = pd.DataFrame({'post': post_list, 'type': type_list, 'symptom': symp_list, 'description': desc_list})
    # null 값 개수 확인
    print(df.isnull().sum())
    # null 값 제거
    df = df.dropna()
    return df

def save_df_result(df, output_dir, output_name, len_df):
    if not os.path.exists(output_dir):
        print("The output directory is not existed.")
        os.makedirs(output_dir)
    df.to_csv(output_dir + output_name, index=False)
    print("The result is saved as ", output_dir + output_name)


def main(args):
    description = load_symptom_description(args.load_dir)
    symptom_names = list(description.keys())
    symp_list, desc_list = make_symptom_description(description, symptom_names, args.num_iter)
    result, error_case = load_batch_result(args.input_dir, args.input_data)
    df = make_df_result(result, symp_list, desc_list)
    save_df_result(df, args.output_dir, args.output_data, len(df))
    print("The number of error cases: ", error_case)


if __name__ == "__main__":
    print("\n\n\n::Start Process::\n\n\n")

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default="./input/")
    parser.add_argument('--input_dir', type=str, default="./output/")
    parser.add_argument('--input_data', type=str, default="batch_results_generating_synthetic_data_single.jsonl")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--output_data', type=str, default="batch_results_generating_synthetic_data_single_formatted.csv")
    parser.add_argument('--num_iter', type=int, default=1)
    args = parser.parse_args()

    # Start Process!
    main(args)
    print("\n\n\n::End Process::\n\n\n")

