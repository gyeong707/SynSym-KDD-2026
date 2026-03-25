import os
import re
import json
import pandas as pd
import argparse
from pathlib import Path


def load_symptom_description(load_dir):
    path = Path(load_dir) / 'description_dictionary.json'
    with open(path, 'r', encoding='utf-8') as f:
        description = json.load(f)
    return description


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
        print("Processing the result: ", idx)
        try:
            pred_parse = json.loads(res)
            json_result.append(pred_parse)
            print(res)
        except:
            try:
                res = re.sub(r'"\n', '",\n', res)
                res = re.sub(r',\n}', '\n}', res)
                pred_parse = json.loads(res)
                json_result.append(pred_parse)
            except:
                try:
                    pred_parse = res.split('```json\n')[1].split('\n```')[0].strip()
                    pred_parse = json.loads(pred_parse)
                    json_result.append(pred_parse)
                except:
                    print("Error in parsing the result.")
                    print(res)
                    json_result.append({})
                    error_case += 1
    return json_result, error_case


def make_df_result(data, symptom_dict):
    symp_list = []
    column_list = []
    content_list = []
    print(len(symptom_dict), len(data))
    if len(symptom_dict) == len(data):
        for key, value in zip(symptom_dict, data):
            columns = list(value.keys())
            for cols in columns:
                symp_list.append(key)
                column_list.append(cols)
                content_list.append(value[cols])

        df = pd.DataFrame({'symptom': symp_list, 'columns': column_list, 'explanations': content_list})
        return df
    else:
        print("The length of data and dictionary is not matched.")
        return 0


def save_json_dictionary(df, symptom_dict, output_dir, output_name, len_df):
    save_dic = {}
    columns = ['Descriptions']

    for symptom in symptom_dict:
        print("Symptom::", symptom)
        symptom_df = df[df['symptom'] == symptom]
        symptom_dic = {}
        for col in columns:
            target = symptom_df[symptom_df["columns"] == col]['explanations']
            if col == "Symptom":
                symptom_dic[col] = list(target)[0]
            else:
                symptom_dic[col] = list(target)
            save_dic[symptom] = symptom_dic
            print(save_dic[symptom])
    
    # make output dicrectory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + output_name + ".json", 'w') as f:
        json.dump(save_dic, f, indent=4)
    print("The dictionary is saved as ", output_dir + output_name + ".json")
    

def main(args):
    symptom_dict = load_symptom_description(args.load_dir)
    data, error_case = load_batch_result(args.input_dir, args.input_name)
    df = make_df_result(data, symptom_dict)
    save_json_dictionary(df, symptom_dict, args.output_dir, args.output_name, len(df))
    print("The number of error cases: ", error_case)


if __name__ == "__main__":
    print("\n\n\n::Start Process::\n\n\n")
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default="./input/")
    parser.add_argument('--input_dir', type=str, default="./output/")
    parser.add_argument('--input_name', type=str, default="batch_results_generating_description.jsonl")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--output_name', type=str, default="batch_results_generating_description_depression")
    args = parser.parse_args()

    # Start Process!
    main(args)
    print("\n\n\n::End Process::\n\n\n")