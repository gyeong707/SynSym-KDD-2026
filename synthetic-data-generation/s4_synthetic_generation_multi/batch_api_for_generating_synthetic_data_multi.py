# Synthetic Data Generation
import os
import json
import time
import pandas as pd
import argparse
from os import path
from pathlib import Path
from openai import OpenAI
from copy import deepcopy
import ast
import random
from dotenv import load_dotenv

# Load .env from synthetic-data-generation directory
_env_path = Path(__file__).resolve().parent
while _env_path.name != 'synthetic-data-generation' and _env_path.parent != _env_path:
    _env_path = _env_path.parent
load_dotenv(_env_path / '.env')

# OPEN AI - API KEY
API_KEY = os.getenv('OPENAI_API_KEY')


TEMPLATE = """
TASK:
You are a mental health professional. 
Given a specific symptom of {mental_disorder}, your task is to generate realistic statements that reflect the co-occurrence of multiple psychiatric symptoms.

INSTRUCTIONS:
1. You will be provided with a list of psychiatric symptoms and their corresponding clinical descriptions.
2. Generate a total of 10 first-person statements, with 5 for each of the following expression types:
   a. Clinical Style: Formal and objective statements that incorporate clinical terminology, resembling phrasing used in medical or diagnostic documentation.
   b. Colloquial Style: Informal, first-person descriptions that reflect how individuals express symptoms on social media platforms in a direct and explicit manner.

3. Follow these guidelines:
  a. Use first-person perspective (“I”).
  b. Each statement should reflect the co-occurrence of all the listed symptoms.
  c. Try to blend the symptoms together naturally within a shared emotional or situational context. 
  d. Vary the length, phrasing, and level of detail across the statements.
  e. Avoid metaphors, abstract language, or symbolic expressions.
  f. Only include expressions that directly reflect the given symptom. Do not include or imply any other symptoms in the generated statements.

4. Output the results in JSON format. Use the following structure:
{{
  "statement_1": {{
    "content": "I’ve been having suicidal ideation accompanied by a pervasive lack of interest in daily life.",
    "type": "clinical"
  }},
  "statement_2": {{
    "content": ""Sometimes I think about ending it all because I don’t care about anything anymore."",
    "type": "colloquial"
  }},
  ...
}}

IMPORTANT:  
The generated data will be used to detect psychiatric symptoms in real social media users. Some content may include sensitive or intense expressions, but DO NOT AVOID GENERATING THEM. Remember that if you avoid realistic expressions, models learning from this data may miss important indicators of mental disorders.
"""


def load_symptom_description(load_dir):
    with open(load_dir + 'batch_results_generating_description_depression.json', 'r') as f:
        description = json.load(f)
    return description

def load_combination(comb_data):
    df = pd.read_csv(comb_data)
    df['symptoms'] = df['symptoms'].apply(lambda x: ast.literal_eval(x))
    return df['symptoms'].tolist()

def initialize_template(model, temperature):
    return {
    "custom_id": "custom",
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": model,
             "messages": [],
            "temperature": temperature
             }
    }

def make_symptom_description(description, target_combination):
    target_dict = {}
    for target in target_combination:
        target_dict[target] = description[target]
    description = "\n".join([f"{k}: {v}" for k, v in target_dict.items()])
    return description


def make_batch_dataset(desc_list, target_disease, num_iter, init_template, template, input_dir, input_data):
    batches = []
    id = 0
    for desc_id, target_desc in enumerate(desc_list): # description
        print(f'Target description: {target_desc} (number: {desc_id})')
        for _ in range(num_iter):
            print(f'Iteration: {id}')
            temp = deepcopy(init_template)
            temp['custom_id'] = f'{id}'
            filled_template = template.format(mental_disorder=target_disease)
            temp['body']['messages'].append({"role": "system", "content": filled_template})
            temp['body']['messages'].append({"role": "user", "content": target_desc})
            batches.append(temp)
            id += 1
    
    if path.exists(input_dir) == False:
        os.makedirs(input_dir)

    with open(input_dir + input_data, 'w') as file:
        for item in batches:
            json_string = json.dumps(item)
            file.write(json_string + '\n')
        

def call_batch_api(input_dir, input_data):
    client = OpenAI(api_key=API_KEY)
    batch_input_file = client.files.create(
        file=open(input_dir + input_data,  "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    batch_call = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "Batch call for mgkang"
        }
    )
    client_id = batch_call.id
    return client, client_id


def check_status(client, client_id):
    status = client.batches.retrieve(client_id).status
    print("The status of the batch call is: ", status)
    return status


def save_result(client, client_id, output_dir, output_data):
    output_file_id = client.batches.retrieve(client_id).output_file_id
    result = client.files.content(output_file_id).content
    output_file_name = output_dir + output_data

    if path.exists(output_dir) == False:
        os.makedirs(output_dir)

    with open(output_file_name, 'wb') as file:
        file.write(result)

    results = []
    return results


def mapping_target_disease(target):
    if target == "depression": return "Depressive Disorders"
    elif target == "anxiety": return "Anxiety Disorders"
    elif target == "eating": return "Eating Disorders"
    elif target == "bipolar": return "Bipolar Disorders"
    elif target == 'ocd': return "Obsessive-Compulsive Disorders"
    elif target == 'ptsd': return "Post-traumatic Stress Disorder"
    elif target == 'adhd': return "Attention Deficit Hyperactivity Disorder"


def main(args):
    target_disease = mapping_target_disease(args.target)
    print("Target diseaes: ", target_disease)

    description = load_symptom_description(args.load_dir)
    combination = load_combination(args.comb_data)

    target_setnences = []
    for row in combination:
        target_sentence = ""
        # random shuffle
        random.shuffle(row)
        for symptom in row:
            target_sentence += symptom + ": " + str(description[symptom]) + "\n"
        print(">>", target_sentence)
        target_setnences.append(target_sentence)
    
    init_template = initialize_template(args.model, args.temperature)
  
    make_batch_dataset(target_setnences, target_disease, args.num_iter, init_template, TEMPLATE, args.input_dir, args.input_data)
    client, client_id = call_batch_api(args.input_dir, args.input_data)
    while True:
        status = check_status(client, client_id)
        if status == "completed":
            print("Batch processing completed.")
            break
        elif status == "failed":
            print("Batch processing failed.")
        time.sleep(1)

    results = save_result(client, client_id, args.output_dir, args.output_data)
    return results



if __name__ == "__main__":
    print("::Start Process::")

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default="./input/")
    parser.add_argument('--comb_data', type=str, default="./input/batch_results_590.csv")
    parser.add_argument('--input_dir', type=str, default="./output/")
    parser.add_argument('--input_data', type=str, default="batch_inputs_sdg_multi_depression.jsonl")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--output_data', type=str, default="batch_results_sdg_multi_depression.jsonl")
    parser.add_argument('--model', type=str, default="gpt-4o")
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--target', type=str, default="depression")
    args = parser.parse_args()

    # Start Process!
    main(args)
    print("::End Process::")
