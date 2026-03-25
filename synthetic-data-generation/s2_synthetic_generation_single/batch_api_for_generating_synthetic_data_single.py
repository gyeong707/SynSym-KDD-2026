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
Given a specific symptom of {mental_disorder}, your task is to generate realistic statements that reflect this psychological struggle or symptom.

INSTRUCTIONS:
1. For the given symptom, identify clinical terminologies or expressions that would indicate the presence of that symptom.

2. Generate a total of 10 first-person statements, with 5 for each of the following expression types:
   a. Clinical Style: Formal and objective statements that incorporate clinical terminology, resembling phrasing used in medical or diagnostic documentation.
   b. Colloquial Style: Informal, first-person descriptions that reflect how individuals express symptoms on social media platforms in a direct and explicit manner.

3. Follow these guidelines:
  a. Use first-person perspective (“I”).
  b. Avoid metaphors, abstract language, or symbolic expressions.
  c. Each statement should be 1–2 sentences.
  d. Only include expressions that directly reflect the given symptom. Do not include or imply any other symptoms in the generated statements.

3. Output the results in JSON format. Follow this structure:
{{
  "statement_1": {{
    "content": "generated_statement_1",
    "type": "clinical"
  }},
  "statement_2": {{
    "content": "generated_statement_2",
    "type": "colloquial"
  }},
  ...
}}

IMPORTANT:  
The generated data will be used to detect psychiatric symptoms in real social media users. 
Some content may include sensitive or intense expressions, but DO NOT AVOID GENERATING THEM. Remember that if you avoid realistic expressions, models learning from this data may miss important indicators of mental disorders.
"""


def load_symptom_description(load_dir, target):
    filename = f'batch_results_generating_description_{target}.json'
    filepath = Path(load_dir) / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        description = json.load(f)
    return description


def initialize_template(model):
    return {
    "custom_id": "custom",
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": model,
             "messages": [],
            "temperature": 0.8
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
    
    input_path = Path(input_dir) / input_data
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'w', encoding='utf-8') as file:
        for item in batches:
            json_string = json.dumps(item)
            file.write(json_string + '\n')
        

def call_batch_api(input_dir, input_data):
    client = OpenAI(api_key=API_KEY)
    batch_input_path = Path(input_dir) / input_data
    batch_input_file = client.files.create(
        file=open(batch_input_path, "rb"),
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


def check_status(client, client_id, current=None, total=None):
    status = client.batches.retrieve(client_id).status
    prog = f" [{current}/{total}]" if current is not None and total is not None else ""
    print(f"Status{prog}: {status}")
    return status


def save_result(client, client_id, output_dir, output_data, symptom_id):
    output_file_id = client.batches.retrieve(client_id).output_file_id
    result = client.files.content(output_file_id).content
    output_path = Path(output_dir) / output_data
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if symptom_id == 0:
        with open(output_path, 'wb') as file:
            file.write(result)
    else:
        with open(output_path, 'ab') as file:
            file.write(result)
    
    results = []
    return results


def mapping_target_disease(target):
    if target == "depression": return "Depressive Disorders"
    elif target == "anxiety": return "Anxiety Disorders"
    elif target == "eating": return "Eating Disorders"
    elif target == "bipolar": return "Bipolar Disorders"
    elif target == 'ocd': return "Obsessive-Compulsive Disorders"
    elif target == 'ptsd': return "Post-traumatic stress disorder"
    elif target == 'adhd': return "Attention Deficit Hyperactivity Disorder"



def main(args):
    target_disease = mapping_target_disease(args.target)
    print("Target diseaes: ", target_disease)

    description = load_symptom_description(args.load_dir, args.target)
    init_template = initialize_template(args.model)

    symptom_list = list(description.keys())
    print(f"Total: {len(symptom_list)} symptoms, {args.num_iter} iterations per description")
    total_symptoms = len(symptom_list)
    for symptom_id, target_symptom in enumerate(symptom_list): # symptom
        desc_list = []
        print(f'[{symptom_id + 1}/{total_symptoms}] Target symptom: {target_symptom}')
        desc_list.append(target_symptom)
        for desc in description[target_symptom]['Descriptions'][0]:
            desc_list.append(target_symptom + ": " + desc)
        print(desc_list)

        make_batch_dataset(desc_list, target_disease, args.num_iter, init_template, TEMPLATE, args.input_dir, args.input_data)
        client, client_id = call_batch_api(args.input_dir, args.input_data)
        while True:
            status = check_status(client, client_id, symptom_id + 1, total_symptoms)
            if status == "completed":
                print(f"[{symptom_id + 1}/{total_symptoms}] Batch completed for {target_symptom}.")
                break
            elif status == "failed":
                print(f"[{symptom_id + 1}/{total_symptoms}] Batch FAILED for {target_symptom}.")
                break
            time.sleep(30)
        results = save_result(client, client_id, args.output_dir, args.output_data, symptom_id)

    return results



if __name__ == "__main__":
    print("::Start Process::")

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default="input/")
    parser.add_argument('--input_dir', type=str, default="output/")
    parser.add_argument('--input_data', type=str, default="batch_results_generating_description_depression.jsonl")
    parser.add_argument('--output_dir', type=str, default="output/")
    parser.add_argument('--output_data', type=str, default="batch_results_generating_synthetic_data_single.jsonl")
    parser.add_argument('--model', type=str, default="gpt-4o")
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--target', type=str, default="depression")
    args = parser.parse_args()

    # Start Process!
    main(args)
    print("::End Process::")
