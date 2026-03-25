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
TASK  
You are a mental health professional. Your task is to break down a given psychiatric symptom related to {mental_disorder} into a comprehensive list of *fine-grained sub-concepts*.  

Each symptom is initially defined by a high-level keyword and its corresponding clinical description or examples based on DSM-5.  
You must analyze the description and generate specific, realistic expressions that represent how individuals may manifest the symptom.  

INSTRUCTION  
1. Carefully examine the given symptom description.
2. Generate a list of detailed sub-concepts that reflect various real-world manifestations of the symptom.
   - Use clinical, direct language referring to the description.
   - Prioritize using the words and expressions from the given description.
   - Include all example expressions from the description, then add other relevant sub-concepts that capture different aspects of the symptom.
   - Ensure each sub-concept represents a distinct aspect or way the symptom can appear.
   - Generate at least {num_descriptions} sub-concepts.
3. Output the results in the following JSON format.

OUTPUT FORMAT  
{{
  "Symptom": "Anger_Irritability",
  "Descriptions": [
    "Exhibiting increased irritability",
    "Feeling angry",
    ...
  ]
}}
"""


def load_symptom_description(load_dir):
    path = Path(load_dir) / 'description_dictionary.json'
    with open(path, 'r', encoding='utf-8') as f:
        description = json.load(f)
    return description


def initialize_template(model):
    return {
    "custom_id": "custom",
    "method": "POST", 
    "url": "/v1/chat/completions",
    "body": {"model": model,
             "messages": [],
            "temperature": 0.0
             }
    }

def make_batch_dataset(descriptions, target_disease, init_template, template, output_dir, batch_name, num_descriptions):
    batches = []
    for id, symptom in enumerate(descriptions):
        symptom_descriptions = symptom + " " + str(descriptions[symptom])
        temp = deepcopy(init_template)
        temp['custom_id'] = f'{id}'
        filled_template = template.format(symptom=symptom, mental_disorder=target_disease, num_descriptions=num_descriptions)
        temp['body']['messages'].append({"role": "system", "content": filled_template})
        temp['body']['messages'].append({"role": "user", "content": symptom_descriptions})
        batches.append(temp)
    
    if path.exists(output_dir) == False:
        os.makedirs(output_dir)

    with open(Path(output_dir) / batch_name, 'w') as file:
        for item in batches:
            json_string = json.dumps(item)
            file.write(json_string + '\n')
        

def call_batch_api(output_dir, batch_name):
    client = OpenAI(api_key=API_KEY)
    batch_input_path = Path(output_dir) / batch_name
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


def check_status(client, client_id):
    status = client.batches.retrieve(client_id).status
    print("The status of the batch call is: ", status)
    return status


def save_result(client, client_id, output_dir, result_name):
    output_file_id = client.batches.retrieve(client_id).output_file_id
    result = client.files.content(output_file_id).content
    output_file_path = Path(output_dir) / result_name

    if path.exists(output_dir) == False:
        os.makedirs(output_dir)

    with open(output_file_path, 'wb') as file:
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

    # Load data
    description = load_symptom_description(args.load_dir)
    print("Number of symptom: ", len(description), list(description.keys()))

    # Initialize template
    init_template = initialize_template(args.model)
    make_batch_dataset(description, target_disease, init_template, TEMPLATE, args.output_dir, args.batch_name, args.num_descriptions)

    # Call batch-api
    client, client_id = call_batch_api(args.output_dir, args.batch_name)
    while True:
        status = check_status(client, client_id)
        if status == "completed":
            print("Batch processing completed.")
            break
        elif status == "failed":
            print("Batch processing failed.")
        time.sleep(30)
    results = save_result(client, client_id, args.output_dir, args.result_name)

    # Validation on 5 sample data
    for res in results[:5]:
        print(f'id : {id}', res['custom_id'])
        result = res['response']['body']['choices'][0]['message']['content']        
        print(f'result : {result}\n')


if __name__ == "__main__":
    print("::Start Process::")

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default="./input/")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--batch_name', type=str, default="batch_inputs_generating_description.jsonl")
    parser.add_argument('--result_name', type=str, default="batch_results_generating_description.jsonl")
    parser.add_argument('--model', type=str, default="gpt-4o")
    parser.add_argument('--target', type=str, default="depression")
    parser.add_argument('--num_descriptions', type=int, default=20)
    args = parser.parse_args()

    # Start Process!
    main(args)
    print("::End Process::")
