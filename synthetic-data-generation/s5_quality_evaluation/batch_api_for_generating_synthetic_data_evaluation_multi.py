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
You are a mental health professional evaluating how well a social media post expresses the given symptoms of {mental_disorder}.  
For each symptom provided, rate how clearly the post reflects that symptom on a scale from 0 to 5.

INPUT  
Post: {post}  
Symptoms:  
{description}

RATING CRITERIA  
- 0: Not interpretable as the symptom at all  
- 1: Very vaguely might be related to the symptom  
- 2: Somewhat suggests the symptom, but unclear  
- 3: Moderately clear expression of the symptom  
- 4: Clear expression of the symptom  
- 5: Very clear and strong expression of the symptom  

OUTPUT FORMAT  
Provide your evaluation in the following JSON format.  
Each symptom must be evaluated individually. The output should be a single JSON object with the post and a list of symptom scores:

{{
  "post": "{post}",
  "symptom_scores": [
    {{
      "symptom": "Symptom Name",
      "score": int,  # 0-5
      "reason": "Explanation of why this score was given"
    }},
    {{
      "symptom": "Symptom Name",
      "score": int,  # 0-5
      "reason": "Explanation of why this score was given"
    }}
    ...
  ]
}}
NOTE: The evaluated data will be used to detect symptoms related to {mental_disorder} in real social media users. Some content may include sensitive or intense expressions, but rate them solely based on how well they express the target symptom.
"""


def load_entailment_dataset(load_dir, load_data):
    df = pd.read_csv(load_dir + load_data)
    return df


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


def make_batch_dataset(entailment_df, target_disease, init_template, template, input_dir, input_data):
    batches = []

    for id in range(len(entailment_df)):
        target_post = entailment_df['post'][id]
        target_desc = entailment_df['symptom'][id]
        temp = deepcopy(init_template)
        temp['custom_id'] = f'{id}'
        filled_template = template.format(post=target_post, description=target_desc, mental_disorder=target_disease)
        temp['body']['messages'].append({"role": "system", "content": filled_template})
        batches.append(temp)
    
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
    elif target == 'ptsd': return "Post-traumatic stress disorder"
    elif target == 'adhd': return "Attention Deficit Hyperactivity Disorder"


def main(args):
    target_disease = mapping_target_disease(args.target)
    print("Target diseaes: ", target_disease)

    # load data
    entailment_df = load_entailment_dataset(args.load_dir, args.load_data)
    init_template = initialize_template(args.model)

    make_batch_dataset(entailment_df, target_disease, init_template, TEMPLATE, args.input_dir, args.input_data)
    client, client_id = call_batch_api(args.input_dir, args.input_data)
    while True:
        status = check_status(client, client_id)
        if status == "completed":
            print("Batch processing completed.")
            break
        elif status == "failed":
            print("Batch processing failed.")
        time.sleep(10)
    results = save_result(client, client_id, args.output_dir, args.output_data)

    return results



if __name__ == "__main__":
    print("::Start Process::")

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, default="./input/")
    parser.add_argument('--load_data', type=str, default="batch_sdg_df_merged_added_label_with_knowledge_5900.csv")
    parser.add_argument('--input_dir', type=str, default="./output/")
    parser.add_argument('--input_data', type=str, default="batch_results_sdg_evaluation_multi.jsonl")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--output_data', type=str, default="batch_results_sdg_evaluation_multi.jsonl")
    parser.add_argument('--model', type=str, default="gpt-4o")
    parser.add_argument('--target', type=str, default="depression")
    args = parser.parse_args()

    # Start Process!
    main(args)
    print("::End Process::")
