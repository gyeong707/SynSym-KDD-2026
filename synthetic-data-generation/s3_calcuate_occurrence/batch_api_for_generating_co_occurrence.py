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
Task:
You are a mental health expert specializing in depressive disorders. 
Your task is to generate 20 clinically plausible combinations of depressive symptoms based on the DSM-5 criteria for Major Depressive Disorder.
Each combination should include 2 to 5 symptoms that frequently co-occur in real-world patients.

For output, use the following symptom dictionary:
{
  "Anger_Irritability": 0,
  "Decreased_energy_tiredness_fatigue": 1,
  "Depressed_Mood": 2,
  "Genitourinary_symptoms": 3,
  "Hyperactivity_agitation": 4,
  "Inattention": 5,
  "Indecisiveness": 6,
  "Suicidal_ideas": 7,
  "Worthlessness_and_guilty": 8,
  "Loss_of_interest_or_motivation": 9,
  "Pessimism": 10,
  "Poor_memory": 11,
  "Sleep_disturbance": 12,
  "Weight_and_appetite_change": 13
}

Clinical Background Knowledge:
1. Depressed Mood is widely recognized as the most central and frequently occurring symptom of depression. It is commonly observed alongside a broad spectrum of other symptoms, and shows particularly strong associations with Irritability, Anhedonia, Worthlessness and Guilt, Sleep Disturbance, and Psychomotor Retardation.
2. Irritability, Anhedonia, Worthlessness and Guilt, Low Self-Esteem, Pessimism, and Fatigue are also frequently co-occurring symptoms that contribute to the core affective and cognitive profile of depressive episodes. These symptoms tend to exhibit high comorbidity with a range of other depressive features.
3. Sleep Disturbance is strongly linked with Fatigue and Weight/Appetite Change, forming a well-documented physiological cluster within depressive presentations. While it may co-occur with other symptoms, its presence is often more context-specific and typically reflects the somatic dimension of depression.
4. Suicidality commonly emerges in conjunction with symptoms such as Anhedonia, Pessimism, Depressed Mood, Low Self-Esteem, and Loss of Interest. Due to its clinical severity and context-dependent nature, it is expected to occur less frequently, predominantly within combinations representing more severe or high-risk depressive states.

Instructions:
1. Based on the clinical background knowledge above, generate 10 symptom combinations.
2. Each combination must include 2 to 5 symptom keys.
3. Ensure that the combinations reflect a diverse range of depression severity levels (e.g., mild, moderate, severe).
4. Return your output as a dictionary in the following format:

{{
  "combination_1": {{
    "symptoms": ["2", "8"],
    "severity": "moderate"
  }},
  "combination_2": {{
    "symptoms": ["2", "7", "9", "10"],
    "severity": "severe"
  }},
  ...
}}
"""


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

def make_batch_dataset(init_template, template, num_iter, input_dir, input_name):
    batches = []
    for iter in range(num_iter):
        print(f'Iteration: {iter}')
        temp = deepcopy(init_template)
        temp['custom_id'] = f'{iter}'
        temp['body']['messages'].append({"role": "system", "content": template})
        batches.append(temp)

    if path.exists(input_dir) == False:
        os.makedirs(input_dir)

    with open(input_dir + input_name, 'w') as file:
        for item in batches:
            json_string = json.dumps(item, ensure_ascii=False)
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


def save_result(client, client_id, output_dir, output_name):
    output_file_id = client.batches.retrieve(client_id).output_file_id
    result = client.files.content(output_file_id).content
    print(result)
    output_file_name = output_dir + output_name

    if path.exists(output_dir) == False:
        os.makedirs(output_dir)

    result_str = result.decode('utf-8')
    
    with open(output_file_name, 'w', encoding='utf-8') as file:
        file.write(result_str)

    results = []
    with open(output_file_name, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)
    return results


def main(args):
    init_template = initialize_template(args.model, args.temperature)
    make_batch_dataset(init_template, TEMPLATE, args.iteration, args.input_dir, args.input_name)
    client, client_id = call_batch_api(args.input_dir, args.input_name)

    while True:
        status = check_status(client, client_id)
        if status == "completed":
            print("Batch processing completed.")
            break
        elif status == "failed":
            print("Batch processing failed.")
        time.sleep(10)

    results = save_result(client, client_id, args.output_dir, args.output_name)

    # Print example results
    for res in results[:5]:
        print(f'id : {id}', res['custom_id'])
        result = res['response']['body']['choices'][0]['message']['content']        
        print(f'result : {result}\n')
    return results


if __name__ == "__main__":
    print("::Start Process::")

    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./output/")
    parser.add_argument('--input_name', type=str, default="batch_input.jsonl")
    parser.add_argument('--output_dir', type=str, default="./output/")
    parser.add_argument('--output_name', type=str, default="batch_results.jsonl")
    parser.add_argument('--iteration', type=int, default=20)
    parser.add_argument('--model', type=str, default="gpt-4o")
    parser.add_argument('--temperature', type=float, default=0.8)   
    args = parser.parse_args()

    # Start Process!
    main(args)
    print("::End Process::")

