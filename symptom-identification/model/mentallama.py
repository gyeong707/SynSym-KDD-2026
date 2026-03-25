from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report
from transformers import LlamaTokenizer, LlamaForCausalLM,BitsAndBytesConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import trange

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, XLNetModel,XLNetForSequenceClassification, AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import IntervalStrategy

from transformers import LlamaTokenizer, LlamaForSequenceClassification
from transformers import pipeline

import warnings
warnings.filterwarnings(action='ignore')

import gc
from tqdm import tqdm
import random
import numpy as np
import os
import re

from pathlib import Path
from pprint import pprint
import os
import glob
import ast
import json
import time
import deepl

def create_label_index(data, label_cols, num_labels):
    data[label_cols] = data[label_cols].map(lambda x: ast.literal_eval(x))
    labels = torch.zeros(len(data), num_labels)
    for i in range(len(data)):
        target = data[label_cols][i]
        if target == []: pass
        else:
            for t in target:
                labels[i][t] = 1
    return labels

def get_mentallama_labels(decoded):
        decoded = decoded.lower() # 소문자화
        print(decoded) # 디코딩한 날것의 출력 보기
        labels = {
            "depression": None,
            "anxiety": None,
            "sleep": None,
            "eating": None,
            "non-disease": None
        }

        for label in labels:
            posA = decoded.find(f"{label}")
            if (posA != -1):
                labels[label] = 1
            else:
                labels[label] = 0

        return labels
# def get_mentallama_labels(decoded):
#     decoded = decoded.lower() # 소문자화
#     print(decoded) # 디코딩한 날것의 출력 보기
#     labels = [0, 0, 0, 0, 0]  # 리스트로 초기화

#     label_names = ["depression", "anxiety", "sleep", "eating", "non-disease"]

#     for i, label in enumerate(label_names):
#         posA = decoded.find(label)
#         if posA != -1:
#             labels[i] = 1

#     return labels

def multilabel_stratified_split(data, label, seed, fold, n_splits=5, shuffle=True, column_name='pre_question'):
    X = data[column_name].values
    y = label
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for i, (train_index, test_index) in enumerate(mskf.split(X, y)):
        print(i, fold)
        if fold == i:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            return X_train, X_test, y_train, y_test, train_index, test_index
        
def tensor_to_list(y_test_tensor):
    y_test_list = y_test_tensor.int().tolist()
    return y_test_list

def main(token_model, fold, label_cols, SEED, num_labels):
    print(" :: 1. Load Data :: ")
    file = '/home/dsail/migyeongk/dsail/projects/Mental-Disorder-Detection/emnlp_2024/disease_detection/data/ours/240523_merged_modified_factor_add_factor_text_length+6349+gpt4o_result.csv'
    data = pd.read_csv(file)
    
    label_index = create_label_index(data, label_cols, num_labels)
    
    X_train, X_test, y_train, y_test, train_index, test_index = multilabel_stratified_split(
        data, label_index, SEED, fold, n_splits=5, shuffle=True, column_name='pre_question')
    
    # df_test = df['title'] + ' ' + df['content'] 여쭤보기 
    df_test = X_test
    print(f'# of test:{len(df_test)}')
    print()

    print(" :: 2. Finetune Model :: ")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(token_model) # 모델 토크나이저
    tokenizer.pad_token = tokenizer.eos_token # 패딩
    model = LlamaForCausalLM.from_pretrained(
        token_model,
        #load_in_8bit=False,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map={'': 'cuda:1'}
    ) # 모델 불러오기
    

    pipe = pipeline(
        "text-generation",#"zero-shot-classification",#
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:1'}
    ) # 모델 텍스트 생성 파이프라인
    
    # run 
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    _pred, str_pred = run(512, df_test, pipe, max_attempts=3)
    
    # df_make = pd.DataFrame({'sentence':df_test.tolist(), 'diseases':_pred})
    # df_make.to_csv(f"result/{now}mentalllama_multi_few.csv", index=False)

    report = classification_report(y_test, _pred, output_dict=True)
    print(f"Classification report :\n{report}")

    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.rename(index={'0': 'Depression', '1': 'Anxiety', '2': 'Sleep','3': 'Eating', '4': 'Non-Disease'})

    log_dir = 'saved/log/Mentalllama/fold_'+str(fold)

    os.makedirs(log_dir, exist_ok=True)
    report_df.to_csv(log_dir+"/classification_report.csv")

    # print(X_test,len(X_test))
    # print(y_test,len(y_test))
    # print(str_pred,len(str_pred))
    
    y_test = tensor_to_list(y_test)
    res_df = pd.DataFrame(data={'text': X_test, 'label': y_test, 'pred': str_pred})
    res_df.to_csv(log_dir+'/error_analysis.csv', index=True)
    
    
def labels_to_list(decoded_labels):
    label_list = [decoded_labels[key] for key in decoded_labels]
    return label_list

# prompt 할 데이터  구역 정하기
def run(max_tok, df_test, pipe, max_attempts=3):
    pred = []
    str_pred = []
    for i in tqdm(range(len(df_test))):  # len(df_test)
        text = df_test[i]

        auth_key = "afebcba6-1a31-44bf-90a6-96666ad6eaac"  # Replace with your key
        translator = deepl.Translator(auth_key)

        post = translator.translate_text(text).text
        print(post)
        
        attempts = 0
        while attempts < max_attempts:
            prompt = f"""
            <<SYS>>You're a mental health professional.<</SYS>>

            [INST]
            Based on the questions given, infer the disorder the user is currently experiencing. 
            Your inference should be based on DSM-5 diagnostic criteria.
            The inference results are displayed separately for each of the five categories. 
            The five categories include four mental disease (depression disorders, anxiety disorders, sleep disorders, and eating disorders) and Non-Disease.
            Non-Disease can include cases where the user appears to be experiencing a condition outside of the four target conditions, or where it is unknown what condition the user is experiencing, or cases where the user has psychiatric symptoms but is not considered a mental disease.
            Here are some examples.

            EXAMPLE1
            Question: I usually go to school and am very sleepy during the day when I am active, but at night or in the early morning, I can't sleep and even if I lie down to sleep, I can't sleep well. Today I lay down to sleep around 3am, but I feel like I keep thinking and I don't feel like I'm sleeping, so I checked the time and it was 7:30. When I wake up, my eyes are very dry. On days when I feel like I slept a little deeply, it's very difficult to get up no matter how long it is, and even if I get up, I keep sleeping for about 1-2 hours. What is the symptom?
            Answer: The Mental disease that the user expericences are ['Sleep']
            
            EXAMPLE2
            Question: I've been suffering from binge eating disorder ever since I went on a crash diet 2 years ago. I lost 8 kilos through diet and exercise, but since then I've been terrified of gaining it back, so I don't eat much during the day and eat like crazy every weekend when I come home. When I'm home, even if I'm full, I eat uncontrollably until I run out of food, binge until my stomach bursts, take stomach medicine for bloating (I don't vomit), take laxatives to empty my bowels, then binge again the next day, and so I've gained 6kg in the two weeks I've been home. I hate looking in the mirror because I've gained weight, I don't want to go out, I don't want to see my friends, I'm depressed all day, I'm like I shouldn't eat from tomorrow and just workout, and then I eat as soon as I wake up and I'm heavy, and then I don't workout and I'm stuck in the house again... I feel so pathetic about myself. lol
            Answer: The Mental disease that the user expericences are ['Depression','Eating']

            EXAMPLE3
            Question: I'm in the first year of middle school. I'm suffering from a very difficult thought right now. I keep thinking about killing my mum and wondering if it will be the end of the world if I die too. I have health concerns for reference, I play gun games for about 3 hours a day, and I'm a boy. I'm really worried, is this a mental disorder, and if so, is it treatable? I would really appreciate an answer.
            Answer: The Mental disease that the user expericences are ['Non-Disease']

            Now, it's your turn.
            Question: {post}
            Answer:
            """
            sequences = pipe(
                prompt,
                max_new_tokens=max_tok,
                do_sample=True,
                temperature = 0.01
            ) # 제작해둔 파이프 라인에 프롬프트 넣어서 결과 생성
            
            generated_text = sequences[0]['generated_text'].replace(prompt, '') # 출력에서 프롬프트 부분 제거
            generated_text = generated_text.replace('                                     ', '')
            gc.collect()
            torch.cuda.empty_cache()

            decoded_labels = get_mentallama_labels(generated_text) # 요구한 라벨에 대한 출력만 뽑아내서 라벨 딕셔너리 생성

            if 'error' not in decoded_labels.values():
                break

            attempts += 1
            #print(f'{i}번째, {attempts}번째 시도')

        
        label_list = labels_to_list(decoded_labels)
        
        print(i,"pred")
        print(label_list)
        print("==============================")
        
        pred.append(label_list)
        str_pred.append(str(label_list))
    return pred, str_pred

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model", type=str, default="klyang/MentaLLaMA-chat-7B", help="Model of operation")
    parser.add_argument("--fold", type=int, help="fold")
    parser.add_argument("--label_cols", type=str, default="disease_idx", help="Column name for labels")
    parser.add_argument("--SEED", type=int, default=42, help="Seed for randomness")
    parser.add_argument("--num_labels", type=int, default=5, help="number of labels")
    
    args = parser.parse_args()
    main(args.model, args.fold, args.label_cols, args.SEED, args.num_labels)
