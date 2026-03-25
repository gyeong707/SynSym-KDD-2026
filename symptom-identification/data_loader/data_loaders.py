from torch.utils.data import Dataset, DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import torch
import ast
import numpy as np

# Default setting
class MLCDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx): 
        text = self.X[idx]
        label = self.y[idx]

        self.encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        item = {
            'input_ids': self.encoding['input_ids'].flatten(),
            'attention_mask': self.encoding['attention_mask'].flatten(),
            'labels': label
        }
        return item


# Default setting
class NLIDataset(Dataset):
    def __init__(self, premise, hypothesis, y, tokenizer, max_length):
        self.premise = premise
        self.hypothesis = hypothesis
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.premise)
    
    def __getitem__(self, idx): 
        premise = self.premise[idx]
        hypothesis = self.hypothesis[idx]
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        self.encoding = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
    )
        item = {
            'input_ids': self.encoding['input_ids'].flatten(),
            'token_type_ids': self.encoding['token_type_ids'].flatten(),
            'attention_mask': self.encoding['attention_mask'].flatten(),
            'labels': label
        }
        return item
    

class EvalDataset(Dataset):
    def __init__(self, premise, hypothesis, tokenizer, max_length):
        self.premise = premise
        self.hypothesis = hypothesis
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.premise)
    
    def __getitem__(self, idx): 
        premise = self.premise[idx]
        hypothesis = self.hypothesis[idx]
        
        self.encoding = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
    )
        item = {
            'input_ids': self.encoding['input_ids'].flatten(),
            'token_type_ids': self.encoding['token_type_ids'].flatten(),
            'attention_mask': self.encoding['attention_mask'].flatten(),
        }
        return item

#-----------------------------------External Functions------------------------------------#

def multilabel_stratified_split(data, label, seed, fold, n_splits=5, shuffle=True, column_name='post'):
    y = label
    X = data[column_name].values  # sentence
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for i, (train_index, test_index) in enumerate(mskf.split(X, y)):
        print(i, fold)
        if fold == i:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            return X_train, X_test, y_train, y_test, train_index, test_index


def make_label_vector(data, label_cols):
    # apply ast.literal_eval to convert string to list
    print(data[label_cols][0])
    data[label_cols] = data[label_cols].map(lambda x: ast.literal_eval(x))
    return torch.tensor(data[label_cols], dtype=torch.float32)


def revert_label_index(labels, label_dict):
    reversed_dict = {v: k for k, v in label_dict.items()}
    data = []
    for i in range(labels.shape[0]):
        temp = []
        row = labels[i]
        indices = np.where(row == True)[0]
        for idx in indices:
            temp.append(reversed_dict[idx])
        data.append(temp)
    return data

def decoding_input_ids(tokenizer, encoded_text):
    return tokenizer.decode(encoded_text, skip_special_tokens=True)

def make_mlc_instance(premises, symptom_list, desc_list):
    X_premise = []
    X_hypothesis = []

    for premise in premises:
        for _, desc in zip(symptom_list, desc_list):
            X_premise.append(premise)      
            X_hypothesis.append(desc)

    return X_premise, X_hypothesis