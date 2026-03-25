import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from base import BaseModel


class BertMultiLabelClassification(BaseModel):
    def __init__(self, model_type, num_labels) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_labels = num_labels
        # multi-label binary classification
        self.encoder = AutoModel.from_pretrained(model_type, use_auth_token=True, use_safetensors=True)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        x = outputs.last_hidden_state[:, 0, :]   # [CLS] 
        # x = outputs.last_hidden_state.mean(1)  # [bs, seq_len, hidden_size] -> [bs, hidden_size]
        x = self.dropout(x)
        logits = self.clf(x)
        return logits


class BertNaturalLanguageInference(nn.Module):
    def __init__(self, model_type, num_labels):
        super().__init__()
        self.model_type = model_type
        self.num_labels = num_labels
        # natural language inference
        self.encoder = AutoModel.from_pretrained(model_type, use_auth_token=True)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids)
        x = outputs.last_hidden_state[:, 0, :]   # [CLS]
        x = self.dropout(x)
        logits = self.clf(x)
        return logits


