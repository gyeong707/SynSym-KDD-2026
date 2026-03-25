import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import jaccard_score
import numpy as np

# Multi-Class Classification
def cross_entropy_loss(output, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)

# Multi-Label Classification
def bce_with_logits(output, target):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output, target)

# Multi-Label Classification without sigmoid function
def bce_loss(output, target):
    criterion = nn.BCELoss()
    return criterion(output, target)

# Negative Log Likelihood Loss (Multi-Class Classification without log_softmax)
def nll_loss(output, target):
    return F.nll_loss(output, target)



