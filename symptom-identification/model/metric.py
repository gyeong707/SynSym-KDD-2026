from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef

def accuracy(y_true, y_pred):
    score = accuracy_score(y_true, y_pred)
    return score

def precision(y_true, y_pred):
    score = precision_score(y_true, y_pred, average='macro')
    return score

def recall(y_true, y_pred):
    score = recall_score(y_true, y_pred, average='macro')
    return score

def f1score(y_true, y_pred):
    score = f1_score(y_true, y_pred, average='macro')
    return score

def print_classification_report(target_name, y_true, y_pred, output_dict=False):
    report = classification_report(
      y_true,
      y_pred,
      target_names=target_name,
      zero_division=0,
      output_dict=output_dict
    )

    print(report)
    return report
