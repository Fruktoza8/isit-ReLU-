from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall

def loss_function(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
