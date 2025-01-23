import pandas as pd
import numpy as np
from sklearn.metrics import *
import random
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch import nn


from handle_file import handle_file

hf = handle_file()

## Datasets
def import_datasets(config,  min_visit , inputs= "inputs_normal_range_preprocessed_100_therap_removed_tolist"):
    
    """Import train data """
        
    data =  hf._load_pkl(config['data'])
    
    
    """Remove patients that have less visits than min visit"""
    # Remove patients with visits less than min visit

    previous_shape = data.shape[0]

    data['length'] = data[inputs].apply(lambda x: len([i for i in range(len(x)) if x[i] == 'SEP']))
    data = data[data['length'] >= min_visit]
    data = data.reset_index(drop=True)

    print("We lost {} patient(s)".format(previous_shape - data.shape[0]))
    
    """split data into train and validation data"""
    train, valid = np.split(data.sample(frac=1, random_state=42), [int(.9*len(data))])
    train.index = range(len(train))
    valid.index = range(len(valid))
    
    return train, valid

## Scores
""" Calculate precision for the NDP """
def compute_precision(logits, labels, training=True):
    """
    Compute score for the pretraining task which represents precision:
    Take as input logits in probabilities and labels in a 2D array
    Return: precision score
    
    """
    
    if training:
        labels, logits = labels.detach().cpu().numpy(), logits.detach().cpu().numpy()
        
    logs = nn.LogSoftmax(dim = 1)
 
    truepred = logs(torch.tensor(logits))
    outs = np.zeros(labels.shape)
    ind = [np.argmax(pred_x) for pred_x in truepred.numpy()]

    for i in range(len(ind)):
        outs[i][ind[i]] = 1
    
    precision = precision_score(labels, outs, average='macro')

    return precision


""" Calculate precision for the MLM """

def cal_acc(label, pred):
    logs = nn.LogSoftmax(dim = 1)
    label=label.cpu().numpy()
    ind = np.where(label!=-1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = logs(torch.tensor(truepred))
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = precision_score(truelabel, outs, average='macro')
    return precision

## 3. MISC

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
