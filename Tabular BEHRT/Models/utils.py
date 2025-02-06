from itertools import chain
import numpy as np

def flatten_array(x):
    x = np.hstack(x)
    x = x.flatten()
    return x



def input_vocab(inputs, symbol= None):
    token2idx, idx2token = {}, {}
    inputs_list = np.unique(flatten_array(inputs.values))  ## the unique values in different lists from data.input 
    inputs_list = inputs_list.tolist()
    inputs_list.remove('SEP')
    inputs_list.remove('CLS')
  
    if symbol is None:
        symbol = ['PAD', 'CLS', 'SEP', 'MASK', 'UNK']
    
    for i in range(len(symbol)):
        token2idx[str(symbol[i])] = i
        idx2token[i] = str(symbol[i])
    
    for i, w in enumerate(inputs_list):
        
        token2idx[w] = i + len(symbol)
        idx2token = {i: w for i, w in enumerate(token2idx)}
        
    return token2idx, idx2token

def age_vocab(max_age, mon=1, symbol=None):
    age2idx = {}
    idx2age = {}
    if symbol is None:
        symbol = ['PAD', 'UNK']

    for i in range(len(symbol)):
        age2idx[str(symbol[i])] = i
        idx2age[i] = str(symbol[i])

    if mon == 12:
        for i in range(max_age):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    elif mon == 1:
        for i in range(max_age * 12):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    else:
        age2idx = None
        idx2age = None
    return age2idx, idx2age

def mod_vocab(inputs, symbol= None):
    mod2idx, idx2mod = {}, {}
    inputs_list = np.unique(flatten_array(inputs.values))  ## the unique values in different lists from data.input 
    inputs_list = inputs_list.tolist()
   
  
    if symbol is None:
        symbol = ['PAD', 'UNK']
    
    for i in range(len(symbol)):
        mod2idx[str(symbol[i])] = i
        idx2mod[i] = str(symbol[i])
    
    for i, w in enumerate(inputs_list):
        
        mod2idx[w] = i + len(symbol)
        idx2mod = {i: w for i, w in enumerate(mod2idx)}
        
    return mod2idx, idx2mod

def delay_vocab(max_delay, mon=1, symbol=None):
    """ computes the delay vocabulary
    max_delay : in years
    mon : the unit used 
          if 1 : The delay is in months
          if 12: The delay is in years 
          if 0.25: The delay is in weeks """
    
    delay2idx = {}
    idx2delay = {}
    if symbol is None:
        symbol = ['PAD', 'UNK']

    for i in range(len(symbol)):
        delay2idx[str(symbol[i])] = i
        idx2delay[i] = str(symbol[i])

    if mon == 12:
        for i in range(max_delay):
            delay2idx[str(i)] = len(symbol) + i
            idx2delay[len(symbol) + i] = str(i)
    elif mon == 1:
        for i in range(max_delay * 12):
            delay2idx[str(i)] = len(symbol) + i
            idx2delay[len(symbol) + i] = str(i)
               
    elif mon == 0.25:   ## compte en semaines 
        for i in range(max_delay * 53):
            delay2idx[str(i)] = len(symbol) + i
            idx2delay[len(symbol) + i] = str(i)
            
    else:
        delay2idx = None
        idx2delay = None
    return delay2idx, idx2delay

