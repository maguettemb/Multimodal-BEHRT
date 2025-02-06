from torch.utils.data.dataset import Dataset
import numpy as np
from dataLoader.utils import seq_padding,position_idx,index_seg,random_mask
import torch


def get_delim_for_chunks(sequence, n):
    """Compute chunks of sequence that will be use to predict delay
    Take as input the sequence and the number of visits we want to consider for the chunk
    return the chunked sequence for the task and the real delay following that sequence """
    np.random.seed(seed=2)
    
    num_of_visits=0
    sep_index = [0]  ## start with the first CLS so that when we will choose random start, the first vist can be part of the choices

    for i in range(len(sequence)): 
        if sequence[i] == 'SEP': 
            num_of_visits+=1
            sep_index.append(i+1)
            
    if n >= num_of_visits:
        n = num_of_visits -1 

    start = np.random.choice(sep_index[:len(sep_index)-1-(n)], 1)[0]  ### Start for the chunk must be from 0 to n step before the end of the sep_sequence (in order to have n visits at least from the start) and - 1 is for the last SEP which is the end of the sequence that can't be taken into account to predict delay (bc it's the last visit)
    stop = sep_index[sep_index.index(start) + n]
    
    return start, stop

def get_visits_for_delay_prediction(sequence, n, token2idx):
    start, stop = get_delim_for_chunks(sequence, n)
    
    chunk = sequence[start+1:stop]
    delay = sequence[stop]
    
    output_token = []
    for i, token in enumerate(chunk):
        output_token.append(token2idx.get(token, token2idx['UNK']))
        
    return chunk, output_token, delay
    
def get_other_subsequences(sequence, other_sequence, n):
   
    start, stop = get_delim_for_chunks(sequence, n)
    chunk = other_sequence[start+1:stop]
    return chunk
      
    
class PretrainLoader(Dataset):
    def __init__(self,  token2idx, mod2idx, NPI2idx, max_len, dataframe, mlb, code='inputs', age='age', mod = 'modalities', delay = 'delays_in_months', NPI='NPI', age2idx = None, del2idx=None, age_in_inputs = False, delays_in_inputs = False, right_percentage = 1, split = False, seed = 42, pretrain='NDP', n = 5):
       
        self.vocab = token2idx
        self.mod2idx = mod2idx
        self.NPI2idx = NPI2idx
        self.max_len = max_len
        self.split = split
        self.r_percentage = right_percentage
        self.seed = seed
        self.code = dataframe[code]
        self.modalities = dataframe[mod]
        self.NPI = dataframe[NPI]
        self.age_in_inputs = age_in_inputs
        self.delays_in_inputs = delays_in_inputs
        self.n = n
        self.pretrain = pretrain
        self.del2idx = del2idx
        self.age2idx = age2idx
        self.mlb = mlb
        if age2idx is not None: 
            self.age = dataframe[age]
            
        if del2idx is not None:
            self.delay = dataframe[delay]
        

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        
        if self.pretrain == 'MLM':

            # extract data
            code = self.code[index][(-self.max_len+1):]
            modalities = self.modalities[index][(-self.max_len+1):]
            NPI = self.NPI[index][(-self.max_len+1):]
        
            mask = np.ones(self.max_len)
            mask[len(code):] = 0
        
            # pad age sequence and code sequence
            modalities = seq_padding(modalities, self.max_len, self.split, self.r_percentage, token2idx=self.mod2idx)
            NPI = seq_padding(NPI, self.max_len, self.split, self.r_percentage, token2idx=self.NPI2idx)

    
            # put mask in input sequence
            # token : unchanged sequence, code :  index of sequence with masked token = 3, label: -1 for unmasked token and label index for masked token 
            tokens, code, label = random_mask(code, self.vocab, self.seed)
    
           # get position code and segment code
            tokens = seq_padding(tokens, self.max_len, self.split, self.r_percentage)
            position = position_idx(tokens)
            segment = index_seg(tokens)

           # pad code and label
            code = seq_padding(code, self.max_len, self.split, self.r_percentage, symbol=self.vocab['PAD'])
            label = seq_padding(label, self.max_len, self.split, self.r_percentage, symbol=-1)
            
            if self.del2idx is not None:
                delay = self.delay[index][(-self.max_len+1):]
                delay = seq_padding(delay, self.max_len, self.split, self.r_percentage, token2idx=self.del2idx)
            
            if self.age2idx is not None: 
                age = self.age[index][(-self.max_len+1):]
                age = seq_padding(age, self.max_len, self.split, self.r_percentage, token2idx=self.age2idx)
                
            if self.delays_in_inputs: 
                return torch.LongTensor(code), torch.LongTensor(modalities), torch.LongTensor(segment), torch.LongTensor(position), torch.LongTensor(NPI), torch.LongTensor(mask), torch.LongTensor(label)
            
            elif self.age_in_inputs:     
                return torch.LongTensor(code), torch.LongTensor(modalities), torch.LongTensor(delay), torch.LongTensor(segment), torch.LongTensor(position), torch.LongTensor(NPI), torch.LongTensor(mask), torch.LongTensor(label)
          
            else:
                return torch.LongTensor(code), torch.LongTensor(modalities), torch.LongTensor(age), torch.LongTensor(delay), torch.LongTensor(segment), torch.LongTensor(position), torch.LongTensor(NPI), torch.LongTensor(mask), torch.LongTensor(label)

        elif self.pretrain == 'NDP':
            
            code = self.code[index]
        
            modalities = self.modalities[index]
            modalities = get_other_subsequences(code, modalities, n = self.n)
            modalities = seq_padding(modalities, self.max_len, token2idx=self.mod2idx)
            
            NPI = self.NPI[index]
            NPI = get_other_subsequences(code, NPI, n = self.n)
            NPI = seq_padding(NPI, self.max_len, token2idx=self.NPI2idx)
            
            if self.del2idx is not None: 
                delay = self.delay[index]
                delay = get_other_subsequences(code, delay, n=self.n)
                delay = seq_padding(delay, self.max_len, token2idx=self.del2idx)
                
            if self.age2idx is not None:
                age = self.age[index]
                age = get_other_subsequences(code, age, n= self.n)
                age = seq_padding(age, self.max_len, token2idx=self.age2idx)
                
            tokens, code, label = get_visits_for_delay_prediction(code, n=self.n, token2idx=self.vocab)
            
            mask = np.ones(self.max_len)
            mask[len(code):] = 0
            
            # get position code and segment code
            tokens = seq_padding(tokens, self.max_len, self.split, self.r_percentage)
            position = position_idx(tokens)
            segment = index_seg(tokens)
            
            code = seq_padding(code, self.max_len, self.split, self.r_percentage, symbol=self.vocab['PAD'])
            
            label = torch.LongTensor(self.mlb.transform([[label]]))
            
            if self.delays_in_inputs: 
                return torch.LongTensor(code), torch.LongTensor(modalities), torch.LongTensor(segment), torch.LongTensor(position), torch.LongTensor(NPI), torch.LongTensor(mask), label
            
            elif self.age_in_inputs:     
                return torch.LongTensor(code), torch.LongTensor(modalities), torch.LongTensor(delay), torch.LongTensor(segment), torch.LongTensor(position), torch.LongTensor(NPI), torch.LongTensor(mask), label
            
            else:
                return torch.LongTensor(code), torch.LongTensor(modalities), torch.LongTensor(age), torch.LongTensor(delay), torch.LongTensor(segment), torch.LongTensor(position), torch.LongTensor(NPI), torch.LongTensor(mask), label
 

    def __len__(self):
        return len(self.code)