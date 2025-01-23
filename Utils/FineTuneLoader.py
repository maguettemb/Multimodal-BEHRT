import sys
sys.path.append('/Users/maguette/Desktop/Cluster/BEHRT/Early_integration/')
sys.path.append('/Users/maguette/Desktop/Cluster/BEHRT/')

import numpy as np
from torch.utils.data.dataset import Dataset
from Utils.dataLoader_utils import seq_padding,code2index,label2index, position_idx, index_seg
import torch


class FinetuneDataset(Dataset):
    def __init__(self, token2idx, label2idx, mod2idx, NPI2idx, max_len, dataframe, patid='patid', code='inputs', age='age', label = 'label', mod = 'modalities', delay = 'delays_in_months', NPI='NPI', age2idx = None, del2idx=None, age_in_inputs = True, delays_in_inputs = True):
        
        self.vocab = token2idx
        self.mod2idx = mod2idx
        self.label_vocab = label2idx
        self.NPI2idx = NPI2idx
        self.max_len = max_len
        self.code = dataframe[code]
        self.label = dataframe[label]
        self.patid = dataframe[patid]
        self.modalities = dataframe[mod]
        self.NPI = dataframe[NPI]
        self.age_in_inputs = age_in_inputs
        self.delays_in_inputs = delays_in_inputs
        
        if age2idx is not None: 
            self.age2idx = age2idx
            self.age = dataframe[age]
            
        if del2idx is not None:
            self.del2idx = del2idx
            self.delay = dataframe[delay]
        

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        # cut data
        code = self.code[index]
        label = self.label[index]
        patid = self.patid[index]
        modalities = self.modalities[index]
        NPI = self.NPI[index]
        
        # change id type to list
        patid = [patid]

        # extract data
        code = code[(-self.max_len+1):]
        modalities = modalities[(-self.max_len+1):]
        NPI = NPI[(-self.max_len+1):]
      
        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        # pad age sequence and code sequence
        modalities = seq_padding(modalities, self.max_len, token2idx=self.mod2idx)
        NPI = seq_padding(NPI, self.max_len, token2idx=self.NPI2idx)
        
        
        tokens, code = code2index(code, self.vocab)
        _, label = label2index(label, self.label_vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
      #  label = seq_padding(label, self.max_len, symbol=-1)
    
        if self.del2idx is not None: 
            delay = self.delay[index]
            delay = delay[(-self.max_len+1):]
            delay = seq_padding(delay, self.max_len, token2idx=self.del2idx)
        
        
        if self.age2idx is not None: 
            age = self.age[index]
            age = age[(-self.max_len+1):]
            age = seq_padding(age, self.max_len, token2idx=self.age2idx)
            
        if self.age_in_inputs:
            return torch.LongTensor(code), torch.LongTensor(modalities), torch.LongTensor(NPI), torch.LongTensor(delay), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label), torch.LongTensor(patid)
        if self.delays_in_inputs:
            torch.LongTensor(code), torch.LongTensor(modalities),  torch.LongTensor(NPI), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label), torch.LongTensor(patid)
      
        return torch.LongTensor(age), torch.LongTensor(code), torch.LongTensor(modalities),  torch.LongTensor(NPI), torch.LongTensor(delay), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label), torch.LongTensor(patid)

    def __len__(self):
        return len(self.code)
    
    
 