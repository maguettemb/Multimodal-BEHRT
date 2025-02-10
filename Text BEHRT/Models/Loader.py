import numpy as np
from torch.utils.data.dataset import Dataset
from Utils.dataLoader_utils import seq_padding,code2index,label2index, position_idx, index_seg
import torch


class Loader(Dataset):
    def __init__(self, token2idx, label2idx, dataframe, max_len, patid='patid', code='inputs', age='age', label = 'label', mod = 'modalities', delay = 'delays_in_months'):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.vocab = token2idx
        self.label_vocab = label2idx
        self.max_len = max_len
        self.code = dataframe[code]
        self.label = dataframe[label]
        self.patid = dataframe[patid]

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        # cut data
        code = self.code[index]
        label = self.label[index]
        patid = self.patid[index]
        modalities = self.modalities[index]
        
        # change id type to list
        patid = [patid]

        # extract data
        code = code[(-self.max_len+1):]
        
        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(code):] = 0

        tokens, code = code2index(code, self.vocab)
        _, label = label2index(label, self.label_vocab)

        # get position code and segment code
        tokens = seq_padding(tokens, self.max_len)
        position = position_idx(tokens)
        segment = index_seg(tokens)

        # pad code and label
        code = seq_padding(code, self.max_len, symbol=self.vocab['PAD'])
      #  label = seq_padding(label, self.max_len, symbol=-1)
        

        return torch.LongTensor(code), torch.LongTensor(position), torch.LongTensor(segment), \
               torch.LongTensor(mask), torch.LongTensor(label), torch.LongTensor(patid)

    def __len__(self):
        return len(self.code)
    
    
 