import sys
sys.path.append('BEHRT/')
sys.path.append('BEHRT/Early_integration/')
sys.path.append('BEHRT/Early_integration/Utils')


from utils_for_pretraining import *
from PretrainLoader import *
from optimiser import *
from utils import *
import pandas as pd
import numpy as np
from transformers.configuration_utils import PretrainedConfig
import warnings
warnings.filterwarnings(action='ignore') 
from add_endpoints import add_endp
import time 
import torch
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
import os
from handle_file import handle_file

def empty_cuda():
    """ Empty the cache to enable the use of the gpu """
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    

def training_per_epoch(e, model, optim, config, loader, train_params):
        
         
        output_dir = "./pretrained_models/"
        create_folder(output_dir)  ## on crée un folder ou stocker le modele 
        
        tr_loss, temp_loss = [], []
        precision, temp_precision = [], []
        nb_tr_examples, nb_tr_steps = 0, 0
        cnt= 0
        start = time.time()

        model.train()
        
        for step, batch in enumerate(loader):

            cnt +=1
                
            batch = tuple(t.to(train_params['device']) for t in batch)
            
            if config['age_in_inputs']:
                input_ids, mod_ids, del_ids, segment_ids, posi_ids, NPI_ids, attMask, masked_label = batch
                age_ids = None
                age_ids = age_ids.to(train_params['device'])
                
            if config['delays_in_inputs']:
                input_ids, mod_ids, segment_ids, posi_ids, NPI_ids, attMask, masked_label = batch
                age_ids = None
                age_ids = age_ids.to(train_params['device'])
                
                del_ids = None
                del_ids = del_ids.to(train_params['device'])
                
            else:
                input_ids, mod_ids, age_ids, del_ids, segment_ids, posi_ids, NPI_ids, attMask, masked_label = batch


            loss, pred, label = model(input_ids=input_ids, modalities_ids=mod_ids, seg_ids=segment_ids, posi_ids=posi_ids, NPI_ids=NPI_ids, attention_mask=attMask, masked_lm_labels=masked_label)

            if config.gradient_accumulation_steps >1:
                loss = loss/config.gradient_accumulation_steps

            loss.backward()
            temp_loss.append(loss.item())
            tr_loss.append(loss.item())

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            
            if step % 100 == 0:
                with open("mydocument.txt", mode = "a") as f:
                    print("epoch: {}\t| cnt: {}\t|Loss: {}\t| precision: {:.4f}\t| time: {:.2f}".format(e, cnt, np.mean(temp_loss), cal_acc(label, pred), time.time()-start))
                    temp_loss = []
                    start = time.time()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optim.step()
                optim.zero_grad()

        # print("** ** * Saving fine - tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        create_folder(file_config['model_path'])
        
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        
        cost = time.time() - start

        return np.mean(tr_loss), cost, cal_acc(label, pred), model_to_save
    
def evaluate_per_epoch(model, loader, train_params):
    
    temp_loss, val_loss = [], []
    nb_tr_examples, nb_tr_steps = 0, 0
  
    start = time.time()
 
    model.eval()
    
    for step, batch in enumerate(loader):
        
        
        batch = tuple(t.to(train_params['device']) for t in batch)

        if config['age_in_inputs']:
            input_ids, mod_ids, del_ids, segment_ids, posi_ids, NPI_ids, attMask, masked_label = batch
            age_ids = None
            age_ids = age_ids.to(train_params['device'])

        if config['delays_in_inputs']:
            input_ids, mod_ids, segment_ids, posi_ids, NPI_ids, attMask, masked_label = batch
            age_ids = None
            age_ids = age_ids.to(train_params['device'])

            del_ids = None
            del_ids = del_ids.to(train_params['device'])

        else:
            input_ids, mod_ids, age_ids, del_ids, segment_ids, posi_ids, NPI_ids, attMask, masked_label = batch

        
        with torch.no_grad():
            loss, pred, label = model(input_ids = input_ids, modalities_ids = mod_ids, age_ids=age_ids,delays_ids=del_ids, seg_ids=segment_ids, posi_ids=posi_ids, NPI_ids=NPI_ids, attention_mask=attMask, masked_lm_labels=masked_label)
            loss = loss.cpu()
            val_loss.append(loss)
            
        temp_loss.append(loss)
        
        if step % 50 == 0:
            print("Validation:")
            print("|Val Loss: {}\t| Val precision: {:.4f}\t| time: {:.2f}".format(np.mean(temp_loss), cal_acc(label, pred), time.time()-start))
            temp_loss = []
            start = time.time() 
            
    return cal_acc(label, pred), val_loss, np.mean(val_loss)
