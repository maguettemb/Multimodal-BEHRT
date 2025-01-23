import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler, BertOnlyMLMHead
from transformers.modeling_utils import PreTrainedModel
import numpy as np
from typing import Optional, Tuple, Union
from transformers.models.bert.modeling_bert import *
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import torch
from torch import nn
import numpy as np
import math

class BertEmbeddings(nn.Module):
    """Construct the embeddings from all the modalities: events, age, delays, segment, and modalities
    """

    def __init__(self, config, feature_dict=None):
        super(BertEmbeddings, self).__init__()
        self.activation = config.activation
        self.position_embedding_type = getattr(config, "position_embedding_type", "bert_embedding")
        if feature_dict is None:
            self.feature_dict = {
                'word': True,
                'modalities' : True,
                'seg': True,
                'age': True,
                'delays':True,
                'NPI':True,
                'position': True,
            }
        else:
            self.feature_dict = feature_dict

        if feature_dict['word']:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        if feature_dict['seg']:
            self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        
        if feature_dict['modalities']:
            self.modalities_embeddings = nn.Embedding(config.modalities_vocab_size, config.hidden_size)

        if feature_dict['age']:
            if self.activation: 
                self.age_embeddings = SineActivation(config.age_vocab_size, config.hidden_size)
            else:
                self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
            
        if feature_dict['delays']:
            if self.activation:
                self.delays_embeddings = SineActivation(config.delays_vocab_size, config.hidden_size)
            else:
                self.delays_embeddings = nn.Embedding(config.delays_vocab_size, config.hidden_size)

        if feature_dict['NPI']:
            self.NPI_embeddings = nn.Embedding(config.NPI_vocab_size, config.hidden_size)

        if feature_dict['position']:
            if self.position_embedding_type == "absolute":
                self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            else:
                self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
                from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    

    def forward(self, word_ids=None, modalities_ids=None, age_ids=None, delays_ids=None, seg_ids=None, posi_ids=None, NPI_ids=None):
        embeddings = self.word_embeddings(word_ids)
        
        if self.feature_dict['seg']:
            segment_embed = self.segment_embeddings(seg_ids)
            embeddings+=segment_embed
        
        if self.feature_dict['modalities']:
            modalities_embed = self.modalities_embeddings(modalities_ids)
            embeddings+=modalities_embed

        if self.feature_dict['age']:
            age_embed = self.age_embeddings(age_ids)
            embeddings+=age_embed
            
        if self.feature_dict['delays']:
            delays_embed = self.delays_embeddings(delays_ids)
            embeddings+=delays_embed
        
        if self.feature_dict['NPI']:
            NPI_embed = self.delays_embeddings(NPI_ids)
            embeddings+=NPI_embed

        if self.feature_dict['position']:
            posi_embeddings = self.posi_embeddings(posi_ids)
            embeddings+=posi_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings        

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        emb = list()
        for batch in tau:
            out_list = list()
            for pos in batch: 
                out = t2v(pos.reshape(-1, 1), self.f, self.out_features, self.w, self.b, self.w0, self.b0)
                out_list.append(torch.flatten(out))
            emb.append(torch.stack(out_list, 0))
        return torch.stack(emb, 0)

class BertModel(BertPreTrainedModel):
    def __init__(self, config, feature_dict,  add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config=config, feature_dict = feature_dict)
        self.encoder = BertEncoder(config=config)
        self.pooler = BertPooler(config)
        self.post_init()
       

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        


    def forward(self, input_ids, mod_ids=None, age_ids=None, delays_ids=None, seg_ids=None, posi_ids=None,  NPI_ids=None, attention_mask=None, output_attentions= None, output_hidden_states = None, return_dict = None)-> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions 
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
      
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, modalities_ids, age_ids, delays_ids, seg_ids, posi_ids, NPI_ids)
        encoder_outputs = self.encoder(embedding_output,
                                      attention_mask = extended_attention_mask, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        
       # sequence_output = encoded_layers[-1]
         
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None 
        
     #   if not output_all_encoded_layers:
    #        encoded_layers = encoded_layers[-1]
            
        if not return_dict: 
            return (sequence_output, pooled_output) + encoded_layers[1:]
         #   return encoded_layers, pooled_output

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values = encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForClassification(BertPreTrainedModel):
    def __init__(self, config,  num_labels, feature_dict, class_weights):
        super(BertForClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config, feature_dict)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.class_weights = class_weights
        #self.apply(self.init_bert_weights)
        self.post_init()

    def forward(self, input_ids=None, modalities_ids = None, age_ids=None, delays_ids = None, seg_ids=None, posi_ids=None,  NPI_ids =None, attention_mask=None, labels=None, output_attentions = True, output_hidden_states = True, return_dict = True)-> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(input_ids, modalities_ids, age_ids, delays_ids, seg_ids, posi_ids,  NPI_ids , attention_mask,  output_attentions = output_attentions, output_hidden_states=output_hidden_states, return_dict = return_dict)
       
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        

        if labels is not None:
            if self.num_labels > 1 & (labels.dtype == torch.long or labels.dtype == torch.int):
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            else:
            
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
                loss = loss_fct(logits, labels)
  
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
