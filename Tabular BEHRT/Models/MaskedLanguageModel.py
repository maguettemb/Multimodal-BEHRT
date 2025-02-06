
import torch.nn as nn
import pytorch_pretrained_bert as Bert
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler, BertOnlyMLMHead
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutputWithPoolingAndCrossAttentions

from transformers.modeling_utils import PreTrainedModel
import numpy as np
import torch
from typing import Optional, Tuple, Union

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.config = config
        self.activation = config.activation
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'bert_embedding'
                                              )
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.modalities_embeddings = nn.Embedding(config.modalities_vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.NPI_embeddings = nn.Embedding(config.NPI_vocab_size, config.hidden_size)
        
        if self.position_embedding_type == "absolute":
            self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        
        if self.activation:
            self.age_embeddings = SineActivation(config.age_vocab_size, config.hidden_size)
            self.delays_embeddings = SineActivation(config.delay_vocab_size, config.hidden_size)
            
        else: 
            self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
            self.delays_embeddings = nn.Embedding(config.delay_vocab_size, config.hidden_size)
            
        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)  #eps default = 1e-12
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, modalities_ids = None, age_ids=None, delays_ids = None, seg_ids=None, posi_ids=None, NPI_ids=None, age_in_inputs=False, delays_in_inputs = False):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)
        if delays_ids is None:
            delays_ids = torch.zeros_like(word_ids)
        if modalities_ids is None:
            modalities_ids = torch.zeros_like(word_ids)
        if modalities_ids is None:
            NPI_ids = torch.zeros_like(word_ids)
        
        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        modalities_embed = self.modalities_embeddings(modalities_ids)
        age_embed = self.age_embeddings(age_ids)
        delays_embed = self.delays_embeddings(delays_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)
        NPI_embeddings = self.NPI_embeddings(NPI_ids)
        
        if self.config.age_in_inputs:
            embeddings = word_embed + segment_embed + delays_embed + modalities_embed + posi_embeddings + NPI_embeddings
        if self.config.delays_in_inputs:
            embeddings = word_embed + segment_embed + modalities_embed + posi_embeddings + NPI_embeddings
            
        else:
            embeddings = word_embed + segment_embed + posi_embeddings + modalities_embed + delays_embed + age_embed + NPI_embeddings

            
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
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.age_in_inputs = config.age_in_inputs
        self.delays_in_inputs = config.delays_in_inputs
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.post_init()


    def forward(self, input_ids,  modalities_ids = None, age_ids=None, delays_ids = None, seg_ids=None, posi_ids=None, NPI_ids=None, attention_mask=None,
                output_all_encoded_layers=True)-> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)
        if delays_ids is None:
            delays_ids = torch.zeros_like(input_ids)
        if modalities_ids is None:
            modalities_ids = torch.zeros_like(input_ids)
        if NPI_ids is None: 
            NPI_ids = torch.zeros_like(NPI_ids)
            
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

        embedding_output = self.embeddings(input_ids, modalities_ids, age_ids, delays_ids, seg_ids, posi_ids, NPI_ids, age_in_inputs = self.age_in_inputs, delays_in_inputs = self.delays_in_inputs)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        
        if config.is_decoder:
            raise ('To use BertForMaskedLM, config.is_decoder need to be set to False for bi-directional self-attention')
            
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)#, self.bert.embeddings.word_embeddings.weight)
        self.post_init()

    def forward(self, input_ids, modalities_ids = None, age_ids=None, delays_ids = None, seg_ids =None, posi_ids=None, NPI_ids= None, attention_mask=None, masked_lm_labels=None)-> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        outputs = self.bert(input_ids, modalities_ids, age_ids, delays_ids, seg_ids, posi_ids, NPI_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        
        masked_lm_loss = None
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss, prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
            
        else:
            return prediction_scores
            
   #     return MaskedLMOutput(
     #       loss=masked_lm_loss,
     #       logits=prediction_scores,
    #        hidden_states=outputs.hidden_states,
    #        attentions=outputs.attentions,
    #    )
    
    