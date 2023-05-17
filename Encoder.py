import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
from transformers import AlbertTokenizer, AlbertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from config import Config
import pytorch_lightning as pl


class Encoder(pl.LightningModule):
    r""" The encoder """
    def __init__(self):  
        super(Encoder, self).__init__()

        config = Config()
        data_language = config.data_language
        if data_language == 'EN':
            self.tok = GPT2Tokenizer.from_pretrained("gpt2",do_lower_case=False)
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.tok.pad_token = self.tok.eos_token
        else:
            self.tok = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
            self.model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
            
        
        self.config = Config()
        self.max_length = self.config.max_len
        
        

    def forward(self, inputs): 

        batch_input_encode = self.tok.batch_encode_plus(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        ) 
        # cuda
        batch_input_encode = {k: v.to(self.device) for k, v in batch_input_encode.items()}
        
        outputs = self.model(input_ids=batch_input_encode['input_ids'],
            attention_mask=batch_input_encode['attention_mask'],
            output_hidden_states=True,
            return_dict=True,
            ) 

        # Return to the hidden states of the final model of the output of the last time step

        pooler_output = outputs.hidden_states[-1][:, -1, :]  # [batch, dim]

        return pooler_output  # [batch, dim]
