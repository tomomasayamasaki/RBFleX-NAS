# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:19:30 2020

@author: Jiang Yuxin
"""

import torch
from torch import nn
from BERT_CustomActivation_Models import (
    BertForSequenceClassification, 
    AutoTokenizer
)
import torch.nn.init as init
        
     
        
class BertModel(nn.Module):
    def __init__(self, requires_grad = True, activation='GELU'):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2',num_labels = 2, ACTIVATION = activation)

        self.tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

        #init.xavier_uniform_(self.bert.weight)
        #init.zeros_(self.fc.bias)

        for layer in self.bert.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities