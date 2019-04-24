"""
    Holds PyTorch models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform

import numpy as np

from math import floor, sqrt, sin, cos
import random
import sys
import time

class BaseModel(nn.Module):

    def __init__(self, Y, word_embeddings_matrix, vocab_size, dropout=0.5, gpu=True, embed_size=100, embed_freeze=False, hier=False):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.gpu = gpu
        self.hier = hier
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        
        print('loading model')

        #make embedding layer
        if word_embeddings_matrix is not None:               
            W = torch.Tensor(word_embeddings_matrix)
            self.embed = nn.Embedding.from_pretrained(W, freeze=embed_freeze)
            self.embed.padding_idx = 0
        else:
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)

    def _get_loss(self, yhat, target):
        #calculate the BCE011
        return F.binary_cross_entropy_with_logits(yhat, target)

class Attention(torch.nn.Module):
    def __init__(self, n_dim, n_labels, embed_desc=False, desc_dim=300):

        super(Attention, self).__init__()
        
        self.embed_desc = embed_desc
        self.softmax = nn.Softmax(dim=2)
        
        if self.embed_desc:
            assert n_dim % 2 == 0
            self.gru_labels = nn.GRU(desc_dim, int(n_dim/2), bidirectional= True, batch_first=True)
            self.linear = nn.Linear(n_dim, n_dim)
            self.weight = self.linear.weight
            self.activation = nn.Tanh()
        
        else:
            self.U = nn.Parameter(torch.FloatTensor(n_labels, n_dim))
            self.weight = self.U
            #self.U = nn.Linear(n_dim, n_labels, bias=False)
            #self.weight = self.U.weight

    def forward(self, x, desc_data=None):

        if self.embed_desc:
            n_labels = desc_data.size(0)
            _, desc_data = self.gru_labels(desc_data)
            desc_data = desc_data.transpose(0,1).contiguous().view(n_labels,-1)
            desc_data = self.activation(self.linear(desc_data))
            alpha = self.softmax(desc_data.matmul(x.transpose(1,2)))
        else:
            #alpha = self.softmax(self.U(x).transpose(1,2))
            alpha = self.softmax(self.U.matmul(x.transpose(1,2)))

        return alpha.matmul(x), alpha
        
class ConvEncoder(nn.Module):
    def __init__(self, n_dim_in, n_dim_out, kernel_size, padding=True):
        super().__init__()
        self.conv = nn.Conv1d(n_dim_in, n_dim_out, kernel_size=kernel_size, padding=floor(kernel_size/2) if padding == True else False)
        xavier_uniform(self.conv.weight)
        
    def forward(self, x):
        return self.conv(x.transpose(1,2)).transpose(1,2)
        
class GRUEncoder(nn.Module):
    def __init__(self, n_dim_in, n_dim_out):
    
        assert n_dim_out%2 == 0
        
        super().__init__()
        self.gru = nn.GRU(n_dim_in, int(n_dim_out/2), bidirectional= True, batch_first=True)
        self.linear = nn.Linear(n_dim_out, n_dim_out)
        xavier_uniform(self.linear.weight)
                
    def forward(self, x):
        s0, s1, s2 = x.size()
        x, _ = self.gru(x)
        x = x.transpose(0,1).contiguous().view(s0, s1, -1)
        return self.linear(x)
  
class DummyModel(nn.Module):
    def __init__(self, Y, dicts, gpu, hier=False):
        super(DummyModel, self).__init__()
        self.Y = Y
        self.hier = hier
        self.gpu = gpu
        
        n = sum(i for i in dicts['train_examples'].values())
        self.distribution = torch.cuda.FloatTensor([i/n for i in dicts['train_examples'].values()])
        
    def forward(self, x, target, desc_data=None, get_attention=False):

        #yhat = torch.cuda.FloatTensor(torch.Size([x.size(0), self.Y])).fill_(0)
        yhat = self.distribution.expand_as(target).bernoulli()
        attn = torch.cuda.FloatTensor(x.size(0), self.Y, x.size(1)).fill_(0)
        
        return yhat, F.binary_cross_entropy_with_logits(yhat, target), attn

def attention(q, k, v, d_k, dropout=None): 
          
    scores = torch.matmul(q, k.transpose(-2, -1)) /  sqrt(d_k) 
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    return torch.matmul(scores, v), scores
            
class MultiHeadConvAttention(nn.Module):
    def __init__(self, heads, d_model_start, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_conv = nn.Conv1d(d_model_start, d_model, kernel_size=9, padding=floor(9/2))
        self.v_conv = nn.Conv1d(d_model_start, d_model, kernel_size=9, padding=floor(9/2))
        self.k_conv = nn.Conv1d(d_model_start, d_model, kernel_size=9, padding=floor(9/2))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
        self.activation = nn.Tanh()
        
        xavier_uniform(self.q_conv.weight)
        xavier_uniform(self.v_conv.weight)
        xavier_uniform(self.k_conv.weight)
        xavier_uniform(self.out.weight)

    def forward(self, value):
        
        v = value
        k = v
        q = v
        
        bs = v.size(0)

        # perform linear operation and split into h heads

        k = self.k_conv(k.transpose(1,2)).transpose(1,2).view(bs, -1, self.h, self.d_k)
        q = self.q_conv(q.transpose(1,2)).transpose(1,2).view(bs, -1, self.h, self.d_k)
        v = self.v_conv(v.transpose(1,2)).transpose(1,2).view(bs, -1, self.h, self.d_k)
        
        k = self.activation(k)
        q = self.activation(q)
        v = self.activation(v)
        
        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention
        attended, scores = attention(q, k, v, self.d_k, self.dropout)

        # concatenate heads and put through final linear layer
        concat = attended.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        #output = self.out(concat)
        
        #output = concat + value
        #output = output + value
        
        output = concat
        
        return output, scores
        
class MultiHeadTargetAttention(nn.Module): 
    def __init__(self, Y, heads, d_model, dropout = 0.1): 
        super().__init__() 
  
        self.d_model = d_model 
        self.d_k = d_model // heads 
        self.h = heads
  
        self.q = nn.Parameter(torch.zeros(Y, d_model))
          
        self.v_linear = nn.Linear(d_model, d_model) 
        self.k_linear = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout) 
        self.out = nn.Linear(d_model, d_model)
         
        xavier_uniform(self.v_linear.weight)
        xavier_uniform(self.k_linear.weight)
        xavier_uniform(self.out.weight) 
  
    def forward(self, v): 
  
        k = v 
        q = v 
          
        bs = v.size(0) 
  
        # perform linear operation and split into h heads 
  
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) 
        q = self.q.view(-1, self.h, self.d_k) 
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) 
          
        # transpose to get dimensions bs * h * sl * d_model 
  
        k = k.transpose(1,2) 
        q = q.transpose(0,1) 
        v = v.transpose(1,2) 
  
        # calculate attention 
        attended, scores = attention(q, k, v, self.d_k, self.dropout) 
  
        # concatenate heads and put through final linear layer 
        concat = attended.transpose(1,2).contiguous().view(bs, -1, self.d_model) 
        output = self.out(concat) 
  
        return output, scores
         
class MultiHeadConvTargetAttention(nn.Module): 
    def __init__(self, Y, heads, d_model_start, d_model, dropout = 0.1): 
        super().__init__() 
  
        self.d_model = d_model 
        self.d_k = d_model // heads 
        self.h = heads 
        
        self.activation = nn.Tanh()
  
        self.q = nn.Parameter(torch.zeros(Y, d_model))
        
        self.v_conv = nn.Conv1d(d_model_start, d_model, kernel_size=9, padding=floor(9/2))
        self.k_conv = nn.Conv1d(d_model_start, d_model, kernel_size=9, padding=floor(9/2))
         
        self.dropout = nn.Dropout(dropout) 
        self.out = nn.Linear(d_model, d_model)
         
        xavier_uniform(self.q)
        xavier_uniform(self.v_conv.weight)
        xavier_uniform(self.k_conv.weight)
        xavier_uniform(self.out.weight) 
  
    def forward(self, value): 
  
        v = value
        k = v 
        q = v 
          
        bs = v.size(0) 
  
        # perform linear operation and split into h heads 
  
        k = self.k_conv(k.transpose(1,2)).transpose(1,2).view(bs, -1, self.h, self.d_k)
        q = self.q.view(-1, self.h, self.d_k) 
        v = self.v_conv(v.transpose(1,2)).transpose(1,2).view(bs, -1, self.h, self.d_k)

        k = self.activation(k)
        q = self.activation(q)
        v = self.activation(v)
          
        # transpose to get dimensions bs * h * sl * d_model 
  
        k = k.transpose(1,2) 
        q = q.transpose(0,1) 
        v = v.transpose(1,2) 
  
        # calculate attention 
        attended, scores = attention(q, k, v, self.d_k, self.dropout) 
  
        # concatenate heads and put through final linear layer 
        concat = attended.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
        
        #output = concat + value
        #output = output + value
        
        return output, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
        xavier_uniform(self.q_linear.weight)
        xavier_uniform(self.v_linear.weight)
        xavier_uniform(self.k_linear.weight)
        xavier_uniform(self.out.weight)

    def forward(self, v):

        k = v
        q = v
        
        bs = v.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention
        attended, scores = attention(q, k, v, self.d_k, self.dropout)

        # concatenate heads and put through final linear layer
        concat = attended.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output, scores

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 2500):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        self.pe =  nn.Parameter(torch.zeros(1, max_seq_len, d_model, requires_grad=True))
        xavier_uniform(self.pe)
        
        #for pos in range(max_seq_len):
        #    for i in range(0, d_model, 2):
        #        pe[pos, i] = sin(pos / (10000 ** ((2 * i)/d_model)))
        #        pe[pos, i + 1] = cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        #pe = pe.unsqueeze(0)
        #self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len]
        return x

class ConvAttnPool(BaseModel):

    def __init__(self, Y, dims, kernel_size, embed_matrix, gpu, vocab_size, Y_coarse=None, embed_size=100, embed_freeze=False, dropout=[0.5], hier=False, embed_desc=False, layer_norm=False, fine2coarse = None):
        super(ConvAttnPool, self).__init__(Y, embed_matrix, vocab_size, dropout=dropout[0], gpu=gpu, embed_size=dims[0], embed_freeze=embed_freeze, hier=hier)
        
        self.conv = ConvEncoder(dims[0], dims[1], kernel_size, padding=True)
        
        self.activation = nn.Tanh()
        
        self.attention = Attention(dims[1], Y, embed_desc, desc_dim=self.embed_size)
        xavier_uniform(self.attention.weight)
        
        self.layer_norm = nn.LayerNorm(torch.Size([dims[1]])) if layer_norm else None
        
        self.embed_desc = embed_desc
        
        self.final = nn.Linear(dims[1], Y, bias=True)
        xavier_uniform(self.final.weight)
        
        self.sigmoid = nn.Sigmoid()
        
        if self.hier:
            self.attention_coarse = Attention(dims[1], Y_coarse)
            xavier_uniform(self.attention_coarse.weight)
        
            self.final_coarse = nn.Linear(dims[1], Y_coarse, bias=True)
            xavier_uniform(self.final_coarse.weight)
            
            if fine2coarse is not None:
                self.fine2coarse = torch.LongTensor(fine2coarse)

    def forward(self, x, target, target_coarse=None, desc_data=None, get_attention=True):
        
        x = self.embed(x)
        
        x = self.embed_drop(x)
        
        x = self.activation(self.conv(x))
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        if self.hier:
            m_coarse, alpha_coarse = self.attention_coarse(x)
            yhat_coarse = self.final_coarse.weight.mul(m_coarse).sum(dim=2).add(self.final_coarse.bias)
            
            m, alpha = self.attention(x.detach(), self.embed(desc_data)) if self.embed_desc else self.attention(x.detach())
            yhat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
            
            mask = torch.round(self.sigmoid(yhat_coarse[:,self.fine2coarse]))
            yhat = yhat * mask
            
            loss = self._get_loss(yhat, target) + self._get_loss(yhat_coarse, target_coarse)
            
            yhat_coarse = self.sigmoid(yhat_coarse)
            yhat = self.sigmoid(yhat)
            
            return (yhat, yhat_coarse), loss, (alpha, alpha_coarse)
        else:    
            m, alpha = self.attention(x, self.embed(desc_data)) if self.embed_desc else self.attention(x)
            yhat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
            
            loss = self._get_loss(yhat, target)
            yhat = self.sigmoid(yhat)
            
            return yhat, loss, alpha
            
class HierarchicalConvAttn(BaseModel):

    def __init__(self, Y_fine, dims, kernel_size, embed_matrix, gpu, vocab_size, embed_freeze=False, dropout=[0.5, 0.5], hier=False, Y_coarse=None, fine2coarse=None, embed_desc=False, layer_norm=False):
        super(HierarchicalConvAttn, self).__init__(Y_fine, embed_matrix, vocab_size, dropout=dropout[0], gpu=gpu, embed_size=dims[0], embed_freeze=embed_freeze, hier=hier)

        assert dims[1]%2 == 0
        
        self.gru_words = GRUEncoder(dims[0], dims[1])
        
        self.activation = nn.Tanh()
        
        self.attention_words = Attention(dims[1], 1)
        xavier_uniform(self.attention_words.weight)
        
        self.sents_drop = nn.Dropout(p=dropout[1])
        
        self.conv_sents = ConvEncoder(dims[1], dims[2], kernel_size, padding=True)

        self.attention_sents_fine = Attention(dims[2], Y_fine, embed_desc, desc_dim=self.embed_size)
        xavier_uniform(self.attention_sents_fine.weight)
        
        self.final_fine = nn.Linear(dims[2], Y_fine, bias=True)
        xavier_uniform(self.final_fine.weight)
        
        self.embed_desc = embed_desc
        
        self.layer_norm_words = nn.LayerNorm(torch.Size([dims[1]])) if layer_norm else None
        self.layer_norm_sents = nn.LayerNorm(torch.Size([dims[2]])) if layer_norm else None

        if self.hier:
            self.attention_sents_coarse = Attention(dims[2], Y_coarse)
            xavier_uniform(self.attention_sents_coarse.weight)
        
            self.final_coarse = nn.Linear(dims[2], Y_coarse, bias=True)
            xavier_uniform(self.final_coarse.weight)
            
            if fine2coarse is not None:
                self.fine2coarse = torch.LongTensor(fine2coarse)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target_fine, target_coarse=None, target_cat=None, desc_data=None, get_attention=True):
        
        x = self.embed(x)
        x = self.embed_drop(x)
        
        s0, s1, s2, s3 = x.size()
        
        x = x.view(s0*s1, s2, s3)
        
        x = self.gru_words(x)
        x = self.activation(x)
        
        if self.layer_norm_words is not None:
            x = self.layer_norm_words(x)
        
        x, alpha_words = self.attention_words(x)
        
        x = torch.squeeze(x)
        
        x = x.view(s0, s1, -1)
        
        x = self.sents_drop(x)
        
        x = self.activation(self.conv_sents(x))
        
        if self.layer_norm_sents is not None:
            x = self.layer_norm_sents(x)
        
        m_fine, alpha_sents_fine = self.attention_sents_fine(x, self.embed(desc_data)) if self.embed_desc else self.attention_sents_fine(x)
        yhat_fine = self.final_fine.weight.mul(m_fine).sum(dim=2).add(self.final_fine.bias)
        
        if self.hier:
            m_coarse, alpha_sents_coarse = self.attention_sents_coarse(x)
            yhat_coarse = self.final_coarse.weight.mul(m_coarse).sum(dim=2).add(self.final_coarse.bias)

            loss = self._get_loss(yhat_fine, target_fine) + self._get_loss(yhat_coarse, target_coarse)
            
            yhat_coarse = self.sigmoid(yhat_coarse)
            yhat_fine = self.sigmoid(yhat_fine)
            
            return (yhat_fine, yhat_coarse), loss, (alpha_sents_fine, alpha_sents_coarse)
        else:
            loss = self._get_loss(yhat_fine, target_fine)
            
            yhat_fine = self.sigmoid(yhat_fine)
            
            return yhat_fine, loss, alpha_sents_fine
            
class ConvDilated(BaseModel):

    def __init__(self, Y, dims, kernel_size, dilation, embed_matrix, gpu, vocab_size, Y_coarse=None, embed_size=100, embed_freeze=False, dropout=[0.5], hier=False, embed_desc=False, layer_norm=False, fine2coarse = None):
        super(ConvDilated, self).__init__(Y, embed_matrix, vocab_size, dropout=.0, gpu=gpu, embed_size=dims[0], embed_freeze=embed_freeze, hier=hier)
        
        assert len(dropout) == len(dims)-1
        
        self.drops = nn.ModuleList([nn.Dropout(p=drop) for drop in dropout])
        
        self.convs = nn.ModuleList([nn.Conv1d(dims[i], dims[i+1], kernel_size=kernel_size, dilation=dilation[i], padding=floor(kernel_size/2)*dilation[i]) for i in range(len(self.drops))])
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(torch.Size([dims[i+1]])) for i in range(0,len(self.drops))])
        
        self.activation = nn.Tanh()
        
        self.attention = Attention(dims[-1], Y, embed_desc, desc_dim=self.embed_size)
        xavier_uniform(self.attention.weight)
        
        self.embed_desc = embed_desc
        
        self.final = nn.Linear(dims[-1], Y, bias=True)
        xavier_uniform(self.final.weight)
        
        self.sigmoid = nn.Sigmoid()
        
        if self.hier:
            self.attention_coarse = Attention(dims[-1], Y_coarse)
            xavier_uniform(self.attention_coarse.weight)
        
            self.final_coarse = nn.Linear(dims[-1], Y_coarse, bias=True)
            xavier_uniform(self.final_coarse.weight)
            
            if fine2coarse is not None:
                self.fine2coarse = torch.LongTensor(fine2coarse)

    def forward(self, x, target, target_coarse=None, desc_data=None, get_attention=True):
        
        x = self.embed(x)
        
        for i in range(len(self.convs)):
            x = self.drops[i](x)
            x = x.transpose(1,2)
            x = self.convs[i](x)
            x = x.transpose(1,2)
            x = self.activation(x)
            x = self.layer_norms[i](x)

        if self.hier:
            m_coarse, alpha_coarse = self.attention_coarse(x)
            yhat_coarse = self.final_coarse.weight.mul(m_coarse).sum(dim=2).add(self.final_coarse.bias)
            
            m, alpha = self.attention(x.detach(), self.embed(desc_data)) if self.embed_desc else self.attention(x.detach())
            yhat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
            
            mask = torch.round(self.sigmoid(yhat_coarse[:,self.fine2coarse]))
            yhat = yhat * mask
            
            loss = self._get_loss(yhat, target) + self._get_loss(yhat_coarse, target_coarse)
            
            yhat_coarse = self.sigmoid(yhat_coarse)
            yhat = self.sigmoid(yhat)
            
            return (yhat, yhat_coarse), loss, (alpha, alpha_coarse)
        else:    
            m, alpha = self.attention(x, self.embed(desc_data)) if self.embed_desc else self.attention(x)
            yhat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
            
            loss = self._get_loss(yhat, target)
            yhat = self.sigmoid(yhat)
            
            return yhat, loss, alpha
            