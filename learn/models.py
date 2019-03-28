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

from constants import *

def attention(q, k, v, d_k, dropout=None): 
          
    scores = torch.matmul(q, k.transpose(-2, -1)) /  sqrt(d_k) 
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    return torch.matmul(scores, v), scores

class BaseModel(nn.Module):

    def __init__(self, Y, word_embeddings_matrix, vocab_size, dropout=0.5, gpu=True, embed_size=100, embed_trainable=False, hier=False):
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
            self.embed = nn.Embedding.from_pretrained(W, freeze=(not embed_trainable))
            self.embed.padding_idx = 0
        else:
            self.embed = nn.Embedding(vocab_size+2, embed_size, padding_idx=0)

    def _get_loss(self, yhat, target):
        #calculate the BCE011
        return F.binary_cross_entropy_with_logits(yhat, target)

class Attention(torch.nn.Module):
    def __init__(self, embed_size, num_labels, embed_desc=False, kernel_size=5, input_size=100):
        kernel_size=3
        super(Attention, self).__init__()
        
        self.embed_desc = embed_desc
        self.softmax = nn.Softmax(dim=2)
        
        if self.embed_desc:
            self.label_conv = nn.Conv1d(input_size, embed_size, kernel_size=kernel_size, padding=floor(kernel_size/2))
            #self.tanh = nn.Tanh()
            #self.linear = nn.Linear(embed_size, embed_size)
            #xavier_uniform(self.linear.weight)
            self.weight = self.label_conv.weight
        else:
            self.U = nn.Linear(embed_size, num_labels, bias=False)
            self.weight = self.U.weight

    def forward(self, x, desc_data=None):

        if self.embed_desc:
            desc_data = desc_data.transpose(1,2)
            desc_data = self.label_conv(desc_data)
            #desc_data = self.tanh(desc_data)
            #desc_data = F.max_pool1d(torch.tanh(desc_data), kernel_size=desc_data.size(2)).transpose(1,2)
            desc_data = F.avg_pool1d(torch.tanh(desc_data), kernel_size=desc_data.size(2)).transpose(1,2)
            desc_data = torch.squeeze(desc_data)
            #desc_data = self.linear(desc_data)
            alpha = self.softmax(desc_data.matmul(x.transpose(1,2)))
        else:
            alpha = self.softmax(self.U(x).transpose(1,2))

        return alpha.matmul(x), alpha


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

class HierarchicalConvAttn(BaseModel):

    def __init__(self, Y_fine, embed_file, kernel_size_words, num_filter_maps_words, kernel_size_sents, num_filter_maps_sents, gpu, vocab_size, Y_coarse=None, embed_size=100, embed_trainable=False, dropout_words=0.5, dropout_sents=0.5, hier=False, fine2coarse=None, embed_desc=False, layer_norm=False):
        super(HierarchicalConvAttn, self).__init__(Y_fine, embed_file, vocab_size, dropout=dropout_words, gpu=gpu, embed_size=embed_size, embed_trainable=embed_trainable, hier=hier)

        self.conv_words = nn.Conv1d(self.embed_size, num_filter_maps_words, kernel_size=kernel_size_words)
        xavier_uniform(self.conv_words.weight)
        
        self.tanh = nn.Tanh()
        
        self.attention_words = Attention(num_filter_maps_words, 1)
        xavier_uniform(self.attention_words.weight)
        
        self.sents_drop = nn.Dropout(p=dropout_sents)
        
        self.conv_sents = nn.Conv1d(num_filter_maps_words, num_filter_maps_sents, kernel_size=kernel_size_sents, padding=floor(kernel_size_sents/2))
        xavier_uniform(self.conv_sents.weight)

        self.attention_sents_fine = Attention(num_filter_maps_sents, Y_fine, embed_desc, kernel_size=kernel_size_words, input_size=embed_size)
        xavier_uniform(self.attention_sents_fine.weight)
        
        self.final_fine = nn.Linear(num_filter_maps_sents, Y_fine, bias=True)
        xavier_uniform(self.final_fine.weight)
        
        self.embed_desc = embed_desc
        
        self.layer_norm_words = nn.LayerNorm(torch.Size([num_filter_maps_words])) if layer_norm else None
        self.layer_norm_sents = nn.LayerNorm(torch.Size([num_filter_maps_sents])) if layer_norm else None

        if self.hier:
            self.attention_sents_coarse = Attention(num_filter_maps_sents, Y_coarse)
            xavier_uniform(self.attention_sents_coarse.weight)
        
            self.final_coarse = nn.Linear(num_filter_maps_sents, Y_coarse, bias=True)
            xavier_uniform(self.final_coarse.weight)
            
            if fine2coarse is not None:
                self.fine2coarse = torch.LongTensor(fine2coarse)
            
            #self.drop_cat = nn.Dropout(p=0.5)
            
            #self.final_cat = nn.Linear(Y_coarse + Y_fine, Y_coarse + Y_fine, bias=True)
            #xavier_uniform(self.final_coarse.weight)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target_fine, target_coarse=None, target_cat=None, desc_data=None, get_attention=True):
        
        x = self.embed(x)
        x = self.embed_drop(x)
        
        s0, s1, s2, s3 = x.size()
        
        x = x.view(s0*s1, s2, s3)
        
        x = x.transpose(1, 2)

        x = self.tanh(self.conv_words(x).transpose(1,2))
        
        if self.layer_norm_words is not None:
            x = self.layer_norm_words(x)
        
        x, alpha_words = self.attention_words(x)
        
        x = torch.squeeze(x)
        
        x = x.view(s0, s1, self.conv_words.out_channels)
        
        x = self.sents_drop(x)
        
        x = x.transpose(1, 2)
        
        x = self.tanh(self.conv_sents(x).transpose(1,2))
        
        if self.layer_norm_sents is not None:
            x = self.layer_norm_sents(x)
        
        m_fine, alpha_sents_fine = self.attention_sents_fine(x, self.embed(desc_data)) if self.embed_desc else self.attention_sents_fine(x)
        yhat_fine = self.final_fine.weight.mul(m_fine).sum(dim=2).add(self.final_fine.bias)
        
        if self.hier:
            m_coarse, alpha_sents_coarse = self.attention_sents_coarse(x)
            yhat_coarse = self.final_coarse.weight.mul(m_coarse).sum(dim=2).add(self.final_coarse.bias)
            
            #yhat_cat = self.drop_cat(torch.cat([yhat_coarse, yhat_fine], dim=1))
            #yhat_cat  = self.final_cat(yhat_cat)
            #loss = self._get_loss(yhat_cat, target_cat)
            #yhat_split = torch.split(self.sigmoid(yhat_cat), [yhat_coarse.size(1), yhat_fine.size(1)], dim=1)
            #return (yhat_split[1], yhat_split[0]) , loss, (alpha_sents_fine, alpha_sents_coarse)

            #yhat_fine = yhat_fine.mul(yhat_coarse[:,self.fine2coarse])

            loss = self._get_loss(yhat_fine, target_fine) + self._get_loss(yhat_coarse, target_coarse)
            
            yhat_coarse = self.sigmoid(yhat_coarse)
            yhat_fine = self.sigmoid(yhat_fine)
            
            return (yhat_fine, yhat_coarse), loss, (alpha_sents_fine, alpha_sents_coarse)
        else:
            loss = self._get_loss(yhat_fine, target_fine)
            
            yhat_fine = self.sigmoid(yhat_fine)
            
            return yhat_fine, loss, alpha_sents_fine


class ConvAttnPool(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu, vocab_size, Y_coarse=None, embed_size=100, embed_trainable=False, dropout=0.5, hier=False, embed_desc=False, layer_norm=False, fine2coarse = None):
        super(ConvAttnPool, self).__init__(Y, embed_file, vocab_size, dropout=dropout, gpu=gpu, embed_size=embed_size, embed_trainable=embed_trainable, hier=hier)
            
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=floor(kernel_size/2))
        xavier_uniform(self.conv.weight)
        
        self.tanh = nn.Tanh()
        
        self.attention = Attention(num_filter_maps, Y, embed_desc, kernel_size=kernel_size, input_size=embed_size)
        xavier_uniform(self.attention.weight)
        
        self.layer_norm = nn.LayerNorm(torch.Size([num_filter_maps])) if layer_norm else None
        
        self.embed_desc = embed_desc
        
        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y, bias=True)
        xavier_uniform(self.final.weight)
        
        self.sigmoid = nn.Sigmoid()
        
        if self.hier:
            self.attention_coarse = Attention(num_filter_maps, Y_coarse)
            xavier_uniform(self.attention_coarse.weight)
        
            self.final_coarse = nn.Linear(num_filter_maps, Y_coarse, bias=True)
            xavier_uniform(self.final_coarse.weight)
            
            if fine2coarse is not None:
                self.fine2coarse = torch.LongTensor(fine2coarse)

    def forward(self, x, target, target_coarse=None, desc_data=None, get_attention=True):
        
        #get embeddings and apply dropout
        x = self.embed(x)
        
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #apply convolution and nonlinearity (tanh)
        x = self.tanh(self.conv(x).transpose(1,2))
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        m, alpha = self.attention(x, self.embed(desc_data)) if self.embed_desc else self.attention(x)
        
        #desc_data = self.embed(desc_data)
        #desc_data = desc_data.transpose(1,2)
        #desc_data = self.conv(desc_data)
        #desc_data = F.max_pool1d(torch.tanh(desc_data), kernel_size=desc_data.size(2)).transpose(1,2)
        #desc_data = torch.squeeze(desc_data).unsqueeze(0)
        #alpha = F.softmax(desc_data.matmul(x.transpose(1,2)), dim=2)

        #m = alpha.matmul(x)
        
        yhat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        
        if self.hier:
            m_coarse, alpha_coarse = self.attention_coarse(x)
            yhat_coarse = self.final_coarse.weight.mul(m_coarse).sum(dim=2).add(self.final_coarse.bias)
            
            lmbda = 0.001
            loss = self._get_loss(yhat, target) + self._get_loss(yhat_coarse, target_coarse) + lmbda*(1/2)*torch.sum(torch.norm(self.attention_coarse.weight[self.fine2coarse] - self.attention.weight, 2) ** 2) 
            
            yhat_coarse = self.sigmoid(yhat_coarse)
            yhat = self.sigmoid(yhat)
            
            return (yhat, yhat_coarse), loss, (alpha, alpha_coarse)
        else:
            loss = self._get_loss(yhat, target)
            
            yhat = self.sigmoid(yhat)
            
            return yhat, loss, alpha
        
class ConvAttnPool_old(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu, vocab_size, embed_size=100, embed_trainable=False, dropout=0.5):
        super(ConvAttnPool_old, self).__init__(Y, embed_file, vocab_size, dropout=dropout, gpu=gpu, embed_size=embed_size, embed_trainable=embed_trainable)

        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform(self.conv.weight)

        #context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)
        
    def forward(self, x, target, desc_data=None, get_attention=True):
        #get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #apply convolution and nonlinearity (tanh)
        x = torch.tanh(self.conv(x).transpose(1,2))
        #apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
        #document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        #final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
            
        #final sigmoid to get predictions
        yhat = y
        loss = self._get_loss(yhat, target)
        yhat = torch.sigmoid(yhat)
        return yhat, loss, alpha
    
class VanillaConv(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=True, vocab_size=0, embed_size=100, embed_trainable=False, dropout=0.5):
        super(VanillaConv, self).__init__(Y, embed_file, vocab_size, dropout=dropout, embed_size=embed_size, embed_trainable=embed_trainable) 
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #conv/max-pooling
        c = self.conv(x)
        if get_attention:
            #get argmax vector too
            x, argmax = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            x = F.max_pool1d(torch.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)

        #linear output
        x = self.fc(x)

        #final sigmoid to get predictions
        yhat = torch.sigmoid(x)
        loss = self._get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                #generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1,self.Y).t()
                #apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1,self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    #this window was never a max
                    attns.append(torch.zeros(self.Y).cuda())
            #combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        #put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1,2)
        return attn_full
        
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


class VanillaRNN(BaseModel):
    """
        General RNN - can be LSTM or GRU, uni/bi-directional
    """

    def __init__(self, Y, embed_file, vocab_size, rnn_dim, cell_type, num_layers, gpu, embed_size=100, embed_trainable=False, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, vocab_size, embed_size=embed_size, embed_trainable=embed_trainable, gpu=gpu)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        #recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, floor(self.rnn_dim/self.num_directions), self.num_layers, bidirectional=bidirectional)
        #linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        #arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        #clear hidden state, reset batch size at the start of each batch
        self.refresh(x.size()[0])

        #embed
        embeds = self.embed(x).transpose(0,1)
        #apply RNN
        out, self.hidden = self.rnn(embeds, self.hidden)

        #get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0,1).contiguous().view(self.batch_size, -1)
        #apply linear layer and sigmoid to get predictions
        yhat = torch.sigmoid(self.final(last_hidden))
        loss = self._get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu:
            h_0 = torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                  floor(self.rnn_dim/self.num_directions)).zero_()
            if self.cell_type == 'lstm':
                c_0 = torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                      floor(self.rnn_dim/self.num_directions)).zero_()
                return (h_0, c_0)
            else:
                return h_0
        else:
            h_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions))
            if self.cell_type == 'lstm':
                c_0 = torch.zeros(self.num_directions*self.num_layers, self.batch_size, floor(self.rnn_dim/self.num_directions))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()
        

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
        
        self.tanh = nn.Tanh()
        
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
        
        k = self.tanh(k)
        q = self.tanh(q)
        v = self.tanh(v)
        
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
        
        self.tanh = nn.Tanh()
  
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

        k = self.tanh(k)
        q = self.tanh(q)
        v = self.tanh(v)
          
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
        
class ConvAttnPool_(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu, vocab_size, Y_coarse=None, embed_size=100, embed_trainable=False, dropout=0.5, hier=False, embed_desc=False, layer_norm=False, fine2coarse=None):
        super(ConvAttnPool, self).__init__(Y, embed_file, vocab_size, dropout=dropout, gpu=gpu, embed_size=embed_size, embed_trainable=embed_trainable, hier=hier)

        #initialize conv layer as in 2.1
        #self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=floor(kernel_size/2))
        #xavier_uniform(self.conv.weight)
        
        #self.pos_encoder = PositionalEncoder(embed_size, max_seq_len=3200)
        #self.conv_attention_multi = MultiHeadConvAttention(1, embed_size, num_filter_maps)
        self.conv_attention_multi = MultiHeadConvTargetAttention(Y, 1, embed_size, num_filter_maps)
        
        self.tanh = nn.Tanh()
        
        #self.attention = Attention(num_filter_maps, Y, embed_desc, kernel_size=kernel_size, input_size=embed_size)
        #xavier_uniform(self.attention.weight)
        
        self.layer_norm = nn.LayerNorm(torch.Size([num_filter_maps])) if layer_norm else None
        
        self.embed_desc = embed_desc
        
        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y, bias=True)
        xavier_uniform(self.final.weight)
        
        self.sigmoid = nn.Sigmoid()
        
        if self.hier:
            self.attention_coarse = Attention(num_filter_maps, Y_coarse)
            xavier_uniform(self.attention_coarse.weight)
        
            self.final_coarse = nn.Linear(num_filter_maps, Y_coarse, bias=True)
            xavier_uniform(self.final_coarse.weight)

    def forward(self, x, target, target_coarse=None, desc_data=None, get_attention=True):
        
        #get embeddings and apply dropout
        x = self.embed(x)
        
        #x = self.pos_encoder(x)
        
        x = self.embed_drop(x)
        #x = x.transpose(1, 2)

        #apply convolution and nonlinearity (tanh)
        #x = self.tanh(self.conv(x).transpose(1,2))
        
        x, alpha = self.conv_attention_multi(x)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        #m, alpha = self.attention(x, self.embed(desc_data)) if self.embed_desc else self.attention(x)
        m = x
        
        #desc_data = self.embed(desc_data)
        #desc_data = desc_data.transpose(1,2)
        #desc_data = self.conv(desc_data)
        #desc_data = F.max_pool1d(torch.tanh(desc_data), kernel_size=desc_data.size(2)).transpose(1,2)
        #desc_data = torch.squeeze(desc_data).unsqueeze(0)
        #alpha = F.softmax(desc_data.matmul(x.transpose(1,2)), dim=2)

        #m = alpha.matmul(x)
        
        yhat = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        
        if self.hier:
            m_coarse, alpha_coarse = self.attention_coarse(x)
            yhat_coarse = self.final_coarse.weight.mul(m_coarse).sum(dim=2).add(self.final_coarse.bias)
                          
            loss = self._get_loss(yhat, target) + self._get_loss(yhat_coarse, target_coarse)
            
            yhat_coarse = self.sigmoid(yhat_coarse)
            yhat = self.sigmoid(yhat)
            
            return (yhat, yhat_coarse), loss, (alpha, alpha_coarse)
        else:
            loss = self._get_loss(yhat, target)
            
            yhat = self.sigmoid(yhat)
            
            return yhat, loss, alpha
