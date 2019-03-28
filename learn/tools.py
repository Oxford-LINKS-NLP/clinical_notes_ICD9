"""
    Various utility methods
"""
import csv
import json
import math
import os
import pickle

import torch
from torch.autograd import Variable

from learn import models
from constants import *
import datasets
import persistence
import numpy as np

from dataproc.extract_wvs import load_embeddings

def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """

    Y = len(dicts['ind2c'])
    Y_coarse = len(dicts['ind2c_coarse']) if args.hier else None
    
    if args.embed_file and not args.test_model:
        print("loading pretrained embeddings (trainable={0}, normalize={1})...".format(args.embed_trainable, args.embed_normalize))            
        word_embeddings_matrix = load_embeddings(args.embed_file, dicts['ind2w'], args.embed_size, args.embed_normalize)
    else:
        word_embeddings_matrix = None
        
    vocab_size = len(dicts['ind2w'])
                    
    if args.model == "rnn":
        model = models.VanillaRNN(Y, word_embeddings_matrix, vocab_size, args.rnn_dim, args.cell_type, args.rnn_layers, args.gpu, args.embed_size,
                                  args.bidirectional, embed_trainable=args.embed_trainable)
    elif args.model == "dummy":
        model = models.DummyModel(Y, dicts, args.gpu)
    elif args.model == "cnn_vanilla":
        filter_size = int(args.filter_size)
        model = models.VanillaConv(Y, word_embeddings_matrix, filter_size, args.num_filter_maps, args.gpu, vocab_size, args.embed_size, args.dropout,
                                   embed_trainable=args.embed_trainable)
    elif args.model == "conv_attn":
        filter_size = int(args.filter_size)
        model = models.ConvAttnPool(Y, word_embeddings_matrix, filter_size, args.num_filter_maps, args.gpu, vocab_size,
                                    embed_size=args.embed_size, embed_trainable=args.embed_trainable, dropout=args.dropout, hier=args.hier, Y_coarse=Y_coarse, embed_desc=args.embed_desc, layer_norm=args.layer_norm, fine2coarse=dicts['fine2coarse'])
    elif args.model == "conv_attn_old":
        filter_size = int(args.filter_size)
        model = models.ConvAttnPool_old(Y, word_embeddings_matrix, filter_size, args.num_filter_maps, args.gpu, vocab_size,
                                    embed_size=args.embed_size, embed_trainable=args.embed_trainable, dropout=args.dropout)
    elif args.model == "hier_conv_attn":
        model = models.HierarchicalConvAttn(Y, word_embeddings_matrix,
                                    args.filter_size_words, args.num_filter_maps_words, args.filter_size_sents, args.num_filter_maps_sents,
                                    args.gpu, vocab_size, embed_size=args.embed_size, embed_trainable=args.embed_trainable, dropout_words=args.dropout, dropout_sents=args.dropout_sents, hier=args.hier, Y_coarse=Y_coarse, fine2coarse=dicts['fine2coarse'], layer_norm=args.layer_norm, embed_desc=args.embed_desc)

    if args.test_model:
        sd = torch.load(os.path.abspath(args.test_model))
        model.load_state_dict(sd)

    if args.gpu:
        model.cuda()

    return model
