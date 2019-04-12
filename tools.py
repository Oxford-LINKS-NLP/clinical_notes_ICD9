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
        print("loading pretrained embeddings (freeze={0}, normalize={1})...".format(args.embed_freeze, args.embed_normalize))            
        word_embeddings_matrix = load_embeddings(args.embed_file, dicts['ind2w'], args.dims[0], args.embed_normalize)
    else:
        word_embeddings_matrix = None
        
    vocab_size = len(dicts['ind2w'])
                    
    if args.model == "dummy":
        model = models.DummyModel(Y, dicts, args.gpu)
    elif args.model == "conv_attn":
        model = models.ConvAttnPool(Y, args.dims, args.filter_size, word_embeddings_matrix, args.gpu, vocab_size,
                                    embed_freeze=args.embed_freeze, dropout=args.dropout,
                                    hier=args.hier, Y_coarse=Y_coarse, fine2coarse=dicts['fine2coarse'],
                                    embed_desc=args.embed_desc, layer_norm=args.layer_norm)
    elif args.model == "hier_conv_attn":
        model = models.HierarchicalConvAttn(Y, args.dims, args.filter_size, word_embeddings_matrix, args.gpu, vocab_size,
                                    embed_freeze=args.embed_freeze, dropout=args.dropout,
                                    hier=args.hier, Y_coarse=Y_coarse, fine2coarse=dicts['fine2coarse'],
                                    embed_desc=args.embed_desc, layer_norm=args.layer_norm)

    if args.test_model:
        sd = torch.load(os.path.abspath(args.test_model))
        model.load_state_dict(sd)

    if args.gpu:
        model.cuda()

    return model
