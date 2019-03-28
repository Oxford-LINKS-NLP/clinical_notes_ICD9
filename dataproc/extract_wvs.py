"""
    Use the vocabulary to load a matrix of pre-trained word vectors
"""
import csv
import os
import gensim.models
from tqdm import tqdm

from constants import *
import datasets

import numpy as np

def gensim_to_embeddings(wv_file, vocab_file, Y, outfile=None):
    model = gensim.models.Word2Vec.load(wv_file)
    wv = model.wv
    #free up memory
    del model

    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}

    W, words = build_matrix(ind2w, wv)

    if outfile is None:
        outfile = wv_file.replace('.w2v', '.embed')

    #smash that save button
    save_embeddings(W, words, outfile)

def build_matrix(ind2w, wv):
    """
        Go through vocab in order. Find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i]).
        Put results into one big matrix.
        Note: ind2w starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    """
    W = np.zeros((len(ind2w)+1, len(wv.word_vec(wv.index2word[0])) ))
    words = [PAD_CHAR]
    W[0][:] = np.zeros(len(wv.word_vec(wv.index2word[0])))
    for idx, word in tqdm(ind2w.items()):
        if idx >= W.shape[0]:
            break    
        W[idx][:] = wv.word_vec(word)
        words.append(word)
    return W, words

def save_embeddings(W, words, outfile):
    with open(outfile, 'w') as o:
        #pad token already included
        for i in range(len(words)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")

def load_embeddings(embed_file, ind2w, embed_size, embed_normalize):
    #also normalizes the embeddings
    word_embeddings = {}
    vocab_size = len(ind2w)
    
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            idx = len(line) - embed_size
            word = '_'.join(line[:idx]).lower().strip()
            vec = np.array(line[idx:]).astype(np.float)
            word_embeddings[word] = vec

    W = np.zeros((vocab_size+2, embed_size))
    words_found = 0
    
    for ind, word in ind2w.items():

        try: 
            W[ind] = word_embeddings[word]
            words_found += 1
        except KeyError:
            W[ind] = np.random.randn(1, embed_size)
        if embed_normalize:
            W[ind] = W[ind] / (np.linalg.norm(W[ind]) + 1e-6)

    W[vocab_size-1] = np.random.randn(1, embed_size)
    
    if embed_normalize:
        W[vocab_size-1] = W[vocab_size-1] / (np.linalg.norm(W[vocab_size-1]) + 1e-6)

    print('vocabulary coverage: {}'.format(words_found/vocab_size))
    
    return W

