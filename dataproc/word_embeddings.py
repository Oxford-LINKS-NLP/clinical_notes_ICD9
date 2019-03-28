"""
    Pre-train embeddings using gensim w2v implementation (CBOW by default)
"""
import gensim.models.word2vec as w2v
import csv
import jsonlines

from constants import *

class ProcessedIter(object):

    def __init__(self, Y, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield (row[3].split())

def word_embeddings(Y, notes_file, embedding_size, min_count, n_iter, skipgram=False, workers=8):
    if skipgram:
        mode='skipgram'
    else:
        mode='cbow'
    modelname = "processed_%s_%s.w2v" % (Y, mode)
    sentences = ProcessedIter(Y, notes_file)

    model = w2v.Word2Vec(size=embedding_size, min_count=min_count, sg=int(skipgram), workers=workers, iter=n_iter)
    print("building word2vec vocab on %s..." % (notes_file))
    
    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file
 
class NotesIter(object):

    def __init__(self, corpus_size, filename):
        self.filename = filename
        self.idx = 7 if corpus_size == 'full' else 2
        
    def __iter__(self):
        with jsonlines.open(self.filename) as reader:
            for note in reader:
                for sentence in note[self.idx]:
                    yield sentence
    
def word2vec(corpus_size, notes_file, out_dir, embedding_size, n_iter, skipgram=False, workers=8):
    
    mode = 'skipgram' if skipgram else 'cbow'

    modelname = 'word_embeddings_{}_{}_{}.w2v'.format(corpus_size, mode, n_iter)
    sentences = NotesIter(corpus_size, notes_file)

    model = w2v.Word2Vec(size=embedding_size, min_count=0, sg=int(skipgram), workers=workers, iter=n_iter)
    print("building word2vec vocab on %s..." % (notes_file))
    
    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    out_file =  '{}/{}'.format(out_dir, modelname)
    print("writing embeddings to %s" % (out_file))
    model.wv.save_word2vec_format(out_file, binary=False)
    
    with open(out_file, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(out_file, 'w') as fout:
        fout.writelines(data[1:])
    
    return out_file

