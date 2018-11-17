import pandas as pd
#import multiprocessing as mp
from clarityNLP.segmentation import Segmentation
import csv
import sys
import os

import ndjson

n_cpus =  os.cpu_count()
n_threads = n_cpus * 4

if len(sys.argv) != 4:
	sys.exit('invalid number of arguments')

MODE = int(sys.argv[1])

if MODE < 0 or MODE > 2:
	sys.exit('invalid mode number (must be 0, 1 or 2)')

PATH_IN = sys.argv[2]
PATH_OUT = sys.argv[3]

CHUNKSIZE = 1*n_cpus

def process_chunk(notes_chunk, i, seg_obj, notes_sentences_handle):
	print('processing chunk {0}, chunksize {1}'.format(i, CHUNKSIZE))
	
	documents = seg_obj.parse_documents(notes_chunk['TEXT'], CHUNKSIZE, n_cpus, n_threads)
	n_tokens = [sum(len(sentence) for sentence in doc) for doc in documents]
	rows = list(zip(notes_chunk['HADM_ID'].fillna(-1).astype('int32'), notes_chunk['SUBJECT_ID'], notes_chunk['CATEGORY'], notes_chunk['DESCRIPTION'], notes_chunk['ISERROR'].fillna(0).astype(bool), n_tokens, documents))
	print('\n'.join(token for doc in documents for sentence in doc for token in sentence))

	ndjson.dump(rows, notes_sentences_handle)
	notes_sentences_handle.write('\n')

csv_reader = pd.read_csv(PATH_IN + 'NOTEEVENTS.csv', usecols=['HADM_ID', 'SUBJECT_ID', 'CATEGORY', 'DESCRIPTION', 'ISERROR','TEXT'], dtype={'HADM_ID': 'str', 'SUBJECT_ID':'int32', 'CATEGORY': 'str', 'DESCRIPTION':'str', 'ISERROR':'str', 'TEXT':'str'}, keep_default_na=False, na_values='', chunksize=CHUNKSIZE)

seg_obj = Segmentation(MODE)

with open(PATH_OUT + 'sentences/notes_sentences' + '.ndjson', 'w') as notes_sentences_handle:

	for i, notes_chunk in enumerate(csv_reader):
		process_chunk(notes_chunk, i, seg_obj, notes_sentences_handle)
	notes_sentences_handle.close()
