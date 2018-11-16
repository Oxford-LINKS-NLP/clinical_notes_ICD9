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

global MODE

MODE = int(sys.argv[1])

if MODE < 0 or MODE > 2:
	sys.exit('invalid mode number (must be 0, 1 or 2)')

PATH_IN = sys.argv[2]
PATH_OUT = sys.argv[3]

CHUNKSIZE = 100*n_cpus

def process_chunk(notes_chunk, i, seg_obj, notes_sentences_handle):
	print('processing chunk ' + str(i) + ', chunksize ' + str(CHUNKSIZE))
	
	documents = seg_obj.parse_documents(notes_chunk['TEXT'], CHUNKSIZE, n_cpus, n_threads)
	rows = list(zip(notes_chunk['HADM_ID'], documents))
	ndjson.dump(rows, notes_sentences_handle)
	notes_sentences_handle.write('\n')

csv_reader = pd.read_csv(PATH_IN + 'NOTEEVENTS.csv', chunksize=CHUNKSIZE)#, usecols=['HADM_ID', 'TEXT'], dtype={'HADM_ID': 'uint32', 'TEXT': 'object'})

seg_obj = Segmentation(MODE)

with open(PATH_OUT + 'sentences/notes_sentences' + '.csv', 'w') as notes_sentences_handle:

	for i, notes_chunk in enumerate(csv_reader):
		process_chunk(notes_chunk, i, seg_obj, notes_sentences_handle)
	notes_sentences_handle.close()
