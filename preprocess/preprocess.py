import pandas as pd
from clarityNLP.segmentation import Tokenizer
import sys
import os
import ndjson
import argparse

parser = argparse.ArgumentParser(description='Tokenize MIMIC-III clinical notes')
parser.add_argument('input_file', type=str, help='input file path')
parser.add_argument('output_folder', type=str, help='output folder path')
parser.add_argument('--base-batch-size', type=int, required=False, dest='base_batch_size', default=100,
							help='number of documents to process for each core (default: 100)')
parser.add_argument('--n-threads', type=int, required=False, dest='n_threads', default=4,
							help='number of threads per core (default: 4)')
args = parser.parse_args()

output_file = os.path.join(args.output_folder + 'notes_tokenized.ndjson')
n_cpus =  os.cpu_count()
n_threads = n_cpus * args.n_threads
batch_size = args.base_batch_size * n_cpus

MODE = 2

def process_batch(notes_batch, i, seg_obj, notes_tokenized_file):
	print('processing batch {0}, batchsize {1}'.format(i, batch_size))
	
	documents = tokenizer.tokenize_documents(notes_batch['TEXT'])
	n_tokens = [sum(len(sentence) for sentence in doc) for doc in documents]
	rows = list(zip(notes_batch['HADM_ID'].fillna(-1).astype('int32'), notes_batch['SUBJECT_ID'], notes_batch['CATEGORY'], notes_batch['DESCRIPTION'], notes_batch['ISERROR'].fillna(0).astype(bool), n_tokens, documents))
	#print('\n'.join(token for doc in documents for sentence in doc for token in sentence))

	ndjson.dump(rows, notes_tokenized_file)
	notes_tokenized_file.write('\n')

csv_reader = pd.read_csv(args.input_file, chunksize=batch_size, usecols=['HADM_ID', 'SUBJECT_ID', 'CATEGORY', 'DESCRIPTION', 'ISERROR','TEXT'], dtype={'HADM_ID': 'str', 'SUBJECT_ID':'int32', 'CATEGORY': 'str', 'DESCRIPTION':'str', 'ISERROR':'str', 'TEXT':'str'}, keep_default_na=False, na_values='')

tokenizer = Tokenizer(args.base_batch_size, n_cpus, n_threads, MODE)

with open(output_file, 'x') as notes_tokenized_file:

	for i, notes_batch in enumerate(csv_reader):
		process_batch(notes_batch, i, seg_obj, notes_tokenized_file)
