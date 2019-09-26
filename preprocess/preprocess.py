import pandas as pd
from tokenizer.tokenizer import Tokenizer
import os
import jsonlines
import argparse
from numpy import int32

MODE = 2

def process_batch(notes_batch, i, tokenizer, notes_tokenized_file):
	print('processing batch {0}, batchsize {1}'.format(i, args.batch_size))

	documents = tokenizer.tokenize_documents(notes_batch['TEXT'])
	n_tokens = [sum(len(sentence) for sentence in doc) for doc in documents]
	rows = list(zip(notes_batch['SUBJECT_ID'], notes_batch['HADM_ID'].fillna(-1).astype(int32), notes_batch['CHARTDATE'].fillna('1970-01-01'), notes_batch['CATEGORY'], notes_batch['DESCRIPTION'], notes_batch['ISERROR'].fillna(0).astype(bool), n_tokens, documents))
	#print('\n'.join(token for doc in documents for sentence in doc for token in sentence))

	for row in rows:
		notes_tokenized_file.write(row)

def main(args):
	if os.path.exists(args.output_file):
		raise FileExistsError('File exists: {}'.format(args.output_file))
	csv_reader = pd.read_csv(args.input_file, chunksize=args.batch_size, usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY', 'DESCRIPTION', 'ISERROR','TEXT'], dtype={'SUBJECT_ID':int32, 'HADM_ID': 'str', 'CATEGORY': 'str', 'DESCRIPTION':'str', 'ISERROR':'str', 'TEXT':'str'}, keep_default_na=False, na_values='')
	
	with jsonlines.open(args.output_file, 'w') as notes_tokenized_file:
		tokenizer = Tokenizer(args.batch_size, args.n_cpus, args.n_threads, MODE)
		
		for i, notes_batch in enumerate(csv_reader):
			process_batch(notes_batch, i, tokenizer, notes_tokenized_file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Tokenize MIMIC-III clinical notes')
	parser.add_argument('input_file', type=str, help='input file path')
	parser.add_argument('output_folder', type=str, help='output folder path')
	parser.add_argument('--batch-size', type=int, required=False, dest='batch_size', default=100,
								help='number of documents to process for each batch (default: 100)')
	parser.add_argument('--n-threads', type=int, required=False, dest='n_threads', default=4,
								help='number of threads per core (default: 4)')
	args = parser.parse_args()

	args.output_file = os.path.join(args.output_folder + 'notes_tokenized.ndjson')
	args.n_cpus =  os.cpu_count()
	args.n_threads = args.n_cpus * args.n_threads
	
	main(args)
