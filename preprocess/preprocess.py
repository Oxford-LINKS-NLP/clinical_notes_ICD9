import pandas as pd
import multiprocessing as mp
from clarityNLP.segmentation import Segmentation
import pickle
import sys

PATH_IN = sys.argv[1]
PATH_OUT = sys.argv[2]

CHUNKSIZE = 100*32

def process_note(seg_obj, hadm_id, text, notes_sentences_pickler, notes_words_handle):
	
	sentence_list, word_list = seg_obj.parse_sentences(text)
	
	notes_sentences_pickler.dump((hadm_id, sentence_list))
	#flat_word_list = [word.lower() for sentence in word_list for word in sentence]
	[notes_words_handle.write('%s\n' % word.lower()) for sentence in word_list for word in sentence]

	return sentence_list, word_list

def process_chunk(notes_chunk, i, seg_obj):
	print('processing chunk ' + str(i) + ', chunksize ' + str(CHUNKSIZE))
	with open(PATH_OUT + 'sentences/notes_sentences_' + str(i) + '.pkl', 'wb') as notes_sentences_handle, open(PATH_OUT + 'words/notes_words_' + str(i) + '.txt', 'w') as notes_words_handle:
		#notes_sentences_pickler = pickle.Pickler(notes_sentences_handle)
		#notes_chunk.apply(lambda row: process_note(seg_obj, row.HADM_ID, row.TEXT, notes_sentences_pickler, notes_words_handle), axis=1)

		s, w = seg_obj.parse_documents(notes_chunk['TEXT'])
		
		#notes_sentences_pickler.dump((hadm_id, sentence_list))
		#flat_word_list = [word.lower() for sentence in word_list for word in sentence]
		#[notes_words_handle.write('%s\n' % word.lower()) for sentence in word_list for word in sentence]
	
	notes_sentences_handle.close()
	notes_words_handle.close()

#pool_size = mp.cpu_count()

csv_reader = pd.read_csv(PATH_IN + 'NOTEEVENTS.csv', chunksize=CHUNKSIZE)#, usecols=['HADM_ID', 'TEXT'], dtype={'HADM_ID': 'uint32', 'TEXT': 'object'})

seg_obj = Segmentation()

for i, notes_chunk in enumerate(csv_reader):
	process_chunk(notes_chunk, i, seg_obj)

#pool = mp.Pool(processes=pool_size)
#results = [pool.apply_async(process_chunks, args=(notes_chunk, i)) for i, notes_chunk in enumerate(csv_reader)]
#[p.get() for p in results]
