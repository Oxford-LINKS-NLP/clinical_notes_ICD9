#!/usr/bin/env python3

import re2 as re
import en_core_web_md as english_model
from timeit import default_timer as timer
from toolz import partition_all
from multiprocessing import Pool

import clarityNLP.segmentation_helper as seg_helper

VERSION_MAJOR = 0
VERSION_MINOR = 1

# set to True to enable debug output
TRACE = False

MODULE_NAME = 'segmentation.py'

###############################################################################

def get_sentences(self, sentence_list, subs):
	for sentence in self.nlp_words.pipe(sentence_list, n_threads=self.n_threads):
		sent = (word.text for word in sentence if not word.is_punct and not word.is_space)
		sent1 = (str.lower(subs.get(word, word).strip().rstrip(':-').replace(' ', '_')) for word in sent)
		yield list(sent1)

def parse_tokenized_document(self, document, subs):

	sentences = (sent.string.strip() for sent in document.sents)
	
	# fix various problems and undo the substitutions
	sentences = seg_helper.split_concatenated_sentences(sentences)
	
	sentences = seg_helper.fixup_sentences(list(sentences))
	sentences = seg_helper.split_section_headers(sentences)
	sentences = seg_helper.delete_junk(list(sentences))
	
	sentences = list(get_sentences(self, sentences, subs))

	return sentences

def do_substitutions(documents, mode):
	return [seg_helper.do_substitutions(seg_helper.cleanup_report(document), mode) for document in documents]

###############################################################################
class Tokenizer(object):

	def __init__(self, base_batch_size, n_cpus, n_threads, mode):
	
		print('loading models...', end=' ')
		self.nlp_sentences = english_model.load()
		self.nlp_sentences.remove_pipe('tagger')
		self.nlp_sentences.remove_pipe('ner')
		self.nlp_words = english_model.load()
		self.nlp_words.remove_pipe('parser')
		self.nlp_words.remove_pipe('ner')
		print('done')

		self.base_batch_size = base_batch_size
		self.n_cpus = n_cpus
		self.n_threads = n_threads
		self.mode = mode

	def tokenize_documents(self, documents):
		
		# Do some cleanup and substitutions before tokenizing. The substitutions
		# replace strings of tokens that tend to be incorrectly split with
		# a single token that will not be split.
		
		print('start parsing')
		start = timer()
		print('\tcleaning and substitutions...')
			
		with Pool(processes=self.n_cpus) as pool:

			partitions = partition_all(self.base_batch_size, documents)

			results_async = [pool.apply_async(do_substitutions, args=(batch, self.mode)) for batch in partitions]
			results_partitioned = (res.get() for res in results_async)
			results = (result for result_partition in results_partitioned for result in result_partition)
			#results.sort(key=lambda tup : len(tup[0]))
			documents, subs_lists = zip(*results)

		end = timer()
		print('\tdone ({0:.2f}s)'.format(end-start))
		
		# do the tokenization with the substitutions in place
		start = timer()
		print('\ttokenization...')
		
		documents = list(parse_tokenized_document(self, doc, subs) for doc, subs in zip(self.nlp_sentences.pipe(documents, n_threads=self.n_threads, batch_size=self.base_batch_size), subs_lists))

		end = timer()
		print('\tdone ({0:.2f}s)'.format(end-start))
			
		return documents
