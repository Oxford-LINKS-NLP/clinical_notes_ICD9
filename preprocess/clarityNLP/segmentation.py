#!/usr/bin/env python3

import re2 as re
import os
import sys
import json
import time
import optparse
import en_core_web_md as english_model
from concurrent import futures
import threading
from timeit import default_timer as timer

from toolz import partition_all
from joblib import Parallel, delayed

import clarityNLP.segmentation_helper as seg_helper

#import pyximport; pyximport.install()
#from clarityNLP.cythonmodule import parse_tokenized_document_cython


VERSION_MAJOR = 0
VERSION_MINOR = 1

# set to True to enable debug output
TRACE = False

MODULE_NAME = 'segmentation.py'

###############################################################################

def get_words(self, sentence_list, subs, size_meas_subs):
	for sentence in self.nlp_words.pipe(sentence_list, n_threads=128):
		s = (word.text for word in sentence if not word.is_punct and not word.is_space)
		s = list(seg_helper.undo_substitutions(s, subs, size_meas_subs))
	yield (seg_helper.undo_substitutions(sentences, subs, size_meas_subs), s)

def parse_tokenized_document(self, document, subs, size_meas_subs):
	sentences = (sent.string.strip() for sent in document.sents)
	
	# fix various problems and undo the substitutions
	sentences = seg_helper.split_concatenated_sentences(sentences)
	
	sentences = seg_helper.fixup_sentences(sentences)
	sentences = seg_helper.split_section_headers(sentences)
	sentences = seg_helper.delete_junk(sentences)
	
	#nlp.remove_pipe('parser')
	#nlp.add_pipe(nlp.create_pipe('tagger'))
	
	(sentences, words) = get_words(sentences, subs, size_meas_subs)

	yield (sentences, words)

def do_substitutions(documents):
	return [seg_helper.do_substitutions(seg_helper.cleanup_report(document)) for document in documents]

def parse_documents(self, documents):

	# nlp = segmentation_init()
	# doc = nlp(text)
	# return [sent.string.strip() for sent in doc.sents]
	
	# Do some cleanup and substitutions before tokenizing. The substitutions
	# replace strings of tokens that tend to be incorrectly split with
	# a single token that will not be split.
	
	start = timer()
	print('\tcleaning and substitutions...', end=' ')
		
	#results = self.executor.map(do_substitutions, documents)
	#results = list(results)
	
	partitions = partition_all(100, documents)
	executor = Parallel(n_jobs=32)
	do = delayed(do_substitutions)
	tasks = (do(batch) for batch in partitions)
	results = executor(tasks)
	
	results = (item for sublist in results for item in sublist)
	
	#results = [do_substitutions(document) for document in documents]
	documents, subs_list, size_meas_subs_list = zip(*results)

	end = timer()
	print('\tdone ({0:.2f}s)'.format(end-start))

	# do the tokenization with the substitutions in place
	start = timer()
	print('\ttokenization...', end=' ')
	
	#documents = [doc for doc in nlp.pipe(documents, n_threads=32, batch_size=50)]
	
	documents = (parse_tokenized_document(self, doc, subs, size_meas_subs) for doc, subs, size_meas_subs in zip(self.nlp_sentences.pipe(documents, n_threads=128, batch_size=32*100), subs_list, size_meas_subs_list))

	end = timer()	
	print('\tdone ({0:.2f}s)'.format(end-start))
	
	#start = timer()
	#print('\tparsing tokenized documents...', end=' ')

	#results = ex.map(parse_tokenized_document, documents)
	#documents = list(results)
	
	#end = timer()	
	#print('\tdone ({0:.2f}s)'.format(end-start))
		
	return list(zip(*documents))


###############################################################################
class Segmentation(object):

	def __init__(self):
		self.regex_multi_space = re.compile(r' +')
		self.regex_multi_newline = re.compile(r'\n+')
		print('loading models...', end=' ')
		self.nlp_sentences = english_model.load()
		self.nlp_sentences.remove_pipe('tagger')
		self.nlp_sentences.remove_pipe('ner')
		self.nlp_words = english_model.load()
		self.nlp_words.remove_pipe('parser')
		self.nlp_words.remove_pipe('ner')
		print('done')

		self.executor = futures.ThreadPoolExecutor(max_workers=32)

	def remove_newlines(self, text):

		# replace newline with space
		no_newlines = self.regex_multi_newline.sub(' ', text)

		# replace multiple consecutive spaces with single space
		cleaned_text = self.regex_multi_space.sub(' ', no_newlines)
		return cleaned_text

	def parse_documents(self, documents):
		print('start parsing')
		return parse_documents(self, documents)
