#!/usr/bin/env python3

import re2 as re

import en_core_web_md as english_model
from spacy.tokenizer import Tokenizer as SpacyTokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex, update_exc
from spacy.matcher import Matcher

from timeit import default_timer as timer
from toolz import partition_all
from multiprocessing import Pool

from tokenizer import tokenizer_utils

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
	sentences = tokenizer_utils.split_concatenated_sentences(sentences)

	sentences = tokenizer_utils.fixup_sentences(list(sentences))
	sentences = tokenizer_utils.split_section_headers(sentences)
	sentences = tokenizer_utils.delete_junk(list(sentences))

	sentences = list(get_sentences(self, sentences, subs))

	return sentences

def do_substitutions(documents, mode):
	return [tokenizer_utils.do_substitutions(tokenizer_utils.cleanup_report(document), mode) for document in documents]

###############################################################################
class Tokenizer(object):

	def __init__(self, base_batch_size, n_cpus, n_threads, mode):

		print('loading models...', end=' ')
		self.nlp_sentences = english_model.load()
		self.nlp_sentences.remove_pipe('tagger')
		self.nlp_sentences.remove_pipe('ner')
		self.nlp_words = english_model.load()
		self.nlp_words.remove_pipe('tagger')
		self.nlp_words.remove_pipe('parser')
		self.nlp_words.remove_pipe('ner')

		prefixes_custom = tuple([r'\[\*\*'])
		suffixes_custom = tuple([r'\*\*\]'])
		infixes_custom = tuple([r'\[\*\*', r'\*\*\]'])

		exceptions_custom = tokenizer_utils.abbrev_list

		exceptions = update_exc(self.nlp_sentences.Defaults.tokenizer_exceptions, exceptions_custom)

		prefix_re = compile_prefix_regex(self.nlp_sentences.Defaults.prefixes + prefixes_custom)
		suffix_re = compile_suffix_regex(self.nlp_sentences.Defaults.suffixes + suffixes_custom)
		infix_re  = compile_infix_regex(self.nlp_sentences.Defaults.infixes + infixes_custom)

		tokenizer = SpacyTokenizer(self.nlp_sentences.vocab, rules=exceptions,
							prefix_search=prefix_re.search,
							suffix_search=suffix_re.search,
							infix_finditer=infix_re.finditer, token_match=self.nlp_sentences.Defaults.token_match)

		self.nlp_sentences.tokenizer = tokenizer
		self.nlp_words.tokenizer = tokenizer

		matcher = Matcher(self.nlp_sentences.vocab)
		
		def on_match_pattern1(matcher, doc, id, matches):
			for match_id, start, end in matches:
				string_id = self.nlp_sentences.vocab.strings[match_id]
				n_tokens = end-start
				if n_tokens == 4:
					span1 = doc[start:start+2]
					span2 = doc[start+2:end]
					span1.merge()
					span2.merge()
				else:
					if doc[start+1].text == '.':
						span1 = doc[start:start+2]
						span1.merge()
						print(doc[start].text)
					else:
						span2 = doc[start+1:end]
						span2.merge()
		
		for id, pattern in tokenizer_utils.generate_matcher_pattern1():
			matcher.add(id, on_match_pattern1, pattern)
				
		self.nlp_sentences.add_pipe(matcher, before='parser')

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

	def tokenize_documents_test(self, documents):

		print('start parsing')
		start = timer()
		print('\tcleaning and tokenization...')

		documents = (tokenizer_utils.cleanup_report(doc) for doc in documents)
		documents = (tokenizer_utils.merge_anon_tokens(doc) for doc in self.nlp_sentences.pipe(documents, n_threads=self.n_threads, batch_size=self.base_batch_size))
		docs = []
		for document in documents:
			sentences = (sent.string.strip() for sent in document.sents)
		#	sentences = tokenizer_utils.split_section_headers(sentences)
		#	sentences = tokenizer_utils.delete_junk(list(sentences))
			sentences = list(str.lower(word.text).strip().rstrip(':-').replace(' ', '_') for sentence in self.nlp_words.pipe(sentences, n_threads=self.n_threads) for word in sentence if not word.is_punct and not word.is_space)
			docs.append(sentences)
		
		end = timer()
		print('\tdone ({0:.2f}s)'.format(end-start))

		return docs
