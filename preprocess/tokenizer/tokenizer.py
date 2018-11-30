#!/usr/bin/env python3

import re2 as re
from timeit import default_timer as timer

import en_core_web_md as english_model
from spacy.tokenizer import Tokenizer as SpacyTokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex, update_exc
from spacy.matcher import Matcher

from tokenizer import tokenizer_utils

class Tokenizer(object):

	def __init__(self, batch_size, n_cpus, n_threads, mode):

		print('loading model...', end=' ')
		self.nlp = english_model.load()
		self.nlp.remove_pipe('tagger')
		self.nlp.remove_pipe('ner')
		
		prefixes_custom = tuple([r'\[\*\*'])
		suffixes_custom = tuple([r'\*\*\]'])
		infixes_custom = tuple([r'\[\*\*', r'\*\*\]'])

		exceptions_custom = {id : pattern for id, pattern in tokenizer_utils.generate_matcher_pattern1()}		
		exceptions = update_exc(self.nlp.Defaults.tokenizer_exceptions, exceptions_custom)

		prefix_re = compile_prefix_regex(self.nlp.Defaults.prefixes + prefixes_custom)
		suffix_re = compile_suffix_regex(self.nlp.Defaults.suffixes + suffixes_custom)
		infix_re  = compile_infix_regex(self.nlp.Defaults.infixes + infixes_custom)

		tokenizer = SpacyTokenizer(self.nlp.vocab, rules=exceptions,
							prefix_search=prefix_re.search,
							suffix_search=suffix_re.search,
							infix_finditer=infix_re.finditer, token_match=self.nlp.Defaults.token_match)

		self.nlp.tokenizer = tokenizer

		matcher = Matcher(self.nlp.vocab)
		
		def on_match_pattern2(matcher, doc, id, matches):
			for match_id, start, end in matches:
				#string_id = self.nlp.vocab.strings[match_id]
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
					else:
						span2 = doc[start+1:end]
						span2.merge()
						
		def on_match_pattern3(matcher, doc, id, matches):
			for match_id, start, end in matches:
				#string_id = self.nlp.vocab.strings[match_id]
				span = doc[start+1:]
				span.merge()
		
		for id, pattern in tokenizer_utils.generate_matcher_pattern2():
			matcher.add(id, on_match_pattern2, pattern)
		for id, pattern in tokenizer_utils.generate_matcher_pattern3():
			matcher.add(id, on_match_pattern3, pattern)
				
		self.nlp.add_pipe(matcher, before='parser')
		#self.nlp.add_pipe(tokenizer_utils.split_section_headers , before='parser')

		print('done')

		self.batch_size = batch_size
		self.n_cpus = n_cpus
		self.n_threads = n_threads
		self.mode = mode

	def tokenize_documents(self, documents):

		print('start parsing')
		start = timer()
		print('\tcleaning and tokenization...')

		documents = (tokenizer_utils.cleanup_report(doc) for doc in documents)
		documents = (tokenizer_utils.merge_anon_tokens(doc) for doc in self.nlp.pipe(documents, n_threads=self.n_threads, batch_size=self.batch_size))

		docs = []
		for doc in documents:
			sentences = [[str.lower(tokenizer_utils.do_substitutions(word.text)).strip(' :-').replace(' ', '_') for word in sentence if not word.is_punct and not word.is_space] for sentence in doc.sents]
			print(sentences)
			docs.append(sentences)
		
		#for document in documents:
		#	sentences = (sent.string.strip() for sent in document.sents)
		#	sentences = tokenizer_utils.split_section_headers(sentences)
		#	sentences = tokenizer_utils.delete_junk(list(sentences))
		#	sentences = list(str.lower(word.text).strip(' :-').replace(' ', '_') for sentence in self.nlp_words.pipe(sentences, n_threads=self.n_threads) for word in sentence if not word.is_punct and not word.is_space)
		#	docs.append(sentences)
		
		end = timer()
		print('\tdone ({0:.2f}s)'.format(end-start))

		return docs
