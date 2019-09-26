#!/usr/bin/env python3

import re2 as re
import string
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
		
		punct = list(string.punctuation)
		punct.remove('.')
		punct.append('[**')
		punct.append('**]')
		punct = [re.escape(p) for p in punct]
		
		prefixes_custom = tuple(punct)
		infixes_custom = tuple(punct)
		suffixes_custom = tuple(punct)
		
		#prefixes_custom = tuple([r'\[\*\*', r'('])
		#suffixes_custom = tuple([r'\*\*\]', r')'])
		#infixes_custom = tuple([r'\[\*\*', r'\*\*\]', r'(', r')', r'>', r'<', r'->', r'-->', r'--->'])

		exceptions_custom = {id : pattern for id, pattern in tokenizer_utils.generate_matcher_pattern1()}		
		exceptions = update_exc(self.nlp.Defaults.tokenizer_exceptions, exceptions_custom)

		prefix_re = compile_prefix_regex(self.nlp.Defaults.prefixes + prefixes_custom)
		infix_re  = compile_infix_regex(infixes_custom + self.nlp.Defaults.infixes)
		suffix_re = compile_suffix_regex(self.nlp.Defaults.suffixes + suffixes_custom)
		
		tokenizer = SpacyTokenizer(self.nlp.vocab, rules=exceptions,
							prefix_search=prefix_re.search,
							suffix_search=suffix_re.search,
							infix_finditer=infix_re.finditer, token_match=self.nlp.Defaults.token_match)

		self.nlp.tokenizer = tokenizer

		matcher = Matcher(self.nlp.vocab)
						
		def on_match_pattern(matcher, doc, id, matches):
		
			match_id, start, end = matches[id]

			if self.nlp.vocab.strings[match_id].startswith('p3'):
				span = doc[start+1:end]
				span.merge()
				for i in range(id, len(matches)):
					matches[i] = (matches[i][0], matches[i][1] - 1,  matches[i][2] - 1)

			elif self.nlp.vocab.strings[match_id].startswith('p2.1'):
				span1 = doc[start:start+2]
				span2 = doc[start+2:end]
				span1.merge()
				span2.merge()
				for i in range(id, len(matches)):
					matches[i] = (matches[i][0], matches[i][1] - 2,  matches[i][2] - 2)

			elif self.nlp.vocab.strings[match_id].startswith('p2.2'):
				span2 = doc[start+1:end]
				span2.merge()
				for i in range(id, len(matches)):
					matches[i] = (matches[i][0], matches[i][1] - 1,  matches[i][2] - 1)

			elif self.nlp.vocab.strings[match_id].startswith('p2.3'):
				span1 = doc[start:start+2]
				span1.merge()
				for i in range(id, len(matches)):
					matches[i] = (matches[i][0], matches[i][1] - 1,  matches[i][2] - 1)
	
		for id, pattern in tokenizer_utils.generate_matcher_pattern2():
			matcher.add(id, on_match_pattern, pattern)
			
		for id, pattern in tokenizer_utils.generate_matcher_pattern3():
			matcher.add(id, on_match_pattern, pattern)
				
		self.nlp.add_pipe(matcher, before='parser')

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

		docs = [[[token.lower() for token in [tokenizer_utils.do_substitutions(word.text) for word in sentence] if any(char.isalpha() for char in token) ] for sentence in doc.sents] for doc in documents]
		
		end = timer()
		print('\tdone ({0:.2f}s)'.format(end-start))

		return docs
