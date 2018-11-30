#!/usr/bin/env python3

import re2 as re
import hashlib
from spacy.attrs import ORTH

#medical abbreviations from
#https://en.wikipedia.org/wiki/List_of_abbreviations_used_in_medical_prescriptions
#https://en.wikipedia.org/wiki/List_of_medical_abbreviations:_Latin_abbreviations

#pattern1 -> tok.tok. | tok. | tok.tok.tok.
#pattern2 -> tok. tok.
#pattern3 -> tok tok.

#'d. in p. ae.'
#'si op. sit'

#vitals
#Temp.
#measure units

abbrev_pattern1 = ['a.c.',
						'a.c.h.s.',
						'ac&hs',
						'a.d.',
						'ad.',
						'add.',
						'admov.',
						'aeq.',
						'agit.',
						'amp.',
						'aq.',
						'a.l.',
						'a.s.',
						'a.u.',
						'b.d.s.',
						'bib.',
						'b.i.d.',
						'b.d.',
						'bol.',
						'Ph.Br.',
						'b.t.',
						'bucc.',
						'cap.',
						'caps.',
						'c.m.',
						'c.m.s.',
						'c.',
						'cib.',
						'c.c.',
						'cf.',
						'c.n.',
						'cochl.',
						'colet.',
						'comp.',
						'contin.',
						'cpt.',
						'cr.',
						'cuj.',
						'c.v.',
						'cyath.',
						'd.',
						'D/C',
						'decoct.',
						'det.',
						'dil.',
						'dim.',
						'disp.',
						'div.',
						'd.t.d. .',
						'elix.',
						'e.m.p.',
						'emuls.',
						'exhib.',
						'f.',
						'f.h.',
						'fl.',
						'fld.',
						'f.m.',
						'f.s.a.',
						'ft.',
						'garg.',
						'gr.',
						'gtt.',
						'gutt.',
						'h.',
						'H/O',
						'hor.',
						'habt.',
						'h.s.',
						'inj.',
						'i.m.',
						'inf.',
						'i.v.',
						'i.v.p.',
						'lb.',
						'l.c.d.',
						'liq.',
						'lot.',
						'M.',
						'm.',
						'max.',
						'm.d.u.',
						'mg/dL',
						'min.',
						'mist.',
						'mit.',
						'mitt.',
						'nebul.',
						'neb.',
						'noct.',
						'n.p.o.',
						'1/2NS.',
						'o.d.',
						'o.m.',
						'o.n.',
						'o.s.',
						'o.u.',
						'p.',
						'p.c.',
						'p.c.h.s.',
						'pc&hs',
						'Ph.Br.',
						'Ph.Eur.',
						'Ph.Int.',
						'pig.',
						'pigm.',
						'p.o.',
						'ppt.',
						'p.r.',
						'p.r.n.',
						'pt.',
						'pulv.',
						'p.v.',
						'q.1.d.',
						'q.1.h.',
						'q.2.h.',
						'q.4.h.',
						'q.6.h.',
						'q.8.h.',
						'q.a.d.',
						'q.a.m.',
						'q.d.',
						'q.d.a.m.',
						'q.d.p.m.',
						'q.d.s.',
						'q.p.m.',
						'q.h.',
						'q.h.s.',
						'q.i.d.',
						'q.l.',
						'q.n.',
						'q.o.d.',
						'q.p.m.',
						'q.q.',
						'q.q.h.',
						'q.s.',
						'q.v.',
						'rel.',
						'Rel.',
						'rep.',
						'rept.',
						'R/L',
						'rep.',
						's.',
						's.a.',
						'sem.',
						's.i.d.',
						'Sig.',
						'S.',
						'sig.',
						'sing.',
						's.l.',
						'sol.',
						's.o.s.',
						's.s.',
						'st.',
						'Stat.',
						'sum.',
						'supp.',
						'susp.',
						'Sust.Rel.'
						'sust.Rel.'
						'Sust.rel.',
						'sust.rel.',
						'Sust.',
						'sust',
						'syr.',
						'tab.',
						'tal.',
						't.',
						't.d.s.',
						't.i.d.',
						't.d.',
						't.d.s.',
						'tinct.',
						't.i.d.',
						't.i.w.',
						'top.',
						'tinc.',
						'tinct.',
						'trit.',
						'troch.',
						'u.d.',
						'ut.',
						'dict.',
						'ung.',
						'vag.',
						'w/a',
						'w/f',
						'w/o',
						'y.o.'
						]
		
abbrev_pattern2 = ['alt. d.',
						'alt. dieb.',
						'alt. h.',
						'alt. hor.',
						'aq. bull.',
						'aq. com.',
						'aq. dest.',
						'aq. ferv.',
						'cochl. ampl.',
						'cochl. infant.',
						'cochl. mag.',
						'cochl. mod.',
						'cochl. parv.',
						'cyath. vinos.',
						'dieb. alt.',
						'f. pil.',
						'hor. alt.',
						'hor. decub.',
						'hor. intermed.',
						'hor. tert.',
						'lat. dol.',
						'mod. praescript.',
						'omn. bih.',
						'omn. hor.',
						'part. aeq.'
						]
										
abbrev_pattern3 = ['ad lib.',
						'ad us.',
						'bis ind.',
						'ex aq.',
						'non rep.'
						]
										
def generate_matcher_pattern1():
	for abbrev in abbrev_pattern1:
		yield (abbrev, [{ORTH: abbrev}])
										
def generate_matcher_pattern2():
	for abbrev in abbrev_pattern2:
		w1, w2 = [w.strip() for w in abbrev.split('.')][:2]
		yield ('{}_{}_1'.format(w1, w2), [{ORTH: w1}, {ORTH: '.'}, {ORTH: w2}, {ORTH: '.'}])
		yield ('{}_{}_2'.format(w1, w2), [{ORTH: w1+'.'}, {ORTH: w2}, {ORTH: '.'}])
		yield ('{}_{}_3'.format(w1, w2), [{ORTH: w1}, {ORTH: '.'}, {ORTH: w2+'.'}])
		
def generate_matcher_pattern3():
	for abbrev in abbrev_pattern3:
		w1, w2 = [w.strip() for w in abbrev.split(' ')]
		w2 = w2.strip('.')
		yield ('{}_{}_1'.format(w1, w2), [{ORTH: w1}, {ORTH: w2}, {ORTH: '.'}])

ANONTOKEN = 'ANONTOKEN'
FIRST_NAME_TOKEN = 'FIRSTNAMETOKEN'
LAST_NAME_TOKEN = 'LASTNAMETOKEN'
DOCTOR_FIRST_NAME_TOKEN = 'DOCTORFIRSTNAMETOKEN'
DOCTOR_LAST_NAME_TOKEN = 'DOCTORLASTNAMETOKEN'
NAME_TOKEN = 'NAMETOKEN'
NAME_PREFIX_TOKEN = 'NAMEPREFIXTOKEN'
ADDRESS_TOKEN = 'ADDRESSTOKEN'
LOCATION_TOKEN = 'LOCATIONTOKEN'
HOSPITAL_TOKEN = 'HOSPITALTOKEN'
PO_BOX_TOKEN = 'POBOXTOKEN'
STATE_TOKEN = 'STATENAMETOKEN'
COUNTRY_TOKEN = 'COUNTRYNAMETOKEN'
COMPANY_TOKEN = 'COMPANYNAMETOKEN'
TELEPHONE_NUMBER_TOKEN = 'TELEPHONENUMBERTOKEN'
PAGER_NUMBER_TOKEN = 'PAGERNUMBERTOKEN'
SSN_TOKEN = 'SSNTOKEN'
MEDICAL_RECORD_NUMBER_TOKEN = 'MEDICALRECORDNUMBERTOKEN'
UNIT_NUMBER_TOKEN = 'UNITNUMBERTOKEN'
AGE_OVER_90_TOKEN = 'AGEOVER90TOKEN'
EMAIL_ADDRESS_TOKEN = 'EMAILADDRESSTOKEN'
URL_TOKEN = 'URLADDRESSTOKEN'
HOLYDAY_TOKEN = 'HOLYDAYNAMETOKEN'
JOB_NUMBER_TOKEN = 'JOBNUMBERTOKEN'
MD_NUMBER_TOKEN = 'MDNUMBERTOKEN'
DATE_RANGE_TOKEN = 'DATERANGETOKEN'
NUMERIC_IDENTIFIER_TOKEN = 'NUMERICIDENTIFIERTOKEN'
DATE_LITERAL_TOKEN = 'DATETOKEN'
UNIVERSITY_TOKEN = 'UNIVERSITYTOKEN'
DICTATOR_INFO_TOKEN = 'DICTATORINFOTOKEN'
CC_CONTACT_INFO_TOKEN = 'CCCONTACTINFOTOKEN'
CLIP_NUMBER_TOKEN = 'CLIPNUMBERTOKEN'
SERIAL_NUMBER_TOKEN = 'SERIALNUMBERTOKEN'
ATTENDING_INFO_TOKEN = 'ATTENDINGINFOTOKEN'
PROVIDER_NUMBER_TOKEN = 'PROVIDERNUMBERTOKEN'

# regex for locating a PHI [** ... **]

str_anon_date = r'(?P<anon_date>([0-9/\-]+)|([0-9]+))'
str_anon_first_name = r'(?P<anon_first_name>(Known firstname)|(Female First Name)|(Male First Name)|(First Name))'
str_anon_last_name = r'(?P<anon_last_name>(Known lastname)|(Last Name))'
str_anon_doctor_first_name = r'(?P<anon_doctor_first_name>(Doctor First Name))'
str_anon_doctor_last_name = r'(?P<anon_doctor_last_name>(Doctor Last Name))'
str_anon_name = r'(?P<anon_name>(Name)|(Name Initial)|(Initials)|(Initial))'
str_anon_name_prefix = r'(?P<anon_name_prefix>(Name Prefix))'
str_anon_address = r'(?P<anon_address>(Street Address)|(Apartment Address))'
str_anon_university = r'(?P<anon_university>(Location \(Universities\))|(University/College))' #keep before location
str_anon_location = r'(?P<anon_location>(Location))'
str_anon_hospital = r'(?P<anon_hospital>(Hospital)|(Wardname)|(Hospital Unit Name)|(Hospital Ward Name))'
str_anon_po_box = r'(?P<anon_po_box>(PO BOX)|(PO Box))'
str_anon_state = r'(?P<anon_state>(State)|(State/Zipcode))'
str_anon_country = r'(?P<anon_country>(Country))'
str_anon_company = r'(?P<anon_company>(Company))'
str_anon_telephone_number = r'(?P<anon_telephone_number>(Telephone/Fax))'
str_anon_pager_number = r'(?P<anon_pager_number>(Pager number))'
str_anon_social_security_number = r'(?P<anon_social_security_number>(Social Security Number))'
str_anon_medical_record_number = r'(?P<anon_medical_record_number>(Medical Record Number))'
str_anon_unit_number = r'(?P<anon_unit_number>(Unit Number))'
str_anon_age_over_90 = r'(?P<anon_age_over_90>(Age over 90))'
str_anon_email_address = r'(?P<anon_email_address>(E-mail address))'
str_anon_url = r'(?P<anon_url>(URL))'
str_anon_holiday = r'(?P<anon_holiday>(Holiday))'
str_anon_job_number = r'(?P<anon_job_number>(Job Number))'
str_anon_md_number = r'(?P<anon_md_number>(MD Number))'
str_anon_date_range = r'(?P<anon_date_range>(Date range)|(Date Range))'
str_anon_numeric_identifier = r'(?P<anon_numeric_identifier>(Numeric Identifier))'
str_anon_date_literal = r'(?P<anon_date_literal>(Month)|(Month/Day)|(Month/Year)|(Year)|(Month/Day/Year)|(Year/Month/Day)|(Year/Month)|(Day Month)|(Month Day)|(Month Year)|(Month/Year 1)|(January)|(February)|(March)|(April)|(May)|(June)|(July)|(August)|(September)|(October)|(November)|(December))'
str_anon_dictator_info = r'(?P<anon_dictator_info>(Dictator Info))'
str_anon_cc_contact_info = r'(?P<anon_cc_contact_info>(CC Contact Info))'
str_anon_clip_number = r'(?P<anon_clip_number>(Clip Number))'
str_anon_serial_number = r'(?P<anon_serial_number>(Serial Number))'
str_anon_attending_info = r'(?P<anon_attending_info>(Attending Info))'
str_anon_provider_number = r'(?P<anon_provider_number>(Provider Number))'
str_anon_default = r'(.*?)'
str_anon_tokens = r'|'.join([str_anon_first_name, str_anon_last_name, str_anon_doctor_first_name, str_anon_doctor_last_name,
										str_anon_name, str_anon_name_prefix, str_anon_address, str_anon_university, str_anon_location, str_anon_hospital,
										str_anon_po_box, str_anon_state, str_anon_country, str_anon_company,
										str_anon_telephone_number, str_anon_pager_number, str_anon_social_security_number, str_anon_medical_record_number,
										str_anon_unit_number, str_anon_age_over_90, str_anon_email_address, str_anon_url, str_anon_holiday, str_anon_job_number,
										str_anon_md_number, str_anon_date_range, str_anon_numeric_identifier, str_anon_date_literal, str_anon_dictator_info,
										str_anon_cc_contact_info, str_anon_clip_number, str_anon_serial_number, str_anon_attending_info, str_anon_provider_number,
										str_anon_default])
										
str_anon = r'(?P<anon>\[\*\*(({date})|(({tokens})(\(?[0-9]+\)?)?((\s?(\(.*?\)\s)?)|(\s))(?P<anon_id>[0-9]+)?))\*\*\])'.format(date=str_anon_date, tokens=str_anon_tokens)
regex_anon = re.compile(str_anon)

# start of a numbered section, such as a list, but with no whitespace
# separating the numbers from the adjacent text
str_list_start_no_space = r'(?P<list_start_no_space>\b(?P<listnum>\d+(\.|\)))(?P<word>[a-zA-Z]+))'
regex_list_start_no_space = re.compile(str_list_start_no_space)

# find numbered sentences: look for digits followed by '.' or ')',
# whitespace, then a capital letter starting a word
str_list_start = r'(?P<list_start>\b(?P<listnum>\d+(\.|\)))\s+)'
str_list_item = r'(?P<list_item>' + str_list_start + r'([A-Z][a-z]+|\d)\b)'
regex_list_start = re.compile(str_list_start)
regex_list_item  = re.compile(str_list_item)

# find captialized headers
str_caps_word = r'\b([123]-?D|[-_A-Z]+|[-_A-Z]+/[-_A-Z]+)\b'
str_caps_header = r'(?P<caps_header>(' + str_caps_word + r'\s+)*' + str_caps_word + r'\s*#?:)'
regex_caps_header = re.compile(str_caps_header)

# find concatenated sentences with no space after the period

# need at least two chars before '.', to avoid matching C.Diff, M.Smith, etc.
# neg lookahead prevents capturing inside abbreviations such as Sust.Rel.
str_two_sentences = r'(?P<two_sentences>\b[a-zA-Z]{2,}\.[A-Z][a-z]+(?!\.))'
regex_two_sentences = re.compile(str_two_sentences)

regex_multi_space = re.compile(r' +')
regex_multi_newline = re.compile(r'\n+')

def remove_newlines(document):

		# replace newline with space
		no_newlines = regex_multi_newline.sub(' ', document)

		# replace multiple consecutive spaces with single space
		cleaned_document = regex_multi_space.sub(' ', no_newlines)
		return cleaned_document

def generate_token(base_name, mo, mode):
	
	if mode == 0:
		return ANONTOKEN
	elif mode == 1:
		return base_name
	elif mode == 2:
		if mo.group('anon_id'):
			return '{0}_{1}'.format(base_name, hashlib.md5(mo.group('anon').encode()).hexdigest())
		else:
			return base_name
			
def merge_anon_tokens(doc):
	matches = regex_anon.finditer(doc.text)
	if matches == None:
		return doc
	for mo in matches:
		if doc.merge(mo.start(), mo.end()) == None:
			print(mo.string[mo.start():mo.end()])
	return doc

def do_substitutions(anon_token):

	mode = 2
	
	def repl(mo):
		
		nonlocal mode
	
		text = mo.string[mo.start():mo.end()]
		
		if mo.group('anon') and mo.group('anon_date'):
			return mo.group('anon_date').replace('/', '').replace('-', '/')
		elif mo.group('anon') and mo.group('anon_first_name'):
			return generate_token(FIRST_NAME_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_last_name'):
			return generate_token(LAST_NAME_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_doctor_first_name'):
			return generate_token(DOCTOR_FIRST_NAME_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_doctor_last_name'):
			return generate_token(DOCTOR_LAST_NAME_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_name'):
			return generate_token(NAME_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_name_prefix'):
			return generate_token(NAME_PREFIX_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_address'):
			return generate_token(ADDRESS_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_university'):
			return generate_token(UNIVERSITY_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_location'):
			return generate_token(LOCATION_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_hospital'):
			return generate_token(HOSPITAL_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_po_box'):
			return generate_token(PO_BOX_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_state'):
			return generate_token(STATE_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_country'):
			return generate_token(COUNTRY_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_company'):
			return generate_token(COMPANY_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_telephone_number'):
			return generate_token(TELEPHONE_NUMBER_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_pager_number'):
			return generate_token(PAGER_NUMBER_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_social_security_number'):
			return generate_token(SSN_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_medical_record_number'):
			return generate_token(MEDICAL_RECORD_NUMBER_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_unit_number'):
			return generate_token(UNIT_NUMBER_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_age_over_90'):
			return generate_token(AGE_OVER_90_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_email_address'):
			return generate_token(EMAIL_ADDRESS_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_url'):
			return generate_token(URL_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_holiday'):
			return generate_token(HOLYDAY_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_job_number'):
			return generate_token(JOB_NUMBER_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_md_number'):
			return generate_token(MD_NUMBER_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_date_range'):
			return generate_token(DATE_RANGE_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_numeric_identifier'):
			return generate_token(NUMERIC_IDENTIFIER_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_date_literal'):
			return generate_token(DATE_LITERAL_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_dictator_info'):
			return generate_token(DICTATOR_INFO_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_cc_contact_info'):
			return generate_token(CC_CONTACT_INFO_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_clip_number'):
			return generate_token(CLIP_NUMBER_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_serial_number'):
			return generate_token(SERIAL_NUMBER_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_attending_info'):
			return generate_token(ATTENDING_INFO_TOKEN, mo, mode)
		elif mo.group('anon') and mo.group('anon_provider_number'):
			return generate_token(PROVIDER_NUMBER_TOKEN, mo, mode)
		elif mo.group('anon'):
			return generate_token(ANONTOKEN, mo, mode)

	document = regex_anon.sub(repl, anon_token)

	return (document)

def erase_spans(report, span_list):
	"""
	Erase all report chars bounded by each [start, end) span.
	"""

	if len(span_list) > 0:
		prev_end = 0
		new_report = ''
		for span in span_list:
			start = span[0]
			end   = span[1]
			new_report += report[prev_end:start]
			prev_end = end
		new_report += report[prev_end:]
		report = new_report

	return report

def cleanup_report(report):

	# remove (Over) ... (Cont) inserts
	spans = []
	iterator = re.finditer(r'\(Over\)', report)
	for match_over in iterator:
		start = match_over.start()
		chunk = report[match_over.end():]
		match_cont = re.search(r'\(Cont\)', chunk)
		if match_cont:
			end = match_over.end() + match_cont.end()
			spans.append( (start, end))
			
	report = erase_spans(report, spans)

	# insert a space between list numbers and subsequent text, makes
	# lists and start-of-sentence negations easier to identify
	prev_end = 0
	new_report = ''
	iterator = regex_list_start_no_space.finditer(report)
	for match in iterator:
		# end of list num (digits followed by '.' or ')'
		end = match.end('listnum')
		# start of following (concatenated) word
		start = match.start('word')
		new_report += report[prev_end:end]
		new_report += ' '
		prev_end = start
	new_report += report[prev_end:]
	report = new_report

	# remove numbering in lists
	spans = []
	iterator = regex_list_item.finditer(report)
	for match in iterator:
		start = match.start('listnum')
		end   = match.end('listnum')
		spans.append( (start, end))

	report = erase_spans(report, spans)
		
	# Remove long runs of dashes, underscores, or stars
	report = re.sub(r'[-_*]{3,}', ' ', report)
	
	# collapse repeated whitespace (including newlines) into a single space
	report = re.sub(r'\s+', ' ', report)

	# convert unicode left and right quotation marks to ascii
	report = re.sub(u'(\u2018|\u2019)', "'", report)
	
	return report
	
def fixup_sentences(sentence_list):
	"""
	Move punctuation from one sentence to another, if necessary.
	"""
	
	num = len(sentence_list)
	
	i = 1
	while i < num:
		s = sentence_list[i]
		if s.startswith(':') or s.startswith(','):
			# move to end of previous sentence
			sprev = sentence_list[i-1]
			sentence_list[i-1] = sprev + ':'
			sentence_list[i]   = s[1:].lstrip()
		i += 1
	return sentence_list

def split_section_headers(doc):
	"""
	Put an all caps section header in a separate sentence from the subsequent
	text.
	"""

	matches = regex_caps_header.finditer(doc[:-2].text)
	
	for mo in matches:
		print(mo.string[mo.start():mo.end()])
		doc[mo.end()+1].sent_start = True

	return doc
		

def split_concatenated_sentences(sentence_list):

	#sentences = []
	for s in sentence_list:
		match = regex_two_sentences.search(s)
		if match:
			s1 = s[:match.end()]
			s2 = s[match.end():]
			#sentences.append(s1)
			yield s1
			#sentences.append(s2)
			yield s2
		else:
			#sentences.append(s)
			yield s

	#return sentences


###############################################################################
def delete_junk(sentence_list):
	"""
	"""

	sentences = []
	num = len(sentence_list)

	for s in sentence_list:
		i = 0
		while i < num:
			s = sentence_list[i]

			# delete any remaining list numbering
			match = regex_list_start.match(s)
			if match:
				s = s[match.end():]

			# remove any sentences that consist of just '1.', '2.', etc.
			match = re.match(r'\A\s*\d+(\.|\))\s*\Z', s)
			if match:
				i += 1
				continue

			# remove any sentences that consist of '#1', '#2', etc.
			match = re.match(r'\A\s*#\d+\s*\Z', s)
			if match:
				i += 1
				continue

			# remove any sentences consisting entirely of symbols
			match = re.match(r'\A\s*[^a-zA-Z0-9]+\s*\Z', s)
			if match:
				i += 1
				continue

			# merge isolated age + year
			if i < num-1:
				if s.isdigit() and sentence_list[i+1].startswith('y'):
					s = s + ' ' + sentence_list[i+1]
					i += 1

			# if next sentence starts with 'now measures', merge with current
			if i < num-1:
				if sentence_list[i+1].startswith('now measures'):
					s = s + ' ' + sentence_list[i+1]
					i += 1

			sentences.append(s)
			i += 1

	return sentences
