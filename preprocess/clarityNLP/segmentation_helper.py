#!/usr/bin/env python3
"""

This is a helper module for segmentation.py.

For import only.

"""

import re2 as re
import os
import sys
import json

from clarityNLP.date_finder import run as run_date_finder, DateValue, EMPTY_FIELD as EMPTY_DATE_FIELD
from clarityNLP.size_measurement_finder import run as run_size_measurement, SizeMeasurement, EMPTY_FIELD as EMPTY_SMF_FIELD

VERSION_MAJOR = 0
VERSION_MINOR = 2

# set to True to enable debug output
TRACE = False

MODULE_NAME = 'segmentation_helper.py'

FIRST_NAME_TOKEN = 'firstnametok'
LAST_NAME_TOKEN = 'lastnametok'
DOCTOR_FIRST_NAME_TOKEN = 'doctorfirstnametok'
DOCTOR_LAST_NAME_TOKEN = 'doctorlastnametok'
NAME_TOKEN = 'nametok'
NAME_PREFIX_TOKEN = 'nameprefixtok'
ADDRESS_TOKEN = 'addresstok'
LOCATION_TOKEN = 'locationtok'
HOSPITAL_TOKEN = 'hospitaltok'
PO_BOX_TOKEN = 'poboxtok'
STATE_TOKEN = 'statenametok'
COUNTRY_TOKEN = 'countrynametok'
COMPANY_TOKEN = 'companynametok'
TELEPHONE_NUMBER_TOKEN = 'telephonenumbertok'
PAGER_NUMBER_TOKEN = 'pagernumbertok'
SSN_TOKEN = 'ssntok'
MEDICAL_RECORD_NUMBER_TOKEN = 'medicalrecordnumbertok'
UNIT_NUMBER_TOKEN = 'unitnumbertok'
AGE_OVER_90_TOKEN = 'ageover90tok'
EMAIL_ADDRESS_TOKEN = 'emailaddresstok'
URL_TOKEN = 'urladdresstok'
HOLYDAY_TOKEN = 'holydaynametok'
JOB_NUMBER_TOKEN = 'jobnumbertok'
MD_NUMBER_TOKEN = 'mdnumbertok'
DATE_RANGE_TOKEN = 'daterangetok'
NUMERIC_IDENTIFIER_TOKEN = 'numericidentifiertok'
DATE_LITERAL_TOKEN = 'datetok'
UNIVERSITY_TOKEN = 'universitytok'
DICTATOR_INFO_TOKEN = 'dictatorinfotok'
CC_CONTACT_INFO_TOKEN = 'cccontactinfotok'
CLIP_NUMBER_TOKEN = 'clipnumbertok'
SERIAL_NUMBER_TOKEN = 'serialnumbertok'
ATTENDING_INFO_TOKEN = 'attendinginfotok'
PROVIDER_NUMBER_TOKEN = 'providernumbertok'

# regex for locating a PHI [** ... **]

str_anon_start = r'(?P<anon>\[\*\*('
#str_anon_date = r'(?P<anon_date>([0-9]{4}-[0-9]{1,2}-[0-9]{1,2})|([0-9]{4})|([0-9]{4}-[0-9]{1,2})|([0-9]{1,2}-[0-9]{1,2}))'
str_anon_date = r'(?P<anon_date>([0-9/\-]+)|([0-9]+))'
str_sep1 = r'|((('
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
str_anon_end = r')(\(?[0-9]+\)?)?((\s?(\(.*?\)\s)?)|(\s))(?P<anon_id>[0-9]+)?)))\*\*\])'
str_anon = r''.join([str_anon_start, str_anon_date, str_sep1, str_anon_tokens, str_anon_end])
#regex_anon = re.compile(str_anon)

# regex for locating a contrast agent expression
str_contrast = r'(?P<contrast>\bContrast:\s+(None|[a-zA-Z]+\s+Amt:\s+\d+(cc|CC)?))'
#regex_contrast = re.compile(str_contrast)

# regex for locating a field of view expression
str_fov = r'(?P<fov>\bField of view:\s+\d+)'
#regex_fov = re.compile(str_fov)

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

# prescription information
str_word		 = r'\b[-a-z]+\b'
str_words		= r'(' + str_word + r'\s*)*' + str_word
str_drug_name	= r'\b[-A-Za-z]+(/[-A-Za-z]+)?\b'
str_amount_num   = r'\d+(\.\d+)?'
str_amount	   = r'(' + str_amount_num + r'(/' + str_amount_num + r')?)?'
str_units		= r'\b[a-z]+\.?'
str_abbrev	   = r'([a-zA-Z]\.){1,3}'
str_abbrevs	  = r'(' + str_abbrev + r'\s+)*' + str_abbrev
str_prescription = r'(?P<prescription>' + str_drug_name + r'\s+' + str_amount + r'\s*' + str_units + \
				   r'\s+' + str_abbrevs + r'\s+' + str_words + r')'
#regex_prescription = re.compile(str_prescription)

# vitals
str_sep	  = r'([-:=\s]\s*)?'
str_temp	 = r'\b(T\.?|Temp\.?|Temperature)' + str_sep +\
			   r'(' + str_words + r')?' + str_amount_num + r'\s*'
str_height   = r'\b(Height|Ht\.?)' + str_sep + r'(\(in\.?\):?\s*)?' +\
			   str_amount_num + r'\s*(inches|in\.?|feet|ft\.?|meters|m\.?)?\s*'
str_weight   = r'\b(Weight|Wt\.?)' + str_sep + r'(\(lbs?\.?\):?\s*)?' +\
			   str_amount_num + r'\s*(grams|gm\.?|g\.?|ounces|oz\.?|pounds|lbs\.?|kilograms|kg\.?)?\s*'
str_bsa	  = r'\bBSA:?\s+(\(m2\):?\s*)?' + str_amount_num + r'(\s+m2)?\s*'
str_bp	   = r'\bBP' + str_sep + r'(\(mm\s+hg\):?\s*)?\d+/\d+\s*'
str_hr	   = r'\b(Pulse|P|HR)' + str_sep + r'(\(bpm\):?\s*)?' + str_amount_num + r'\s*'
str_rr	   = r'\bRR?' + str_sep + r'(' + str_words + r')?' + str_amount_num + r'\s*'
str_o2	   = r'\b(SpO2%?|SaO2|O2Sats?|O2\s+sat|O2\s+Flow|Sats?|POx|O2)' + str_sep +\
			   r'(' + str_words + r')?' + str_amount_num + r'(/bipap|\s*%?\s*)' +\
			   r'((/|on\s+)?(RA|NRB)|\dL(/|\s*)?NC|on\s+\d\s*L\s+(FM|NC|RA|NRB)|/?\dL)?'
str_status   = r'\bStatus:\s+(In|Out)patient\s*'
str_vitals   = r'(?i:(?P<vitals>(' + str_temp + r'|' + str_height + r'|' + str_weight + r'|' +\
			   str_bsa + r'|' + str_bp + r'|' + str_hr + r'|' + str_rr + r'|' +\
			   str_status + r'|' + str_o2 + r')+))'
#regex_vitals = re.compile(str_vitals, re.IGNORECASE)

# abbreviations
str_weekday  = r'\b(Mon|Tues|Wed|Thurs|Thur|Thu|Fri|Sat|Sun)\.'
str_h_o	  = r'\b\.?H/O'
str_r_o	  = r'\br/o(ut)?'
str_with	 = r'\bw/'
str_am_pm	= r'\b(a|p)\.m\.'
str_time	 = r'(2[0-3]|1[0-9]|[0-9]):[0-5][0-9]\s*(a|A|p|P)(\s*\.)?(m|M)(\s*\.)?'
str_s_p	  = r'\bs/p'
str_r_l	  = r'\b(Right|Left)\s+[A-Z]+'
str_sust_rel = r'\bSust\.?\s*Rel\.?'
str_sig	  = r'\bSig\s*:\s*[a-z0-9]+'
str_abbrev   = r'(?i:(?P<abbrev>' + str_weekday + r'|' + str_h_o + r'|' + str_r_o + r'|'   +\
			   str_with + r'|' + str_time + r'|' + str_am_pm + r'|'		  +\
			   str_s_p + r'|' + str_r_l + r'|' + str_sust_rel + r'|'		 +\
			   str_sig + r'))'
#regex_abbrev = re.compile(str_abbrev, re.IGNORECASE)

# gender
str_gender   = r'(?i:(?P<gender>\b(sex|gender)\s*:\s*(male|female|m\.?|f\.?)))'
#regex_gender = re.compile(str_gender, re.IGNORECASE)

expr_list = [str_abbrev,
			str_vitals,
			str_caps_header,
			str_anon,
			str_contrast,
			str_fov,
			str_prescription,
			str_gender
			]
expr = r'|'.join(expr_list)
regex = re.compile(expr, max_mem=(300 << 20))

###############################################################################
def enable_debug():
	global TRACE
	TRACE = True


###############################################################################
def disable_debug():
	global TRACE
	TRACE = False


###############################################################################
def find_size_meas_subs(report, subs, text):

	json_string = run_size_measurement(report)
	if '[]' == json_string:
		return report
	
	json_data = json.loads(json_string)

	# unpack JSON result into a list of SizeMeasurement namedtuples
	measurements = [SizeMeasurement(**m) for m in json_data]

	counter = 0
	prev_end = 0
	new_report = ''
	for m in measurements:
		chunk1 = report[prev_end:m.start]
		replacement = '{0}{1:03}'.format(text, counter)
		new_report += chunk1 + replacement
		prev_end = m.end
		subs[replacement] = m.text
		counter += 1
	new_report += report[prev_end:]

	return new_report


###############################################################################
#def find_substitutions(report, regex, sub_list, text):
#	"""
#	"""
#
#	subs = []
#	iterator = regex.finditer(report)
#	for match in iterator:
#		subs.append( (match.start(), match.end(), match.group()) )
#
#	if 0 == len(subs):
#		return report
#
#	counter = 0
#	prev_end = 0
#	new_report = ''
#	for start, end, match_text in subs:
#		chunk1 = report[prev_end:start]
#		replacement = '{0}{1:03}'.format(text, counter)
#		new_report += chunk1 + replacement
#		prev_end = end
#		sub_list.append( (replacement, match_text) )
#		counter += 1
#	new_report += report[prev_end:]
#
#	return new_report
		
		
###############################################################################

def do_substitutions(report):

	subs = {}
	size_meas_subs = {}

	token_counter = 0
	anon_counter = 0
	anon_date_counter = 0
	contrast_counter = 0
	fov_counter = 0
	size_meas_counter = 0
	header_counter = 0
	prescription_counter = 0
	vitals_counter = 0
	abbrev_counter = 0
	gender_counter = 0
	
	def repl(mo):
	
		nonlocal token_counter
		nonlocal anon_counter
		nonlocal anon_date_counter
		nonlocal contrast_counter
		nonlocal fov_counter
		nonlocal size_meas_counter
		nonlocal header_counter
		nonlocal prescription_counter
		nonlocal vitals_counter
		nonlocal abbrev_counter
		nonlocal gender_counter
	
		text = mo.string[mo.start():mo.end()]
	
		if mo.group('abbrev'):
			abbrev_counter += 1
			replacement = 'ABBREV{0}'.format(abbrev_counter)
			subs[replacement] = text
			return ' {0} '.format(replacement)
		elif mo.group('vitals'):
			vitals_counter += 1
			replacement = 'VITALS{0}'.format(vitals_counter)
			subs[replacement] = text
			return ' {0} '.format(replacement)
		if mo.group('caps_header'):
			header_counter += 1
			replacement = 'HEADER{0}'.format(header_counter)
			subs[replacement] = text
			return ' {0} '.format(replacement)
		elif mo.group('anon') and mo.group('anon_date'):
			#anon_counter += 1
			#replacement = 'ANONDATE{0}'.format(anon_date_counter)
			#subs[replacement] = mo.group('anon_date')
			return mo.group('anon_date').replace('/', '').replace('-', '/')
		elif mo.group('anon') and mo.group('anon_first_name'):
			return FIRST_NAME_TOKEN
		elif mo.group('anon') and mo.group('anon_last_name'):
			return LAST_NAME_TOKEN
		elif mo.group('anon') and mo.group('anon_doctor_first_name'):
			return DOCTOR_FIRST_NAME_TOKEN
		elif mo.group('anon') and mo.group('anon_doctor_last_name'):
			return DOCTOR_LAST_NAME_TOKEN
		elif mo.group('anon') and mo.group('anon_name'):
			return NAME_TOKEN
			#elif mo.group('anon') and mo.group('anon_id'):
			#	return ' ANONID{0} '.format(mo.group('anon_id'))
		elif mo.group('anon') and mo.group('anon_name_prefix'):
			return NAME_PREFIX_TOKEN
		elif mo.group('anon') and mo.group('anon_address'):
			return ADDRESS_TOKEN
		elif mo.group('anon') and mo.group('anon_university'):
			return UNIVERSITY_TOKEN
		elif mo.group('anon') and mo.group('anon_location'):
			return LOCATION_TOKEN
		elif mo.group('anon') and mo.group('anon_hospital'):
			return HOSPITAL_TOKEN
		elif mo.group('anon') and mo.group('anon_po_box'):
			return PO_BOX_TOKEN
		elif mo.group('anon') and mo.group('anon_state'):
			return STATE_TOKEN
		elif mo.group('anon') and mo.group('anon_country'):
			return COUNTRY_TOKEN
		elif mo.group('anon') and mo.group('anon_company'):
			return COMPANY_TOKEN
		elif mo.group('anon') and mo.group('anon_telephone_number'):
			return TELEPHONE_NUMBER_TOKEN
		elif mo.group('anon') and mo.group('anon_pager_number'):
			return PAGER_NUMBER_TOKEN
		elif mo.group('anon') and mo.group('anon_social_security_number'):
			return SSN_TOKEN
		elif mo.group('anon') and mo.group('anon_medical_record_number'):
			return MEDICAL_RECORD_NUMBER_TOKEN
		elif mo.group('anon') and mo.group('anon_unit_number'):
			return UNIT_NUMBER_TOKEN
		elif mo.group('anon') and mo.group('anon_age_over_90'):
			return AGE_OVER_90_TOKEN
		elif mo.group('anon') and mo.group('anon_email_address'):
			return EMAIL_ADDRESS_TOKEN
		elif mo.group('anon') and mo.group('anon_url'):
			return URL_TOKEN
		elif mo.group('anon') and mo.group('anon_holiday'):
			return HOLYDAY_TOKEN
		elif mo.group('anon') and mo.group('anon_job_number'):
			return JOB_NUMBER_TOKEN
		elif mo.group('anon') and mo.group('anon_md_number'):
			return MD_NUMBER_TOKEN
		elif mo.group('anon') and mo.group('anon_date_range'):
			return DATE_RANGE_TOKEN
		elif mo.group('anon') and mo.group('anon_numeric_identifier'):
			return NUMERIC_IDENTIFIER_TOKEN
		elif mo.group('anon') and mo.group('anon_date_literal'):
			return DATE_LITERAL_TOKEN
		elif mo.group('anon') and mo.group('anon_dictator_info'):
			return DICTATOR_INFO_TOKEN
		elif mo.group('anon') and mo.group('anon_cc_contact_info'):
			return CC_CONTACT_INFO_TOKEN
		elif mo.group('anon') and mo.group('anon_clip_number'):
			return CLIP_NUMBER_TOKEN
		elif mo.group('anon') and mo.group('anon_serial_number'):
			return SERIAL_NUMBER_TOKEN
		elif mo.group('anon') and mo.group('anon_attending_info'):
			return ATTENDING_INFO_TOKEN
		elif mo.group('anon') and mo.group('anon_provider_number'):
			return PROVIDER_NUMBER_TOKEN
		elif mo.group('anon'):
			anon_counter += 1
			replacement = 'ANON{0}'.format(anon_counter)
			subs[replacement] = text
			return ' {0} '.format(replacement)
		elif mo.group('contrast'):
			contrast_counter += 1
			replacement = 'CONTRAST{0}'.format(contrast_counter)
			subs[replacement] = text
			return ' {0} '.format(replacement)
		elif mo.group('fov'):
			fov_counter += 1
			replacement = 'FOV{0}'.format(fov_counter)
			subs[replacement] = text
			return ' {0} '.format(replacement)
		elif mo.group('prescription'):
			prescription_counter += 1
			replacement = 'PRESCRIPTION{0}'.format(prescription_counter)
			subs[replacement] = text
			return ' {0} '.format(replacement)
		elif mo.group('gender'):
			gender_counter += 1
			replacement = 'GENDER{0}'.format(gender_counter)
			subs[replacement] = text
			return ' {0} '.format(replacement)
		else:
			token_counter += 1
			replacement = 'TOKEN{0}'.format(token_counter)
			subs[replacement] = text
			return ' {0} '.format(replacement)

	report = regex.sub(repl, report)
	report = find_size_meas_subs(report, size_meas_subs, 'MEAS')

	#if TRACE:
	#	print('REPORT AFTER SUBSTITUTIONS: \n' + report + '\n')

	return (report, subs, size_meas_subs)

###############################################################################
def replace_text(sentence_list, sub_list):

	num = len(sentence_list)
	for i in range(num):
		count = 0
		#replacements = []
		sentence = sentence_list[i]
		for entry in sub_list:
			sub = entry[0]
			orig = entry[1]
			if -1 != sentence.find(sub):
				sentence = sentence.replace(sub, orig)
				#replacements.append(sub)
				count += 1
		yield sentence
		
		# remove used entries from sub_list
		if count > 0:
			sub_list = sub_list[count:]
	
###############################################################################
def undo_substitutions(sentence_list, subs, size_meas_subs):
	
	# undo in reverse order from that in 'do_substitutions'
	
	def multiple_replace(sentence_list, regex):
		nonlocal subs
		# For each match, look-up corresponding value in dictionary  
		return (regex.sub(lambda mo: subs[mo.string[mo.start():mo.end()]], sentence) for sentence in sentence_list)
	
	# Create a regular expression  from the dictionary keys
	regex = re.compile("({0})".format("|".join(map(re.escape, subs.keys()))), max_mem=(300 << 20))
	
	sentence_list = multiple_replace(sentence_list, regex)
	sentence_list = list(replace_text(sentence_list, size_meas_subs))

	return sentence_list

###############################################################################
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


###############################################################################
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
	

###############################################################################
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

###############################################################################
def split_section_headers(sentence_list):
	"""
	Put an all caps section header in a separate sentence from the subsequent
	text.
	"""

	#sentences = []
	for s in sentence_list:
		subs = []
		iterator = regex_caps_header.finditer(s)
		for match in iterator:
			subs.append( (match.start(), match.end()) )
		if len(subs) > 0:
			prev_end = 0
			for start, end in subs:
				before = s[prev_end:start].strip()
				header = s[start:end].strip()
				prev_end = end
				if len(before) > 0:
					#sentences.append(before)
					yield(before)
				#sentences.append(header)
				yield header
			after = s[prev_end:].strip()
			if len(after) > 0:
				#sentences.append(after)
				yield after
		else:
			#sentences.append(s)
			yield s

	#return sentences
		

###############################################################################
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

	#sentences = []
	num = len(sentence_list)

	#for s in sentence_list:
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

		#sentences.append(s)
		yield s
		i += 1

	#return sentences


###############################################################################
def get_version():
	return '{0} {1}.{2}'.format(MODULE_NAME, VERSION_MAJOR, VERSION_MINOR)

