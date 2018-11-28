#!/usr/bin/env python3

import re2 as re
import hashlib
from spacy.attrs import ORTH

#medical abbreviations from
#https://en.wikipedia.org/wiki/List_of_abbreviations_used_in_medical_prescriptions
#https://en.wikipedia.org/wiki/List_of_medical_abbreviations:_Latin_abbreviations

#pattern1 -> tok. tok.
#pattern2 -> tok tok.

abbrev_composite_pattern2 = ['ad lib.'
										'ad us.'
										'bis ind.'
										'ex aq.'
										'non rep.'
										]

#'d. in p. ae.'
#'si op. sit'

#vitals
#Temp.
#measure units
		
abbrev_composite_pattern1 = ['alt. d.'
										'alt. dieb.'
										'alt. h.'
										'alt. hor.'
										'aq. bull.'
										'aq. com.'
										'aq. dest.'
										'aq. ferv.'
										'cochl. ampl.'
										'cochl. infant.'
										'cochl. mag.'
										'cochl. mod.'
										'cochl. parv.'
										'cyath. vinos.'
										'dieb. alt.'
										'f. pil.'
										'hor. alt.'
										'hor. decub.'
										'hor. intermed.'
										'hor. tert.'
										'lat. dol.'
										'mod. praescript.'
										'omn. bih.'
										'omn. hor.'
										'part. aeq.'
										]
										
def generate_matcher_pattern1():
	for abbrev in abbrev_composite_pattern1:
		w1, w2 = [w.strip() for w in abbrev.split('.')][:2]
		yield ('{}_{}_1'.format(w1, w2), [{ORTH: w1}, {ORTH: '.'}, {ORTH: w2}, {ORTH: '.'}])
		yield ('{}_{}_2'.format(w1, w2), [{ORTH: w1+'.'}, {ORTH: w2}, {ORTH: '.'}])
		yield ('{}_{}_3'.format(w1, w2), [{ORTH: w1}, {ORTH: '.'}, {ORTH: w2+'.'}])

abbrev_list = {'a.c.': [{ORTH: 'a.c.'}],
					'a.c.h.s.': [{ORTH: 'a.c.h.s.'}],
					'ac&hs': [{ORTH: 'ac&hs'}],
					'a.d.': [{ORTH: 'a.d.'}],
					'ad.': [{ORTH: 'ad.'}],
					'add.': [{ORTH: 'add.'}],
					'admov.': [{ORTH: 'admov.'}],
					'aeq.': [{ORTH: 'aeq.'}],
					'agit.': [{ORTH: 'agit.'}],
					'amp.': [{ORTH: 'amp.'}],
					'aq.': [{ORTH: 'aq.'}],
					'a.l.': [{ORTH: 'a.l.'}],
					'a.s.': [{ORTH: 'a.s.'}],
					'a.u.': [{ORTH: 'a.u.'}],
					'b.d.s.': [{ORTH: 'b.d.s.'}],
					'bib.': [{ORTH: 'bib.'}],
					'b.i.d.': [{ORTH: 'b.i.d.'}],
					'b.d.': [{ORTH: 'b.d.'}],
					'bol.': [{ORTH: 'bol.'}],
					'Ph.Br.': [{ORTH: 'Ph.Br.'}],
					'b.t.': [{ORTH: 'b.t.'}],
					'bucc.': [{ORTH: 'bucc.'}],
					'cap.': [{ORTH: 'cap.'}],
					'caps.': [{ORTH: 'caps.'}],
					'c.m.': [{ORTH: 'c.m.'}],
					'c.m.s.': [{ORTH: 'c.m.s.'}],
					'c.': [{ORTH: 'c.'}],
					'cib.': [{ORTH: 'cib.'}],
					'c.c.': [{ORTH: 'c.c.'}],
					'cf.': [{ORTH: 'cf.'}],
					'c.n.': [{ORTH: 'c.n.'}],
					'cochl.': [{ORTH: 'cochl.'}],
					'colet.': [{ORTH: 'colet.'}],
					'comp.': [{ORTH: 'comp.'}],
					'contin.': [{ORTH: 'contin.'}],
					'cpt.': [{ORTH: 'cpt.'}],
					'cr.': [{ORTH: 'cr.'}],
					'cuj.': [{ORTH: 'cuj.'}],
					'c.v.': [{ORTH: 'c.v.'}],
					'cyath.': [{ORTH: 'cyath.'}],
					'd.': [{ORTH: 'd.'}],
					'D/C': [{ORTH: 'D/C'}],
					'decoct.': [{ORTH: 'decoct.'}],
					'det.': [{ORTH: 'det.'}],
					'dil.': [{ORTH: 'dil.'}],
					'dim.': [{ORTH: 'dim.'}],
					'disp.': [{ORTH: 'disp.'}],
					'div.': [{ORTH: 'div.'}],
					'd.t.d. .': [{ORTH: 'd.t.d. .'}],
					'elix.': [{ORTH: 'elix.'}],
					'e.m.p.': [{ORTH: 'e.m.p.'}],
					'emuls.': [{ORTH: 'emuls.'}],
					'exhib.': [{ORTH: 'exhib.'}],
					'f.': [{ORTH: 'f.'}],
					'f.h.': [{ORTH: 'f.h.'}],
					'fl.': [{ORTH: 'fl.'}],
					'fld.': [{ORTH: 'fld.'}],
					'f.m.': [{ORTH: 'f.m.'}],
					'f.s.a.': [{ORTH: 'f.s.a.'}],
					'ft.': [{ORTH: 'ft.'}],
					'garg.': [{ORTH: 'garg.'}],
					'gr.': [{ORTH: 'gr.'}],
					'gtt.': [{ORTH: 'gtt.'}],
					'gutt.': [{ORTH: 'gutt.'}],
					'h.': [{ORTH: 'h.'}],
					'H/O': [{ORTH: 'H/O'}],
					'hor.': [{ORTH: 'hor.'}],
					'habt.': [{ORTH: 'habt.'}],
					'h.s.': [{ORTH: 'h.s.'}],
					'inj.': [{ORTH: 'inj.'}],
					'i.m.': [{ORTH: 'i.m.'}],
					'inf.': [{ORTH: 'inf.'}],
					'i.v.': [{ORTH: 'i.v.'}],
					'i.v.p.': [{ORTH: 'i.v.p.'}],
					'lb.': [{ORTH: 'lb.'}],
					'l.c.d.': [{ORTH: 'l.c.d.'}],
					'liq.': [{ORTH: 'liq.'}],
					'lot.': [{ORTH: 'lot.'}],
					'M.': [{ORTH: 'M.'}],
					'm.': [{ORTH: 'm.'}],
					'max.': [{ORTH: 'max.'}],
					'm.d.u.': [{ORTH: 'm.d.u.'}],
					'mg/dL': [{ORTH: 'mg/dL'}],
					'min.': [{ORTH: 'min.'}],
					'mist.': [{ORTH: 'mist.'}],
					'mit.': [{ORTH: 'mit.'}],
					'mitt.': [{ORTH: 'mitt.'}],
					'nebul.': [{ORTH: 'nebul.'}],
					'neb.': [{ORTH: 'neb.'}],
					'noct.': [{ORTH: 'noct.'}],
					'n.p.o.': [{ORTH: 'n.p.o.'}],
					'1/2NS.': [{ORTH: '1/2NS.'}],
					'o.d.': [{ORTH: 'o.d.'}],
					'o.m.': [{ORTH: 'o.m.'}],
					'o.n.': [{ORTH: 'o.n.'}],
					'o.s.': [{ORTH: 'o.s.'}],
					'o.u.': [{ORTH: 'o.u.'}],
					'p.': [{ORTH: 'p.'}],
					'p.c.': [{ORTH: 'p.c.'}],
					'p.c.h.s.': [{ORTH: 'p.c.h.s.'}],
					'pc&hs': [{ORTH: 'pc&hs'}],
					'Ph.Br.': [{ORTH: 'Ph.Br.'}],
					'Ph.Eur.': [{ORTH: 'Ph.Eur.'}],
					'Ph.Int.': [{ORTH: 'Ph.Int.'}],
					'pig.': [{ORTH: 'pig.'}],
					'pigm.': [{ORTH: 'pigm.'}],
					'p.o.': [{ORTH: 'p.o.'}],
					'ppt.': [{ORTH: 'ppt.'}],
					'p.r.': [{ORTH: 'p.r.'}],
					'p.r.n.': [{ORTH: 'p.r.n.'}],
					'pt.': [{ORTH: 'pt.'}],
					'pulv.': [{ORTH: 'pulv.'}],
					'p.v.': [{ORTH: 'p.v.'}],
					'q.1.d.': [{ORTH: 'q.1.d.'}],
					'q.1.h.': [{ORTH: 'q.1.h.'}],
					'q.2.h.': [{ORTH: 'q.2.h.'}],
					'q.4.h.': [{ORTH: 'q.4.h.'}],
					'q.6.h.': [{ORTH: 'q.6.h.'}],
					'q.8.h.': [{ORTH: 'q.8.h.'}],
					'q.a.d.': [{ORTH: 'q.a.d.'}],
					'q.a.m.': [{ORTH: 'q.a.m.'}],
					'q.d.': [{ORTH: 'q.d.'}],
					'q.d.a.m.': [{ORTH: 'q.d.a.m.'}],
					'q.d.p.m.': [{ORTH: 'q.d.p.m.'}],
					'q.d.s.': [{ORTH: 'q.d.s.'}],
					'q.p.m.': [{ORTH: 'q.p.m.'}],
					'q.h.': [{ORTH: 'q.h.'}],
					'q.h.s.': [{ORTH: 'q.h.s.'}],
					'q.i.d.': [{ORTH: 'q.i.d.'}],
					'q.l.': [{ORTH: 'q.l.'}],
					'q.n.': [{ORTH: 'q.n.'}],
					'q.o.d.': [{ORTH: 'q.o.d.'}],
					'q.p.m.': [{ORTH: 'q.p.m.'}],
					'q.q.': [{ORTH: 'q.q.'}],
					'q.q.h.': [{ORTH: 'q.q.h.'}],
					'q.s.': [{ORTH: 'q.s.'}],
					'q.v.': [{ORTH: 'q.v.'}],
					'rep.': [{ORTH: 'rep.'}],
					'rept.': [{ORTH: 'rept.'}],
					'R/L': [{ORTH: 'R/L'}],
					'rep.': [{ORTH: 'rep.'}],
					's.': [{ORTH: 's.'}],
					's.a.': [{ORTH: 's.a.'}],
					'sem.': [{ORTH: 'sem.'}],
					's.i.d.': [{ORTH: 's.i.d.'}],
					'Sig.': [{ORTH: 'Sig.'}],
					'S.': [{ORTH: 'S.'}],
					'sig.': [{ORTH: 'sig.'}],
					'sing.': [{ORTH: 'sing.'}],
					's.l.': [{ORTH: 's.l.'}],
					'sol.': [{ORTH: 'sol.'}],
					's.o.s.': [{ORTH: 's.o.s.'}],
					's.s.': [{ORTH: 's.s.'}],
					'st.': [{ORTH: 'st.'}],
					'Stat.': [{ORTH: 'Stat.'}],
					'sum.': [{ORTH: 'sum.'}],
					'supp.': [{ORTH: 'supp.'}],
					'susp.': [{ORTH: 'susp.'}],
					'syr.': [{ORTH: 'syr.'}],
					'tab.': [{ORTH: 'tab.'}],
					'tal.': [{ORTH: 'tal.'}],
					't.': [{ORTH: 't.'}],
					't.d.s.': [{ORTH: 't.d.s.'}],
					't.i.d.': [{ORTH: 't.i.d.'}],
					't.d.': [{ORTH: 't.d.'}],
					't.d.s.': [{ORTH: 't.d.s.'}],
					'tinct.': [{ORTH: 'tinct.'}],
					't.i.d.': [{ORTH: 't.i.d.'}],
					't.i.w.': [{ORTH: 't.i.w.'}],
					'top.': [{ORTH: 'top.'}],
					'tinc.': [{ORTH: 'tinc.'}],
					'tinct.': [{ORTH: 'tinct.'}],
					'trit.': [{ORTH: 'trit.'}],
					'troch.': [{ORTH: 'troch.'}],
					'u.d.': [{ORTH: 'u.d.'}],
					'ut.': [{ORTH: 'ut.'}],
					'dict.': [{ORTH: 'dict.'}],
					'ung.': [{ORTH: 'ung.'}],
					'vag.': [{ORTH: 'vag.'}],
					'w/a': [{ORTH: 'w/a'}],
					'w/f': [{ORTH: 'w/f'}],
					'w/o': [{ORTH: 'w/o'}],
					'y.o.': [{ORTH: 'y.o.'}],
					}

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

def do_substitutions(document, mode):
	
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

	document = regex.sub(repl, document)

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
