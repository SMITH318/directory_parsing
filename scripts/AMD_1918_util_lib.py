from enum import Enum
import re

# TODO: how to make more randomly felxible?? could try dropping individual characters and rety matching

RUN_TESTS=__name__ == '__main__'
DEBUGGING = False

LineType = Enum('LineType', 'UNKNOWN, STATE, CITY, DOC_START, DOC_TO_ADDR, DOC_TO_OFF, DOC_FULL')

# "STATE"
STATE_REGEX=r'(?P<state>[A-Z\s]+)'

# "City, XXX,XXX, County.
CITY_COUNTY_REGEX=r'(?P<city>[a-zA-Z\s.,()\']+),\s?(((?P<population>[0-9,]+?)|-),\s?)?(?P<county>[a-zA-Z().\s]+)'

#L*,*1*,*L. - at least one letter, followed by a comma, followed by a number, followed by a comma, followed by a letter (and an optional period)
LOOSE_CITY_COUNTY_REGEX=r'\w.+?,.+?\d.+?,.+?\w.+?'

#(b XXXX)
BIRTH_REGEX=r"(\(b['\s]?(?P<birth>[0-9]{2})\))"
# St.1,'01; OR *
# SCHOOL_YEAR_REGEX=r'(?P<sch_year>[0-9]{2})'
# SCHOOL_ID_REGEX=r'((?P<sch_id>[1-9lI][0-9]?)\s?,)'
# SCHOOL_STATE_REG_EX=r'((?P<sch_state>[A-Z][a-z]{1,3}|[O0]|N\.\s?Y|D\.\s?C|N\.\s?H|N\.\s?C|S\.\s?C)\.?)'

SCHOOL_YEAR_REGEX=r'[0-9]{2}'
SCHOOL_ID_REGEX=r'([1-9lI][0-9]?\s?,?)'
SCHOOL_STATE_REG_EX=r'(([A-Z][a-z]{1,3}|[O0]|N\.\s?Y|D\.\s?C|N\.\s?H|N\.\s?C|S\.\s?C)\.?)'
SCHOOL_WHOLE =r'(' + SCHOOL_STATE_REG_EX + r'\s?('+ SCHOOL_ID_REGEX + r'|[*])\s?((\'\s?'+SCHOOL_YEAR_REGEX + r')|[*]))'
SCHOOLS_REGEX=r'((?P<schs_raw>'+SCHOOL_WHOLE+r'(, '+SCHOOL_WHOLE+r')*\s?;)|(?P<no_sch_info>◊))'
#(l XX)
LICENSE_REGEX=r"(\(([lI][\s'](?P<lic_year>([0-9]{2})|t)|(?P<lic_prev>♁))\)\s?;?)"
ADDRESS_REGEX=r'([A-Z1-9][\w\s.,\'&½-]+?[A-Za-z][\w\s.\'&-]+)' #has to start with cap or number and have at least one non-starting letter
RD_ADDRESS_REGEX=r'(R. ?D.?[1-9]?)'
HOUR_RANGE_REGEX=r'(([1-9][0-9]?(\s?;30)?[.-]\s?[1-9][0-9]?(\s?;30)?)|((until|after)\s?[1-9][0-9]?))'
HOURS_REGEX= r'((?P<hours>'+HOUR_RANGE_REGEX+r'([,;]\s?'+HOUR_RANGE_REGEX+'){0,2});? ?)?'
SOCIETY_REGEX = r'(\((?P<societies>[A-G][1-9][A-G0-9,]*?)\);? ?)?'
ONE_SPECIALTY_REGEX = r"(S|Ob|G|ObG|Or|Pr|Op|A|LR|ALR|OALR|U|D|Pd|N|P|NP|I|l|T|Anes|CP|R|Path|Bact)[*★]?"
SPECIALTY_REGEX = r'((?P<specialty>'+ONE_SPECIALTY_REGEX+'(, ?'+ONE_SPECIALTY_REGEX+')*);? ?)?'
COMMISSION_REGEX = r'(?P<commission>▼[GN]?)?'


# "LAST, FIRST [(col.)][BIRTH][+]-SCHOOL
DOC_ENTRY_REGEX_TO_SCHOOL = (
    r'(?P<last_name>[a-zA-Z \'1]+),\s(?P<first_name>[a-zA-Z.,\s]+)\s?((?P<col>\(col\.\))\s?)?'+
    '('+BIRTH_REGEX+r'\s?)?(?P<break>([+*@⊕]?|-?[EH]?)\s?-)\s?'+
    SCHOOLS_REGEX+r';?\s?'
)
DOC_ENTRY_REGEX_TO_ADDRESS = (
    DOC_ENTRY_REGEX_TO_SCHOOL +
    LICENSE_REGEX+r'?\s?' +
    r'(?P<no_practice>not in practice;?\s?)?' +
    r'((?P<addr>'+ADDRESS_REGEX+'|'+RD_ADDRESS_REGEX+r');?\s?)' + # must have address
    r'(?<![;-]$)' # can't end with ; or -
)
DOC_ENTRY_REGEX_TO_OFFICE = (
    DOC_ENTRY_REGEX_TO_SCHOOL +
    LICENSE_REGEX+r'?\s?' +
    r'(?P<no_practice>not in practice;?\s?)?' +
    r'((?P<addr>'+ADDRESS_REGEX+'|'+RD_ADDRESS_REGEX+r');?\s?)?' + # can have address
    r'(\s?of(- )?fice,\s(?P<office>'+ADDRESS_REGEX+r');?\s?)' + # must have office, "(- )?" handles common line break
    r'(?<![;-]$)' # can't end with ; or -
)
# "LAST, FIRST [(col.)][BIRTH][+]-SCHOOL LICENSE [address][office, address][hours][specialty][commission]
DOC_ENTRY_REGEX = (
    DOC_ENTRY_REGEX_TO_SCHOOL +
    LICENSE_REGEX+r'?\s?' +
    r'(?P<no_practice>not in practice;?\s?)?' +
    r'((?P<addr>'+ADDRESS_REGEX+'|'+RD_ADDRESS_REGEX+r');? ?)?' + # can have address
    r'(\s?of(- )?fice,\s(?P<office>'+ADDRESS_REGEX+r');? ?)?' + # can have office, "(- )?" handles common line break
    HOURS_REGEX +
    SOCIETY_REGEX +
    SPECIALTY_REGEX +
    COMMISSION_REGEX +
    r'(?<![;-]$)' # can't end with ; or -
)

LOOSE_CITY_COUNTY_RE = re.compile(LOOSE_CITY_COUNTY_REGEX)
def is_loose_city(line):
    match = re.search(r"[⊕+*:;]", line) # shouldn't have these characters
    if match:
        return False
    return LOOSE_CITY_COUNTY_RE.match(line) is not None

STATE_RE = re.compile(STATE_REGEX)
def get_state(line):
    matched = STATE_RE.match(line)
    return matched.groupdict()

CITY_COUNTY_RE = re.compile(CITY_COUNTY_REGEX)
def get_city(line):
    matched = CITY_COUNTY_RE.match(line)
    return matched.groupdict()

DOC_ENTRY_RE = re.compile(DOC_ENTRY_REGEX)
def get_full_doctor(line):
    matched = DOC_ENTRY_RE.fullmatch(line)
##    if matched: 
##        print('matched: "', matched.group(0), '"', sep='')
##        print(matched.groupdict())
    return matched.groupdict()


DOC_ENTRY_RE_TO_SCHOOL = re.compile(DOC_ENTRY_REGEX_TO_SCHOOL)
DOC_ENTRY_RE_TO_ADDRESS = re.compile(DOC_ENTRY_REGEX_TO_ADDRESS)
DOC_ENTRY_RE_TO_OFFICE = re.compile(DOC_ENTRY_REGEX_TO_OFFICE)
def get_line_type(line, flag=0):
    if STATE_RE.fullmatch(line, flag):
        # is "STATE."
        return LineType.STATE
    matched = CITY_COUNTY_RE.fullmatch(line, flag)
    if matched:
        #is "City, XXX,XXX, County."
        # if DEBUGGING:
        #     print(matched.groupdict())
        return LineType.CITY
    matched = DOC_ENTRY_RE.match(line, flag) if DEBUGGING else DOC_ENTRY_RE.fullmatch(line, flag) # <========================================== make full
    if matched:
        if DEBUGGING:
            print('matched: "', matched.group(0), '"', sep='')
            print(matched.groupdict())
        # check if DOC_ENTRY_REGEX_TO_SCHOOL or DOC_ENTRY_REGEX_TO_OFFICE cause there might be more
        to_address = DOC_ENTRY_RE_TO_ADDRESS.fullmatch(line, flag)
        if to_address:
            return LineType.DOC_TO_ADDR
        to_office = DOC_ENTRY_RE_TO_OFFICE.fullmatch(line, flag)
        if to_office:
            return LineType.DOC_TO_OFF
        return LineType.DOC_FULL
    matched = DOC_ENTRY_RE_TO_SCHOOL.match(line, flag)
    if matched:
        # is "LAST, FIRST [(col.)][BIRTH][~|+]-"
        # if DEBUGGING:
        #     print('matched: "', matched.group(0), '"', sep='')
        #     print(matched.groupdict())
        return LineType.DOC_START
    
    return LineType.UNKNOWN


if RUN_TESTS:
    LICENSE_TESTS=[("(l'86)", '86'),
                ('(l 95)', '95'),
                ('(l 99)', '99'),
                ('(l t)', 't')]
    SPECIALTY_TESTS = [("D;", "D")]
    HOURS_TESTS=['12-2','9-11,3-5', 'until 7', 'after 6', '9-11;30', '9-11;30,3-5','10-11;30, 2.4;30', '10-11;30, 4.5']
    LINE_TYPE_TESTS = [
                    # ('LOUISIANA', LineType.STATE),
                    ('West Monroe, 775, Ouncldtn', LineType.CITY),
                    ('Whitford, -, Winn', LineType.CITY),
                    ('New Orleans, 312,457, Orleans', LineType.CITY),
                    ('BAYVIEW (WYLAM P.O.), JEFFERSON', LineType.CITY),
                    ('AUSTINVILLE (R.D., DECATUR), 1671, MORGAN', LineType.CITY),
                    ("BEULAH, (R.F.D., BBLANTON), 118, LEE", LineType.CITY),
                    ("GAAR, J. ALBERT (b'80) - Tenn.8,'04;", LineType.DOC_START),
                    ("Aubrey, A. J. (col.) (b'73)-La.4,'99;", LineType.DOC_START),
                    ("SCOTT, W. S.-◊ (l'03)", LineType.DOC_FULL),
                    ("Moore, Elisha B.-◊; (l 78); not in practice", LineType.DOC_FULL),
                    ("OWENS, SEABORN WESLEY-◊; (l 87)", LineType.DOC_FULL),
                    ("Kirven, Thos. C.-Ky.4,'93; (l 93)", LineType.DOC_FULL),
                    ("McKOWEN, EMMETT C. (b 62)-La.l,'86; (l'86); College and High Sts.; 12-2", LineType.DOC_FULL),
                    ("HENBY, EUGENE L. (b 74)+-La.1,'97;(l 95); Water St.; 9-11,3-5", LineType.DOC_FULL),
                    ("Aubrey, A. J. (col.) (b 73)-La.4,'99;(l 99)", LineType.DOC_FULL),
                    ("ARCHINARD, PAUL E.-La.1,'82; (l 82); 1219 N. Rampart St. ; office, 211 Camp St.; 12-2", LineType.DOC_FULL),
                    ("Polk, Wm. T. (b 78)-Tenn.5,'02; (l 06); office, 3d and Murray Sts.", LineType.DOC_TO_OFF),
                    ("Jones, Jas. P. (col.)-Tenn.7,'93; (l 93)", LineType.DOC_FULL),
                    ("Jones, Fred R.-Ind. 7,'87; (l t)", LineType.DOC_FULL),
                    ("FONTAINE, BRYCE W. (b'77) + - Tex.2,'96; (l'01)", LineType.DOC_FULL),
                    ("EDWARDS, CLARENCE J.-Ky.2,'83 ; (I 83)", LineType.DOC_FULL),
                    ("Gcoffrion, Victor-Que.3,'01 ; (l 04)", LineType.DOC_FULL),
                    ("De Poincy, Edgar S.-La.*'81; (l 81); 1227 Esplanade Ave.", LineType.DOC_TO_ADDR),
                    ("Wailes, L. A.- Pa.2,*; (l 61); 2128 Berlin", LineType.DOC_TO_ADDR),
                    ("Lines, Ezra A.-H-◊ (l 97); 1940 N. Rampart St.; 8-12", LineType.DOC_FULL),
                    ("Duperier, Douglas-Mich.1,'95; (l 95)", LineType.DOC_FULL),
                    ("MENVILLE, LEON J.-Md.9,'04; (l 04) ; 9- 11;30, 3-5", LineType.DOC_FULL),
                    ("Belden, Jas. W.-La.1,'88; 1403 Louisiana Ave.; office, 830 Canal St.; 1-4", LineType.DOC_FULL),
                    ("Danos, Joseph L. (b 80)-La.1,'03; not in practice", LineType.DOC_FULL),
                    ("GLAZE, ANDREW LEWIS, JR. (b'88)⊕- Tenn.5,'12; (l 13); D; ▼", LineType.DOC_FULL),
                    ("Henry, Stewart L.-La.1,'66; (l 66); 908 Carrollton Ave.; 7-8, 12-2", LineType.DOC_FULL),
                    ("HOEFELD, ADOLPH O.-La.1,'01; (l 01); 830 Canal St.; 1-3", LineType.DOC_FULL),
                    ("Jordan, Harrison-Tenn.11,'02; (l 02); not in practice", LineType.DOC_FULL),
                    ("Layton, Thos. B.-La.1,'01; (l 01); 1420 Josephine St.", LineType.DOC_TO_ADDR),
                    ("Marks, L. H.- (l 06)", LineType.UNKNOWN), #school required
                    ("LITTELL, ROBT. M.-La.1 ,'94; (l 94)", LineType.DOC_FULL),
                    ("MANBOULES, J. P., Jr.- (l 99)", LineType.UNKNOWN), #school required
                    ("FOSSIER, A. EMILE (b 81) - La.1,'02; (l 02); 1215 Carrollton Ave.; 7-9", LineType.DOC_FULL), 
                    ("POSTELL, LAURENS T. (b 59)+-La.1,'82; (l 82); office, Holloway & Postell Drug Store; 3-5", LineType.DOC_FULL),
                    ("Smith, Temple B. (b'92)-Mo.27,'92; (l t); 664 7th St.; office, 817½ Ryan St.; 10-11;30, 2-4;30", LineType.DOC_FULL),
                    ("PAINE, RUFFIN B. (b'65) + - La.1,'88; (l 88) ; Lake and Coffee Ste.; 10-11;30, 4-5", LineType.DOC_FULL),
                    ("BRUNS, HENRY D. (b 59)+-Pa.2,'81; (l 81); 2308 Prytania St.; office, 211 Camp St.; 12-4", LineType.DOC_FULL),
                    ("GOGGANS, JAMES ADRIAN (b'54)-N.Y.5, '77; (l 82); (A628); S", LineType.DOC_FULL), # society
                    ("Jones, Lee G. (b'73)-Ga.1,'96, Tenn.11,'98; (l t)", LineType.DOC_FULL), # mult schools
                    ("LACEY, EDWARD PARISH (b'56)⊕- Tenn.5,'83; (l 83); 1802, 8th Ave.; office, Realty Bldg", LineType.DOC_TO_OFF),
                    ("NASH, SAML. F. (b'77)⊕-Ala.4,'08; (l 08); 1915 Berkley Ave.; office, 1831½, 2d Ave.; 8-9, 2-3, 5-7", LineType.DOC_FULL),
                    ("POWELL, HENRY BURON (b'84)-Ala.2, '10; (l 10); 1918 Clarendon Ave.; office,", LineType.DOC_START), # I think this behavior is ok
                    ("Spencer, Lucian Allen (b'62)-O.9,'85; (l 85); 1706, 2d Ave.; office, McDonald", LineType.DOC_TO_OFF),
                    ("WALLER, GEO. DE ILOACH (b'70)-Tenn.5, '99; (l 99); 1710, 4th Ave.; office, 210½ 19th St.; 10-12, 2-4", LineType.DOC_FULL),
                    ("Wilborn, Daniel W. (col.) (b'80)-N.C.3, '09; (l 10) ; 1701 Mulberry St", LineType.DOC_TO_ADDR),
                    ("Christian, James Saml. (b'84)-Ala. 4'12; (l 12); R.D.2", LineType.DOC_TO_ADDR),
                    ]
    LOOSE_CITY_TESTS =[
        'Arca11ia, 924, Bienville.',
        'Amite, 1,G47, Tangipaltoa.',
        'Allenton-n, 250, Bos.oier.'
    ]


    any_bad=0
    for test, result in LICENSE_TESTS:
        matched = re.fullmatch(LICENSE_REGEX, test)
        if matched is None or matched.group('lic_year') != result:
            any_bad+=1
            print('bad test: "', test, '" returned ', matched, '\n', sep='')
    for test, result in SPECIALTY_TESTS:
        matched = re.fullmatch(SPECIALTY_REGEX, test)
        if matched is None or matched.group('specialty') != result:
            any_bad+=1
            print('bad test: "', test, '" returned ', matched, '\n', sep='')
    for test in HOURS_TESTS:
        matched = re.fullmatch(HOURS_REGEX, test)
        #print(matched)
        if matched is None or matched.group() != test:
            any_bad+=1
            print('bad test: "', test, '" returned ', matched, '\n', sep='')
    for test, result in LINE_TYPE_TESTS:
        test_result = get_line_type(test)#, re.DEBUG)
        if test_result != result:
            any_bad+=1
            print('bad test: "', test, '" returned ', test_result, '\n', sep='')
    for test in LOOSE_CITY_TESTS:
        test_result = is_loose_city(test)
        if not test_result:
            any_bad+=1
            print('bad test: "', test, '" not a loose city\n', sep='')
    print('****', any_bad, 'bad tests ****')

