from enum import Enum
import re

LineType = Enum('LineType', 'UNKNOWN, STATE, CITY, DOC_START, DOC_FULL')

# "STATE"
STATE_REGEX=r'(?P<state>[A-Z\s]+)'

# "City, XXX,XXX, County.
CITY_COUNTY_REGEX=r'(?P<city>[a-zA-Z\s\']+),\s?((?P<population>[0-9,]+?)|-),\s?(?P<county>[a-zA-Z().\s]+)'

#L*,*1*,*L. - at least one letter, followed by a comma, followed by a number, followed by a comma, followed by a letter (and an optional period)
LOOSE_CITY_COUNTY_REGEX=r'\w.+?,.+?\d.+?,.+?\w.+?'

#(b XXXX)
BIRTH_REGEX=r"(\(b['\s]?(?P<birth>[0-9]{2})\))"
# St.1,'01; OR *
SCHOOL_YEAR_REGEX=r'(?P<sch_year>[0-9]{2})'
SCHOOL_ID_REGEX=r'((?P<sch_id>[1-9lI][0-9]?)\s?,)'
SCHOOL_STATE_REG_EX=r'((?P<sch_state>[A-Z][a-z]{1,3}|[O0]|N\.\s?Y|D\.\s?C|N\.\s?H|S\.\s?C)\.)'
SCHOOL_REGEX=r'((' + SCHOOL_STATE_REG_EX + r'\s?('+ SCHOOL_ID_REGEX + r'|[*])\s?((\'\s?'+SCHOOL_YEAR_REGEX + r'\s?;)|[*]\s))|(?P<no_sch_info>[*]))'
#(l XX)
LICENSE_REGEX=r"(\([lI][\s'](?P<lic_year>([0-9]{2})|t)\)\s?;?)"
ADDRESS_REGEX=r'[A-Z1-9][\w\s.\'&/-]+?[A-Za-z][\w\s.\'&/-]+?' #has to start with cap or number and have at least one non-starting letter
HOUR_RANGE_REGEX=r'(([1-9][0-9]?(\s?:30)?[.-]\s?[1-9][0-9]?(\s?:30)?)|((until|after)\s?[1-9][0-9]?))'
HOURS_REGEX= r'(?P<hours>'+HOUR_RANGE_REGEX+r'([,;]\s?'+HOUR_RANGE_REGEX+'){0,2})'

# "LAST, FIRST [(col.)][BIRTH][+]-SCHOOL
DOC_ENTRY_REGEX_TO_SCHOOL=r'(?P<last_name>[a-zA-Z \'1]+),\s(?P<first_name>[a-zA-Z.\s]+)\s?((?P<col>\(col\.\))\s?)?('+BIRTH_REGEX+r'\s?)?(?P<break>([+*@]?|-?[EH]?)\s?-)\s?'+SCHOOL_REGEX+r'\s?'
# "LAST, FIRST [(col.)][BIRTH][+]-SCHOOL LICENSE [address][office, address][hours]
DOC_ENTRY_REGEX=DOC_ENTRY_REGEX_TO_SCHOOL+LICENSE_REGEX+r'?\s?'+r'(?P<no_practice>\(not in practice\)\s?)?((?P<addr>'+ADDRESS_REGEX+r');?)?(\s?office,\s(?P<office>'+ADDRESS_REGEX+r');?)?\s?'+HOURS_REGEX+r'?(?<!;$)'

def is_loose_city(line):
    match = re.search(r"[+*:;]", line) # shouldn't have these characters
    if match:
        return False
    return re.match(LOOSE_CITY_COUNTY_REGEX, line) is not None

def get_state(line):
    matched = re.match(STATE_REGEX, line)
    return matched.groupdict()

def get_city(line):
    matched = re.match(CITY_COUNTY_REGEX, line)
    return matched.groupdict()

def get_full_doctor(line):
    matched = re.fullmatch(DOC_ENTRY_REGEX, line)
##    if matched: 
##        print('matched: "', matched.group(0), '"', sep='')
##        print(matched.groupdict())
    return matched.groupdict()


def get_line_type(line, flag=0):
    if re.fullmatch(STATE_REGEX,line, flag):
        # is "STATE."
        return LineType.STATE
    matched = re.fullmatch(CITY_COUNTY_REGEX,line, flag)
    if matched:
        #is "City, XXX,XXX, County."
        #print(matched.groupdict())
        return LineType.CITY
    matched = re.fullmatch(DOC_ENTRY_REGEX, line, flag)
    if matched:
        #print('matched: "', matched.group(0), '"', sep='')
        #print(matched.groupdict())
        return LineType.DOC_FULL
    matched = re.match(DOC_ENTRY_REGEX_TO_SCHOOL,line, flag)
    if matched:
        # is "LAST, FIRST [(col.)][BIRTH][~|+]-"
        #print('matched: "', matched.group(0), '"', sep='')
        #print(matched.groupdict())
        return LineType.DOC_START
    
    return LineType.UNKNOWN


RUN_TESTS=__name__ == '__main__'
LICENSE_TESTS=[("(l'86)", '86'),
               ('(l 95)', '95'),
               ('(l 99)', '99'),
               ('(l t)', 't')]
HOURS_TESTS=['12-2','9-11,3-5', 'until 7', 'after 6', '9-11:30', '9-11:30,3-5','10-11:30, 2.4:30', '10-11:30, 4.5']
LINE_TYPE_TESTS = [
                    ('LOUISIANA', LineType.STATE),
                    ('West Monroe, 775, Ouncldtn', LineType.CITY),
                   ('Whitford, -, Winn', LineType.CITY),
                   ('New Orleans, 312,457, Orleans', LineType.CITY),
                   ("GAAR, J. ALBERT (b'80) - Tenn.8,'04;", LineType.DOC_START),
                   ("Aubrey, A. J. (col.) (b'73)-La.4,'99;", LineType.DOC_START),
                   ("SCOTT, W. S.-* (l'03)", LineType.DOC_FULL),
                   ("McKOWEN, EMMETT C. (b 62)-La.l,'86; (l'86); College and High Sts.; 12-2", LineType.DOC_FULL),
                   ("HENBY, EUGENE L. (b 74)+-La.1,'97;(l 95); Water St.; 9-11,3-5", LineType.DOC_FULL),
                   ("Aubrey, A. J. (col.) (b 73)-La.4,'99;(l 99)", LineType.DOC_FULL),
                   ("ARCHINARD, PAUL E.-La.1,'82; (l 82); 1219 N. Rampart St. ; office, 211 Camp St.; 12-2", LineType.DOC_FULL),
                   ("Polk, Wm. T. (b 78)-Tenn.5,'02; (l 06); office, 3d and Murray Sts.", LineType.DOC_FULL),
                   ("Jones, Jas. P. (col.)-Tenn.7,'93; (l 93)", LineType.DOC_FULL),
                   ("Jones, Fred R.-Ind. 7,'87; (l t)", LineType.DOC_FULL),
                   ("FONTAINE, BRYCE W. (b'77) + - Tex.2,'96; (l'01)", LineType.DOC_FULL),
                   ("EDWARDS, CLARENCE J.-Ky.2,'83 ; (I 83)", LineType.DOC_FULL),
                   ("Gcoffrion, Victor-Que.3,'01 ; (l 04)", LineType.DOC_FULL),
                   ("De Poincy, Edgar S.-La.*'81; (l 81); 1227 Esplanade Ave.", LineType.DOC_FULL),
                   ("Wailes, L. A.- Pa.2,* (l 61); 2128 Berlin", LineType.DOC_FULL),
                   ("Lines, Ezra A.-H-* (l 97); 1940 N. Rampart St.; 8-12", LineType.DOC_FULL),
                   ("Duperier, Douglas-Mich.1,'95; (l 95)", LineType.DOC_FULL),
                   ("MENVILLE, LEON J.-Md.9,'04; (l 04) ; 9- 11:30, 3-5", LineType.DOC_FULL),
                   ("Belden, Jas. W.-La.1,'88; 1403 Louisiana Ave.; office, 830 Canal St.; 1-4", LineType.DOC_FULL),
                   ("Danos, Joseph L. (b 80)-La.1,'03; (not in practice)", LineType.DOC_FULL),
                   ("Henry, Stewart L.-La.1,'66; (l 66); 908 Carrollton Ave.; 7-8, 12-2", LineType.DOC_FULL),
                   ("HOEFELD, ADOLPH O.-La.1,'01; (l 01); 830 Canal St.; 1-3", LineType.DOC_FULL),
                   ("Jordan, Harrison-Tenn.11,'02; (l 02); (not in practice)", LineType.DOC_FULL),
                   ("Layton, Thos. B.-La.1,'01; (l 01); 1420 Josephine St.", LineType.DOC_FULL),
                   ("Marks, L. H.- (l 06)", LineType.UNKNOWN), #school required
                   ("LITTELL, ROBT. M.-La.1 ,'94; (l 94)", LineType.DOC_FULL),
                   ("MANBOULES, J. P., Jr.- (l 99)", LineType.UNKNOWN), #school required
                   ("FOSSIER, A. EMILE (b 81) - La.1,'02; (l 02); 1215 Carrollton Ave.; 7-9", LineType.DOC_FULL), 
                   ("POSTELL, LAURENS T. (b 59)+-La.1,'82; (l 82); office, Holloway & Postell Drug Store; 3-5", LineType.DOC_FULL),
                   ("Smith, Temple B. (b'92)-Mo.27,'92; (l t); 664 7th St.; office, 8171/2 Ryan St.; 10-11:30, 2-4:30", LineType.DOC_FULL),
                   ("PAINE, RUFFIN B. (b'65) + - La.1,'88; (l 88) ; Lake and Coffee Ste.; 10-11:30, 4-5", LineType.DOC_FULL),
                   ("BRUNS, HENRY D. (b 59)+-Pa.2,'81; (l 81); 2308 Prytania St.; office, 211 Camp St.; 12-4", LineType.DOC_FULL),
                   ]
LOOSE_CITY_TESTS =['Arca11ia, 924, Bienville.',
                   'Amite, 1,G47, Tangipaltoa.',
                   'Allenton-n, 250, Bos.oier.'
                   ]


if RUN_TESTS:
    any_bad=0
    for test, result in LICENSE_TESTS:
        matched = re.fullmatch(LICENSE_REGEX, test)
        if matched is None or matched.group('lic_year') != result:
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

