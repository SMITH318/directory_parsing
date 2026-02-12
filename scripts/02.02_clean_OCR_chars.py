import re
import json
import unicodedata
from pathlib import Path


# simple mapping of rare/unusual -> common
REPL = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": ",",
    "—": "-", "–": "-", "−": "-",
    "…": "...",
    "\u00A0": " ",  # NBSP
    "\u200B": "", "\u200C": "", "\u200D": "", "\u200E": "", "\u200F": "",
    "·": ".", "•": "-", "•": "-", "×": "x",
    "°": "*",  "°": "*",
    "©": "@", "⋄": "*",
    "◆": "◊",
    "▼": "V", "▽": "V", "ṽ": "V", "Ṿ": "V", "Ṿ":"V", "Ý":"V", '¥':"V", "Ÿ":"V"
}

# compile regexes
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
MULTI_WS_RE = re.compile(r"[\s]+")
MULTI_DASH_RE = re.compile(r"[-–—]+")
L_YEAR_IN_RE = re.compile(r"\([lI1]?[ ']?(?P<year>[0-9]{2})\)")
L_T_RE = re.compile(r'\([lI1] ?[tf]\)')

def clean_text(s: str) -> str:
    # normalize unicode compatibility
    s = unicodedata.normalize("NFKC", s)
    # trim start/end whitespace
    s = s.strip()
    # direct replacements
    for a, b in REPL.items():
        if a in s:
            s = s.replace(a, b)
    # remove control chars (keep newline/tab handled outside JSON strings)
    s = CONTROL_RE.sub("", s)
    #replace all white space, including multiples with single space
    s = MULTI_WS_RE.sub(" ", s)
    #replace multiple dashes and dash-like characters with single dash
    s = MULTI_DASH_RE.sub("-", s)
    #replace colons with semi-colons
    s = re.sub(':', ';', s)
    #replace commas and underscores with periods
    #text=re.sub('[,_]', '.', text)
    #cleanup '(l'89)' format - often 'l' appears as 'I' or '1', space not always there
    s = L_YEAR_IN_RE.sub(r'(l \g<year>)', s)
    #cleanup '(l t)' format - often 'l' appears as 'I' or '1', space not always there, 't' sometimes f
    s = L_T_RE.sub(r'(l t)', s)
    # trim start/end whitespace (again)
    s = s.strip()
    return s

def clean_lines(filename_in, filename_out):
    def strip_str(string):
        return string.strip(' \n\t')# trim start/end whitespace as well as common OCR blips
    
    unexpected_chars =''
    with open(filename_in, 'r',encoding="utf-8") as f:
        with open(filename_out, 'w', newline='', encoding="utf-8") as out_file:
            for line in f:
                entry = json.loads(line)                
                text = clean_text(entry['text'])
                entry['text'] = text
                out_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                # what wierd characters haven't I dealt with?
                match = re.search(r"[^\da-zA-Z(),.'+*: &@◊-]", text) 
                if match:
                    unexpected_chars += match.group(0)
 
                #consider aggregating lines ending in '-' with next line (by not adding \n)
                #aggregate if character before '-' is lowercase letter and first character of next line is lowercase letter
                # if len(line) > 0 and len(text) > i and line[-1] == '-' and line[-2].isalpha() and line[-2].islower() and text[i+1][0].isalpha() and text[i+1][0].islower() :
                #     #print('aggregating "', line, '" and "', lines[i+1], '"')
                #     out_file.write(line[:-1])
                # else:
                #     out_file.write(line+'\n')
    unexpected_chars = set(unexpected_chars)
    if len(unexpected_chars) > 0:
        print(len(unexpected_chars),'unexpected characters found: ', repr(unexpected_chars))

# __main__
if True:
    # Setup project paths
    script_dir = Path(__file__).parent if '__file__' in dir() else Path.cwd()
    project_root = script_dir if (script_dir / "data").exists() else script_dir.parent

    input_file = project_root / "data" / "02_raw" / "ocr_output_reviewed.jsonl"
    output_file = project_root / "data" / "02_raw" / "ocr_output_auto_cleaned.jsonl"

    clean_lines(input_file, output_file)
else: #tests
    clean_lines('clean_tests.txt', 'clean_tests_out.txt')
    
