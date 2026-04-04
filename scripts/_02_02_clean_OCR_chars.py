import re
import json
import unicodedata
from pathlib import Path


# simple mapping of rare/unusual -> common
REPL = {
    # "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    # "‘": "'", "’": "'", "‚": ",",
    # "—": "-", "–": "-", "−": "-",
    # "…": "...",
    # "\u00A0": " ",  # NBSP
    # "\u200B": "", "\u200C": "", "\u200D": "", "\u200E": "", "\u200F": "",
    # "·": ".", "•": "-", "•": "-", "×": "x",
    # "°": "*",  "°": "*",
    # "©": "@", "⋄": "*",
    # "◆": "◊",
    # "▼": "V", "▽": "V", "ṽ": "V", "Ṿ": "V", "Ṿ":"V", "Ý":"V", '¥':"V", "Ÿ":"V"
    # 'δ': '♁'
}

# compile regexes
RE_MAP = {
    # remove control chars (keep newline/tab handled outside JSON strings)
    re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"): "",
    #replace all white space with a space
    re.compile(r"\s"): " ",
    #replace multiple dashes and dash-like characters with single dash
    re.compile(r"[-–—]"): "-",
    #replace any colons with semi-colons
    re.compile(':'): ';',
    #replace commas and underscores with periods
    #text=re.sub('[,_]', '.', text)
    #cleanup '(l'89)' format - often 'l' appears as 'I' or '1', space not always there
    re.compile(r"\([lI1][ ']?(?P<year>[0-9]{2})\)"): r'(l \g<year>)',
    #cleanup '(l †)' to '(l t)' format - often 'l' appears as 'I' or '1', space not always there, 't' sometimes f
    re.compile(r"\([lI1][ ']?[tf†]\)"): r'(l t)',
    #replace single character 1/2 with unbundled
    re.compile(r' ?½|(1⁄2)'): r" 1/2", # ½ causes problems when sent back to Gemini
    #reduce multiples of any non-word character (space, punctuation) to single
    re.compile(r'(?P<char>\W)\1+'): r'\g<char>',
    #replace (δ) or (♂) with (♁)
    re.compile(r'\([δ♂]\)'): r'(♁)',
    #replace(#) with (‡)
    re.compile(r'\(#\)'): r'(‡)',
}

def clean_text(s: str) -> str:
    # normalize unicode compatibility
    s = unicodedata.normalize("NFKC", s)
    # trim start/end whitespace
    s = s.strip()
    # direct replacements
    # for a, b in REPL.items():
    #     if a in s:
    #         s = s.replace(a, b)
    # regex replacements
    for a, b in RE_MAP.items():
        s = a.sub(b, s)

    # trim start/end whitespace (again)
    s = s.strip()
    return s

def get_unexpected_chars(s: str) -> str:
    match = re.search(r"[^\da-zA-Z(),.'▼★;‡* &◊⊕♁△/-]", s)
    return match.group(0) if match else ""

def has_unexpected_chars(s: str) -> bool:
    return get_unexpected_chars(s) != ""

def clean_lines(filename_in, filename_out):
    # def strip_str(string):
    #     return string.strip(' \n\t')# trim start/end whitespace as well as common OCR blips
    
    unexpected_chars =''
    with open(filename_in, 'r',encoding="utf-8") as f:
        with open(filename_out, 'w', newline='', encoding="utf-8") as out_file:
            for line in f:
                entry = json.loads(line)                
                text = clean_text(entry['text'])
                entry['text'] = text
                out_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                # what wierd characters haven't I dealt with?
                unexpected_chars += get_unexpected_chars(entry['text'])
 
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
    else:
        print("No expected characters!!")

# __main__
if __name__ == "__main__":
    if True:
        # Setup project paths
        script_dir = Path(__file__).parent if '__file__' in dir() else Path.cwd()
        project_root = script_dir if (script_dir / "data").exists() else script_dir.parent

        raw_folder = project_root / "data" / "02_raw_batch"

        input_file = raw_folder / "ocr_output_reviewed.jsonl"
        output_file = raw_folder / "ocr_output_auto_cleaned.jsonl"

        clean_lines(input_file, output_file)
    else: #tests
        # print("'"+re.compile(r'(?P<char>\W)\1+').sub(r'\g<char>', ".;;... ''.")+"'")
        clean_lines('clean_tests.txt', 'clean_tests_out.txt')
    
